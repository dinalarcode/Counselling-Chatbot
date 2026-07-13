"""
RAGAS Evaluation for Biblical Counseling Chatbot RAG Pipeline.

Scenario: original/generic RAGAS paper metrics (Es et al., 2024) —
Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall. Mirrors the
thesis proposal's original evaluation plan. See `ragas_manager/ragas_engine.py` for the
LABAN-specific AspectCritic scenario (adversarial cases, stage-compliance
judging) — kept as a separate, independent scenario, not replaced by this
script.

Two-phase workflow:
  Phase 1 (--generate): Sample a standard testset (PAPER_SAMPLE_SIZE=100 rows)
                        from data/dataset_qna.csv, balanced across the 6
                        counseling stages per STAGE_QUOTA (pembukaan capped
                        small — redundant greetings — remaining stages get
                        the bulk of the quota, shortfall in scarce stages
                        backfilled into pembahasan), no adversarial injection.
                        `reference` is auto-filled from the QnA dataset's
                        own `answer` column — no manual authoring step,
                        since this scenario measures standard pipeline
                        performance against the corpus's own ground truth.
  Phase 2 (--evaluate): Run RAG pipeline on each sample, run mandatory
                        pre-flight data + metric validation (fail fast, zero
                        API cost), then evaluate with the four generic RAGAS
                        metrics using OpenAI as judge (via
                        langchain_openai.ChatOpenAI + OpenAIEmbeddings).
                        Generator is configured via opt.CHATBOT_LLM_PROVIDER.

Usage:
  python evaluation/eval_ragas_paper.py --generate
  python evaluation/eval_ragas_paper.py --evaluate

Output files (Phase 2):
  evaluation/results/ragas_paper_metrics/ragas_per_sample.csv
  evaluation/results/ragas_paper_metrics/ragas_summary.csv
  evaluation/results/ragas_paper_metrics/ragas_per_stage.csv
"""

import os
import sys
import re
import random
import argparse

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Path setup — allow running from project root
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import opt

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
QNA_CSV = os.path.join(PROJECT_ROOT, "data", "dataset_qna.csv")
TESTSET_DIR = os.path.join(SCRIPT_DIR, "data")
TESTSET_CSV = os.path.join(TESTSET_DIR, "ragas_testset_paper.csv")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results", "ragas_paper_metrics")
RANDOM_SEED = 42

# Standard testset size for this scenario — intentionally decoupled from
# opt.RAGAS_TESTSET_SIZE (which sizes the AspectCritic scenario in
# ragas_manager/ragas_engine.py), but matched to the same 100-sample scope previously
# established for that scenario.
PAPER_SAMPLE_SIZE = 100

# Stages where Bible verse retrieval is active
BIBLE_VERSE_STAGES = frozenset({"solusi", "relaksasi"})

# All 6 counseling stages, in pipeline order
STAGES = ["pembukaan", "pembahasan", "intervensi", "solusi", "relaksasi", "penutupan"]

# Per-stage sampling target for Phase 1. `pembukaan` is capped small (mostly
# redundant greetings, see STAGE_MAX_CAPS below). Actual yield is capped by
# corpus availability per stage (see balance_by_stage) — `intervensi` in
# particular is scarce in the raw QnA corpus, so its shortfall is backfilled
# into `pembahasan` (which has ample availability) automatically.
STAGE_QUOTA = {
    "pembukaan": 5,
    "pembahasan": 40,
    "intervensi": 15,
    "solusi": 20,
    "relaksasi": 10,
    "penutupan": 10,
}
assert sum(STAGE_QUOTA.values()) == PAPER_SAMPLE_SIZE, (
    "STAGE_QUOTA must sum to PAPER_SAMPLE_SIZE — update both together."
)

# Hard ceiling that also binds during backfill (unlike STAGE_QUOTA, which is
# just a starting request that spare capacity can grow past). Only
# `pembukaan` needs this: its greetings are redundant enough that we never
# want it absorbing leftover quota from scarcer stages like `intervensi`.
STAGE_MAX_CAPS = {"pembukaan": 5}

# Judge embedding model — required by AnswerRelevancy (embedding-based
# semantic similarity between the generated reverse-question and user_input).
JUDGE_EMBEDDING_MODEL = "text-embedding-3-small"


def balance_by_stage(
    df: pd.DataFrame,
    total: int,
    seed: int = RANDOM_SEED,
    quota_override: dict = None,
    max_caps: dict = None,
) -> pd.DataFrame:
    """
    Draw `total` rows from df spread across the 6 counseling stages.

    Without `quota_override`: split as evenly as possible (round-robin,
    scarce stages max out at their full availability, surplus stages absorb
    the remainder). With `quota_override`: start from those per-stage
    targets instead of an even split, still capped by availability, with any
    shortfall backfilled round-robin into stages under their own cap
    (`max_caps`, defaulting to full availability). Stages absent from df are
    skipped with a warning rather than crashing.
    """
    groups = {s: df[df["stage"] == s] for s in STAGES}
    present = {s: g for s, g in groups.items() if len(g) > 0}
    missing = [s for s in STAGES if s not in present]
    if missing:
        print(f"  [WARN] No samples for stage(s): {missing} - excluded from balanced set.")

    max_caps = max_caps or {}

    if quota_override is not None:
        quota = {
            s: min(quota_override.get(s, 0), len(present[s]), max_caps.get(s, len(present[s])))
            for s in present
        }
        remaining = total - sum(quota.values())
        while remaining > 0:
            spare = [
                s for s in present
                if quota[s] < min(len(present[s]), max_caps.get(s, len(present[s])))
            ]
            if not spare:
                break
            for stage in spare:
                if remaining <= 0:
                    break
                quota[stage] += 1
                remaining -= 1
    else:
        total = min(total, len(df))
        quota = dict.fromkeys(present, 0)
        while sum(quota.values()) < total:
            spare = [s for s in present if quota[s] < len(present[s])]
            if not spare:
                break
            for stage in spare:
                if sum(quota.values()) >= total:
                    break
                quota[stage] += 1

    balanced = (
        pd.concat([present[s].sample(n=n, random_state=seed) for s, n in quota.items() if n])
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )

    print(f"\n  Balanced stage distribution:")
    for stage in STAGES:
        avail = len(present.get(stage, []))
        capped = " (all available)" if quota.get(stage, 0) == avail and avail else ""
        print(f"    {stage}: {quota.get(stage, 0)} / {avail} available{capped}")
    print(f"  Balanced total: {len(balanced)} samples")

    return balanced


# =========================================================================
#  PHASE 1: Generate test template
# =========================================================================

def classify_stage(question: str) -> str:
    """
    Heuristic stage assignment based on question content patterns.
    The user can manually adjust in the CSV before running Phase 2.
    """
    q = question.strip().lower()

    # --- pembukaan: greetings ---
    if q in ("halo", "hai", "hi"):
        return "pembukaan"

    # --- penutupan: closure signals ---
    penutupan_patterns = [
        r"terima\s*kasih.*kak.*akan\s*(coba|mencoba)",
        r"terima\s*kasih.*akan\s*coba",
        r"aku\s*merasa\s*lebih\s*lega.*terima\s*kasih",
        r"terima\s*kasih.*sudah\s*mendengarkan",
        r"aku\s*akan\s*coba\s*semuanya",
        r"aku\s*bakal\s*coba.*terima\s*kasih",
    ]
    for pat in penutupan_patterns:
        if re.search(pat, q):
            return "penutupan"

    # --- relaksasi: relaxation/gratitude signals ---
    relaksasi_patterns = [
        r"lebih\s*tenang",
        r"lebih\s*nyaman",
        r"jadi\s*lebih\s*tenang",
        r"saya\s*merasa\s*lebih\s*baik",
        r"rasanya\s*lebih\s*enak",
        r"amin",
    ]
    for pat in relaksasi_patterns:
        if re.search(pat, q):
            return "relaksasi"

    # --- solusi: action/agreement signals ---
    solusi_patterns = [
        r"(saya|aku)\s*(akan\s*)?(coba|mencoba)",
        r"oke.*bisa\s*dicoba",
        r"bisa\s*dicoba",
        r"itu\s*masuk\s*akal",
        r"itu\s*bisa\s*dicoba",
        r"mungkin\s*bisa",
        r"baik.*akan\s*(coba|mencoba)",
        r"aku\s*mau\s*coba",
    ]
    for pat in solusi_patterns:
        if re.search(pat, q):
            return "solusi"

    # --- intervensi: acknowledgment/understanding ---
    intervensi_patterns = [
        r"(saya|aku)\s*mengerti",
        r"(saya|aku)\s*paham",
        r"benar\s*juga",
        r"masuk\s*akal",
        r"setuju",
        r"oh\s*iya",
        r"memang\s*benar",
    ]
    for pat in intervensi_patterns:
        if re.search(pat, q):
            return "intervensi"

    # --- pembahasan: everything else (problem-sharing, emotional content) ---
    return "pembahasan"


def generate_testset():
    """Phase 1: Sample QnA pairs and auto-fill reference from the QnA `answer` column."""
    print("=" * 65)
    print("  [SCENARIO: RAGAS Paper Metrics] Test Set Generator (Phase 1)")
    print("=" * 65)

    # Load QnA dataset
    if not os.path.exists(QNA_CSV):
        print(f"  [ERROR] QnA dataset not found: {QNA_CSV}")
        sys.exit(1)

    df = pd.read_csv(QNA_CSV)
    print(f"\n  QnA dataset loaded: {len(df)} rows")

    # Clean up
    df = df.dropna(subset=["question", "answer"])
    df["question"] = df["question"].astype(str).str.strip()
    df["answer"] = df["answer"].astype(str).str.strip()
    df = df[(df["question"].str.len() > 0) & (df["answer"].str.len() > 0)]
    print(f"  After cleanup: {len(df)} valid rows")

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Classify the WHOLE corpus before sampling. Sampling first would inherit
    # the corpus's natural skew (pembahasan dominates ~80% of the QnA rows).
    df["stage"] = df["question"].apply(classify_stage)
    print(f"\n  Corpus stage distribution (all {len(df)} rows):")
    for stage in STAGES:
        count = int((df["stage"] == stage).sum())
        marker = " (Bible verse)" if stage in BIBLE_VERSE_STAGES else ""
        print(f"    {stage}: {count}{marker}")

    print(f"\n  [INFO] Testset size is strictly set to {PAPER_SAMPLE_SIZE} samples "
          f"(standard scenario — no adversarial injection).")
    sample_df = balance_by_stage(df, total=PAPER_SAMPLE_SIZE, quota_override=STAGE_QUOTA, max_caps=STAGE_MAX_CAPS)

    # Build output DataFrame — reference is auto-filled from the QnA
    # dataset's own `answer` column (the ground-truth counselor response).
    out_df = pd.DataFrame({
        "question": sample_df["question"].values,
        "stage": sample_df["stage"].values,
        "reference": sample_df["answer"].values,
        "intent_override": "",  # Optional - user can fill for Bible verse stages
    })

    # Pre-fill intent_override hint for Bible verse stages
    for idx in out_df.index:
        if out_df.at[idx, "stage"] in BIBLE_VERSE_STAGES:
            out_df.at[idx, "intent_override"] = (
                "Menyatakan Perasaan Sedih dan Kehilangan"  # placeholder
            )

    # Save
    os.makedirs(TESTSET_DIR, exist_ok=True)
    out_df.to_csv(TESTSET_CSV, index=False, encoding="utf-8-sig")
    print(f"\n  [OK] Saved: {TESTSET_CSV}")

    print(f"\n{'=' * 65}")
    print("  NEXT STEPS:")
    print("  1. (Optional) Review/adjust 'reference' — auto-filled from")
    print("     dataset_qna.csv's 'answer' column, no manual authoring required")
    print("  2. Adjust 'stage' column if auto-assignment is incorrect")
    print("  3. Adjust 'intent_override' for Bible verse stages")
    print("  4. Run: python evaluation/eval_ragas_paper.py --evaluate")
    print(f"{'=' * 65}")


# =========================================================================
#  PHASE 2: Run RAG + RAGAS evaluation
# =========================================================================

def validate_rag_results(results: list) -> None:
    """
    Mandatory pre-flight DATA validation — runs after the RAG pipeline but
    strictly BEFORE any RAGAS/judge API call. Catches formatting bugs
    locally so a broken run fails for free instead of after N billed judge
    requests. Column mapping to the classic RAGAS schema: question ->
    question, answer -> response, contexts -> retrieved_contexts,
    ground_truth -> reference.
    """
    df = pd.DataFrame(results)

    required_cols = ["question", "response", "retrieved_contexts", "reference"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        print(f"  [PREFLIGHT FAIL] Missing required column(s): {missing_cols}")
        sys.exit(1)

    for col in ["question", "response", "reference"]:
        is_null = df[col].isna() | (df[col].astype(str).str.strip() == "")
        if is_null.any():
            bad_idx = df.index[is_null].tolist()
            print(f"  [PREFLIGHT FAIL] {int(is_null.sum())} null/empty value(s) in "
                  f"'{col}' — rows: {bad_idx[:10]}")
            sys.exit(1)

    bad_ctx_idx = [
        i for i, ctx in enumerate(df["retrieved_contexts"])
        if not isinstance(ctx, list) or len(ctx) == 0
        or not all(isinstance(c, str) and c.strip() for c in ctx)
    ]
    if bad_ctx_idx:
        print(f"  [PREFLIGHT FAIL] 'retrieved_contexts' must be a non-empty list[str] "
              f"(not plain str/float) — bad rows: {bad_ctx_idx[:10]}")
        sys.exit(1)

    failed_rag = df.index[df["response"].astype(str).str.startswith("ERROR:")].tolist()
    if failed_rag:
        print(f"  [PREFLIGHT FAIL] {len(failed_rag)} sample(s) failed RAG generation "
              f"(response starts with 'ERROR:') — rows: {failed_rag[:10]}")
        sys.exit(1)

    print(f"  [OK] Dataset validation passed — {len(df)} rows, columns={required_cols}, "
          f"no nulls, retrieved_contexts is list[str], no RAG failures")


def validate_metric_bindings(metrics: list) -> None:
    """
    Mandatory pre-flight METRIC validation — asserts every metric has its
    llm (and embeddings, where required) bound BEFORE evaluate() is called.
    """
    from ragas.metrics.base import MetricWithLLM, MetricWithEmbeddings

    for m in metrics:
        if isinstance(m, MetricWithLLM) and m.llm is None:
            print(f"  [PREFLIGHT FAIL] Metric '{m.name}' has no llm bound")
            sys.exit(1)
        if isinstance(m, MetricWithEmbeddings) and m.embeddings is None:
            print(f"  [PREFLIGHT FAIL] Metric '{m.name}' has no embeddings bound")
            sys.exit(1)

    print(f"  [OK] Metric wiring validated — {len(metrics)} metric(s), "
          f"all required llm/embeddings bound")


def run_evaluation():
    """Phase 2: Run RAG pipeline on each sample and evaluate with generic RAGAS metrics."""
    from dotenv import load_dotenv
    load_dotenv()

    print("=" * 65)
    print("  [SCENARIO: RAGAS Paper Metrics] Evaluation (Phase 2)")
    print("=" * 65)

    # -- 1. Validate test set -----------------------------------------------
    if not os.path.exists(TESTSET_CSV):
        print(f"  [ERROR] Test set not found: {TESTSET_CSV}")
        print("  Run 'python evaluation/eval_ragas_paper.py --generate' first.")
        sys.exit(1)

    df = pd.read_csv(TESTSET_CSV)
    print(f"\n  Test set loaded: {len(df)} samples")

    # Phase 1 already balances the template; this is a safety net for
    # hand-edited CSVs and a no-op on an already-balanced one.
    df = balance_by_stage(df, total=len(df))

    # Check for empty references (safety net — Phase 1 auto-fills these)
    empty_refs = df["reference"].isna() | (df["reference"].astype(str).str.strip() == "")
    if empty_refs.any():
        n_empty = empty_refs.sum()
        print(f"  [ERROR] {n_empty} samples have empty 'reference' column.")
        empty_indices = df[empty_refs].index.tolist()
        print(f"  Empty rows: {empty_indices[:10]}{'...' if len(empty_indices) > 10 else ''}")
        sys.exit(1)

    # -- 2. Initialize RAG Engine -------------------------------------------
    print(f"\n  Initializing RAG Engine...")
    from core.rag_engine.rag_engine import RAGEngine
    rag_engine = RAGEngine()
    print("  [OK] RAG Engine initialized")

    # -- 3. Run RAG pipeline on each sample ---------------------------------
    print(f"\n  Running RAG pipeline on {len(df)} samples...")
    results = []

    for i, row in df.iterrows():
        question = str(row["question"]).strip()
        stage = str(row["stage"]).strip()
        reference = str(row["reference"]).strip()
        intent_override_raw = str(row.get("intent_override", "")).strip()

        # Parse intent overrides
        override_intents = None
        if intent_override_raw and intent_override_raw != "nan":
            override_intents = [
                s.strip() for s in intent_override_raw.split(",") if s.strip()
            ]

        # Run RAG
        try:
            rag_result = rag_engine.generate_response(
                user_input=question,
                current_stage=stage,
                override_intents=override_intents,
                # RAGAS evaluates the verse-retrieval pipeline itself, so verses
                # must always be retrieved regardless of the runtime consent gate.
                spiritual_consent=True,
            )

            response = rag_result["response"]
            ctx = rag_result["context_used"]

            # Collect retrieved contexts
            retrieved_contexts = []
            if ctx.get("example_answer"):
                retrieved_contexts.append(ctx["example_answer"])
            if ctx.get("bible_verses"):
                retrieved_contexts.append(ctx["bible_verses"])

            # Fallback: if no context was retrieved, add a placeholder
            if not retrieved_contexts:
                retrieved_contexts = ["(no context retrieved)"]

            results.append({
                "question": question,
                "stage": stage,
                "reference": reference,
                "response": response,
                "retrieved_contexts": retrieved_contexts,
            })

            print(f"    [{i+1}/{len(df)}] stage={stage} | "
                  f"ctx_count={len(retrieved_contexts)} | "
                  f"resp_len={len(response)}")

        except Exception as e:
            print(f"    [{i+1}/{len(df)}] [ERROR] {e}")
            results.append({
                "question": question,
                "stage": stage,
                "reference": reference,
                "response": f"ERROR: {e}",
                "retrieved_contexts": ["(error)"],
            })

    print(f"  [OK] RAG pipeline completed for {len(results)} samples")

    # -- 3b. Pre-flight DATA validation (zero API cost, fail fast) ----------
    print(f"\n  Running pre-flight data validation (no API calls)...")
    validate_rag_results(results)

    # -- 4. Build RAGAS dataset ---------------------------------------------
    print(f"\n  Building RAGAS evaluation dataset...")

    from ragas import evaluate, EvaluationDataset, SingleTurnSample

    samples = [
        SingleTurnSample(
            user_input=r["question"],
            response=r["response"],
            retrieved_contexts=r["retrieved_contexts"],
            reference=r["reference"],
        )
        for r in results
    ]

    eval_dataset = EvaluationDataset(samples=samples)
    print(f"  [OK] EvaluationDataset built with {len(samples)} samples")

    # -- 5. Configure OpenAI LLM judge + embeddings (langchain_openai) ------
    print(f"\n  Configuring OpenAI LLM judge + embeddings via langchain_openai...")

    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    # Guard: ensure OPENAI_API_KEY is present before constructing the client.
    # dotenv has already been loaded above.
    openai_api_key = os.getenv(opt.RAGAS_JUDGE_API_KEY_ENV)
    if not openai_api_key:
        print(f"  [ERROR] {opt.RAGAS_JUDGE_API_KEY_ENV} not found in .env")
        sys.exit(1)

    ragas_judge_llm = ChatOpenAI(
        model=opt.RAGAS_JUDGE_MODEL,
        api_key=openai_api_key,
        temperature=opt.RAGAS_JUDGE_TEMPERATURE,
    )
    ragas_judge_embeddings = OpenAIEmbeddings(
        model=JUDGE_EMBEDDING_MODEL,
        api_key=openai_api_key,
    )
    print(f"  [OK] OpenAI judge configured ({opt.RAGAS_JUDGE_MODEL}, embeddings={JUDGE_EMBEDDING_MODEL})")

    # -- 6. Define metrics --------------------------------------------------
    # Original/generic RAGAS paper metrics (Es et al., 2024). Judge LLM and
    # embeddings are bound explicitly per metric rather than relying on
    # evaluate()'s auto-injection, per this scenario's isolation requirement.
    from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall

    metrics = [
        Faithfulness(llm=ragas_judge_llm),
        AnswerRelevancy(llm=ragas_judge_llm, embeddings=ragas_judge_embeddings),
        ContextPrecision(llm=ragas_judge_llm),
        ContextRecall(llm=ragas_judge_llm),
    ]
    metric_names = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    print(f"  Metrics: {metric_names}")

    # -- 7. Pre-flight METRIC validation (zero API cost, fail fast) ---------
    # Validate metric/LLM/embeddings wiring locally before spending any API
    # calls — combines our own explicit llm/embeddings-bound assertions with
    # ragas's internal validators, so a broken refactor fails immediately
    # instead of after N billed judge calls.
    print(f"\n  Running pre-flight metric validation (no API calls)...")

    from langchain_core.language_models import BaseLanguageModel as LangchainLLM
    from ragas.metrics.base import Metric
    from ragas.validation import validate_required_columns, validate_supported_metrics

    bad_metrics = [m for m in metrics if not isinstance(m, Metric)]
    if bad_metrics:
        print(f"  [PREFLIGHT FAIL] Not valid ragas Metric objects: {bad_metrics}")
        sys.exit(1)
    if not isinstance(ragas_judge_llm, LangchainLLM):
        print(f"  [PREFLIGHT FAIL] judge llm is not a Langchain BaseLanguageModel: {type(ragas_judge_llm)}")
        sys.exit(1)

    validate_metric_bindings(metrics)

    try:
        validate_required_columns(eval_dataset, metrics)
        validate_supported_metrics(eval_dataset, metrics)
    except ValueError as e:
        print(f"  [PREFLIGHT FAIL] {e}")
        sys.exit(1)

    print("  [OK] All pre-flight checks passed (no API calls made) — proceeding to evaluate()")

    # -- 8. Run RAGAS evaluation --------------------------------------------
    print(f"\n  Running RAGAS evaluation (this may take a few minutes)...")

    import traceback

    try:
        ragas_results = evaluate(
            dataset=eval_dataset,
            metrics=metrics,
            llm=ragas_judge_llm,
            embeddings=ragas_judge_embeddings,
        )
    except Exception as e:
        print(f"  [ERROR] RAGAS evaluation failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    print(f"  [OK] RAGAS evaluation completed")

    # -- 9. Save results ----------------------------------------------------
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Convert to DataFrame
    results_df = ragas_results.to_pandas()

    # Add stage and metadata columns
    stages = [r["stage"] for r in results]
    results_df.insert(0, "stage", stages)
    results_df.insert(0, "question", [r["question"] for r in results])

    # Per-sample CSV
    per_sample_path = os.path.join(RESULTS_DIR, "ragas_per_sample.csv")
    results_df.to_csv(per_sample_path, index=False, encoding="utf-8-sig")
    print(f"\n  [OK] Saved: {per_sample_path}")

    # Summary CSV (aggregate across all samples)
    summary_data = {}
    for col in metric_names:
        if col in results_df.columns:
            values = results_df[col].dropna()
            summary_data[col] = {
                "mean": values.mean(),
                "median": values.median(),
                "std": values.std(),
                "min": values.min(),
                "max": values.max(),
                "count": len(values),
            }

    summary_df = pd.DataFrame(summary_data).T
    summary_df.index.name = "metric"
    summary_path = os.path.join(RESULTS_DIR, "ragas_summary.csv")
    summary_df.to_csv(summary_path, encoding="utf-8-sig")
    print(f"  [OK] Saved: {summary_path}")

    # Per-stage CSV
    stage_metrics = []
    for stage in sorted(results_df["stage"].unique()):
        stage_mask = results_df["stage"] == stage
        stage_row = {"stage": stage, "sample_count": stage_mask.sum()}
        for col in metric_names:
            if col in results_df.columns:
                values = results_df.loc[stage_mask, col].dropna()
                stage_row[f"{col}_mean"] = values.mean() if len(values) > 0 else None
                stage_row[f"{col}_std"] = values.std() if len(values) > 0 else None
        stage_metrics.append(stage_row)

    stage_df = pd.DataFrame(stage_metrics)
    stage_path = os.path.join(RESULTS_DIR, "ragas_per_stage.csv")
    stage_df.to_csv(stage_path, index=False, encoding="utf-8-sig")
    print(f"  [OK] Saved: {stage_path}")

    # -- 10. Console summary -------------------------------------------------
    print(f"\n{'=' * 65}")
    print("  RAGAS PAPER METRICS EVALUATION SUMMARY")
    print(f"{'=' * 65}")
    print(f"  Total samples: {len(results_df)}")
    print(f"  Judge LLM: OpenAI {opt.RAGAS_JUDGE_MODEL} | Embeddings: {JUDGE_EMBEDDING_MODEL}")
    print(f"\n  Overall Scores:")

    for col in metric_names:
        if col in results_df.columns:
            val = results_df[col].dropna()
            print(f"    {col:30s}: {val.mean():.4f} "
                  f"(+/- {val.std():.4f})")

    print(f"\n  Per-Stage Breakdown:")
    for _, row in stage_df.iterrows():
        stage_name = row["stage"]
        n = int(row["sample_count"])
        print(f"\n    {stage_name} (n={n}):")
        for col in metric_names:
            mean_col = f"{col}_mean"
            if mean_col in row and pd.notna(row[mean_col]):
                print(f"      {col:28s}: {row[mean_col]:.4f}")

    print(f"\n  Output files:")
    print(f"    {per_sample_path}")
    print(f"    {summary_path}")
    print(f"    {stage_path}")
    print(f"{'=' * 65}")


# =========================================================================
#  Main
# =========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RAGAS Paper-Metrics Evaluation for Biblical Counseling Chatbot"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--generate",
        action="store_true",
        help=f"Phase 1: Generate test template CSV ({PAPER_SAMPLE_SIZE} samples, reference auto-filled)",
    )
    group.add_argument(
        "--evaluate",
        action="store_true",
        help="Phase 2: Run RAG pipeline + RAGAS evaluation (Faithfulness/AnswerRelevancy/ContextPrecision/ContextRecall)",
    )

    args = parser.parse_args()

    if args.generate:
        generate_testset()
    elif args.evaluate:
        run_evaluation()
