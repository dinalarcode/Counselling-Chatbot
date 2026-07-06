"""
LABAN Zero-Shot Evaluation — Pure Cosine Similarity (No Re-Training)
=====================================================================
Loads the pretrained checkpoint, extracts raw pooled embeddings from
the dual-encoder backbone, and measures seen vs unseen F1 via cosine-
similarity thresholding — bypasses the gram-matrix logit head entirely.

Splits:  3 × (7 seen / 3 unseen), same label sets as the old script.
Output:
    evaluation/results/seen_unseen_summary.csv
    evaluation/results/seen_unseen_per_intent.csv
    evaluation/results/seen_unseen_aggregate.csv
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, precision_score, recall_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from config import opt
from models.multilabel.bert_model import BertEmbedding

RESULTS_DIR  = os.path.join(ROOT, "evaluation", "results")
CHECKPOINT   = os.path.join(ROOT, "checkpoint", "IndoBERT_multi_label_zsl.pt")
DATASET_CSV  = os.path.join(ROOT, "data", "augmentation", "dataset_multiintent_augmented.csv")
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN      = opt.max_len
BATCH_SIZE   = 32
RANDOM_SEED  = 42
THRESHOLD    = opt.thresold

os.makedirs(RESULTS_DIR, exist_ok=True)

UNSEEN_SPLITS = [
    {"name": "Split-A", "unseen": [
        "Mengisyaratkan Butuh Bantuan Profesional",
        "Mengisyaratkan Gejala Fisik",
        "Menyatakan Reaksi Terkejut dan Tidak Terduga",
    ]},
    {"name": "Split-B", "unseen": [
        "Menyatakan Perasaan Benci dan Jijik",
        "Menyatakan Rasa Syukur dan Apresiasi",
        "Menyatakan Perasaan Sebelum Menghadapi Kejadian",
    ]},
    {"name": "Split-C", "unseen": [
        "Menyatakan Perasaan Marah dan Frustasi",
        "Menyatakan Perasaan Percaya",
        "Menyatakan Perasaan Sedih dan Kehilangan",
    ]},
]


def load_model():
    model = BertEmbedding()
    state = torch.load(CHECKPOINT, map_location=DEVICE)
    model.load_state_dict(state, strict=False)
    model.to(DEVICE).eval()
    return model


def encode_texts(sub_model, tokenizer, texts):
    """Return (N, hidden_size) pooled embedding tensor."""
    all_embs = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        enc = tokenizer(
            batch, padding=True, truncation=True,
            max_length=MAX_LEN, return_tensors="pt"
        )
        ids  = enc["input_ids"].to(DEVICE)
        mask = enc["attention_mask"].to(DEVICE)
        with torch.no_grad():
            out = sub_model(input_ids=ids, attention_mask=mask, return_dict=True)
            emb = out.pooler_output
        all_embs.append(emb.cpu())
    return torch.cat(all_embs, dim=0)  # (N, 768)


def _scores(targets, preds, idx):
    t, p = targets[:, idx], preds[:, idx]
    return {
        "f1": round(f1_score(t, p, average="macro", zero_division=0), 4),
        "p":  round(precision_score(t, p, average="macro", zero_division=0), 4),
        "r":  round(recall_score(t, p, average="macro", zero_division=0), 4),
    }


def evaluate_split(split_name, unseen_intents, targets, intent_list, utt_embs, label_embs):
    seen_intents = [i for i in intent_list if i not in unseen_intents]

    # pure cosine similarity: (N, 10)
    sim  = (F.normalize(utt_embs, dim=1) @ F.normalize(label_embs, dim=1).T).numpy()
    preds = (sim >= THRESHOLD).astype(int)

    all_idx    = list(range(len(intent_list)))
    seen_idx   = [intent_list.index(i) for i in seen_intents]
    unseen_idx = [intent_list.index(i) for i in unseen_intents]

    m_all    = _scores(targets, preds, all_idx)
    m_seen   = _scores(targets, preds, seen_idx)
    m_unseen = _scores(targets, preds, unseen_idx)

    per_intent = {}
    for idx, intent in enumerate(intent_list):
        t_col, p_col = targets[:, idx], preds[:, idx]
        per_intent[intent] = {
            "f1":       round(f1_score(t_col, p_col, average="binary", zero_division=0), 4),
            "precision": round(precision_score(t_col, p_col, average="binary", zero_division=0), 4),
            "recall":   round(recall_score(t_col, p_col, average="binary", zero_division=0), 4),
            "support":  int(t_col.sum()),
            "group":    "unseen" if intent in unseen_intents else "seen",
        }

    print(f"\n  ── {split_name} ──")
    print(f"  F1-Macro All:    {m_all['f1']:.4f}")
    print(f"  F1-Macro Seen:   {m_seen['f1']:.4f}")
    print(f"  F1-Macro Unseen: {m_unseen['f1']:.4f}")
    print("  Per-intent:")
    for intent, m in per_intent.items():
        tag = "[UNSEEN]" if m["group"] == "unseen" else "[seen]  "
        print(f"    {tag} F1={m['f1']:.4f}  P={m['precision']:.4f}"
              f"  R={m['recall']:.4f}  sup={m['support']:3d}  {intent}")

    return {
        "split": split_name,
        "seen_intents":   "; ".join(seen_intents),
        "unseen_intents": "; ".join(unseen_intents),
        "f1_macro_all":           m_all["f1"],    "precision_macro_all":    m_all["p"],    "recall_macro_all":    m_all["r"],
        "f1_macro_seen":          m_seen["f1"],   "precision_macro_seen":   m_seen["p"],   "recall_macro_seen":   m_seen["r"],
        "f1_macro_unseen":        m_unseen["f1"], "precision_macro_unseen": m_unseen["p"], "recall_macro_unseen": m_unseen["r"],
    }, per_intent


def main():
    print("=" * 65)
    print("  LABAN ZSL — Pure Cosine Similarity Evaluation")
    print(f"  Model:     {opt.MODEL_NAME}")
    print(f"  Threshold: {THRESHOLD}")
    print(f"  Device:    {DEVICE}")
    print("=" * 65)

    df = pd.read_csv(DATASET_CSV, encoding="utf-8-sig")
    df["intent_list"] = df["Intent"].apply(lambda x: x.split("; "))
    mlb = MultiLabelBinarizer()
    encoded = mlb.fit_transform(df["intent_list"])
    intent_list = list(mlb.classes_)
    print(f"\nIntents ({len(intent_list)}): {intent_list}")

    # 20% held-out test set (same seed keeps split consistent with original script)
    _, df_test, _, test_enc = train_test_split(
        df, encoded, test_size=0.2, random_state=RANDOM_SEED
    )
    print(f"Test samples: {len(df_test)}")

    print(f"\nLoading checkpoint: {CHECKPOINT}")
    model     = load_model()
    tokenizer = AutoTokenizer.from_pretrained(opt.MODEL_NAME)

    # Encode once — reused across all splits
    print("Encoding utterances (utterance encoder)...")
    utt_embs = encode_texts(model.bert, tokenizer, df_test["question"].tolist())

    print("Encoding intent labels (label encoder)...")
    label_embs = encode_texts(model.bertlabelencoder, tokenizer, intent_list)

    targets = np.array(test_enc)

    summary_rows, per_intent_rows = [], []
    for split_config in UNSEEN_SPLITS:
        split_name = split_config["name"]
        unseen     = split_config["unseen"]
        for u in unseen:
            assert u in intent_list, f"Intent tidak ditemukan: {u}"

        result, per_intent = evaluate_split(
            split_name, unseen, targets, intent_list, utt_embs, label_embs
        )
        summary_rows.append(result)
        for intent_name, m in per_intent.items():
            per_intent_rows.append({
                "split": split_name, "intent": intent_name, "group": m["group"],
                "f1": m["f1"], "precision": m["precision"],
                "recall": m["recall"], "support": m["support"],
            })

    df_summary    = pd.DataFrame(summary_rows)
    df_per_intent = pd.DataFrame(per_intent_rows)

    summary_csv    = os.path.join(RESULTS_DIR, "seen_unseen_summary.csv")
    per_intent_csv = os.path.join(RESULTS_DIR, "seen_unseen_per_intent.csv")
    df_summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    df_per_intent.to_csv(per_intent_csv, index=False, encoding="utf-8-sig")

    agg = {
        "mean_f1_macro_all":    round(df_summary["f1_macro_all"].mean(), 4),
        "std_f1_macro_all":     round(df_summary["f1_macro_all"].std(), 4),
        "mean_f1_macro_seen":   round(df_summary["f1_macro_seen"].mean(), 4),
        "std_f1_macro_seen":    round(df_summary["f1_macro_seen"].std(), 4),
        "mean_f1_macro_unseen": round(df_summary["f1_macro_unseen"].mean(), 4),
        "std_f1_macro_unseen":  round(df_summary["f1_macro_unseen"].std(), 4),
    }
    agg_csv = os.path.join(RESULTS_DIR, "seen_unseen_aggregate.csv")
    pd.DataFrame([agg]).to_csv(agg_csv, index=False, encoding="utf-8-sig")

    print(f"\n{'='*65}")
    print("  HASIL AGREGAT (mean ± std across 3 splits)")
    print(f"{'='*65}")
    print(f"  F1-Macro All:    {agg['mean_f1_macro_all']:.4f} ± {agg['std_f1_macro_all']:.4f}")
    print(f"  F1-Macro Seen:   {agg['mean_f1_macro_seen']:.4f} ± {agg['std_f1_macro_seen']:.4f}")
    print(f"  F1-Macro Unseen: {agg['mean_f1_macro_unseen']:.4f} ± {agg['std_f1_macro_unseen']:.4f}")
    print(f"\n  Hasil disimpan di:")
    print(f"    {summary_csv}")
    print(f"    {per_intent_csv}")
    print(f"    {agg_csv}")
    print("=" * 65)


if __name__ == "__main__":
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    main()
