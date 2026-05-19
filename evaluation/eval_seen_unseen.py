"""
LABAN Zero-Shot Evaluation — Seen vs Unseen Labels
====================================================
Evaluates the LABAN model's zero-shot generalization by holding out
a subset of intent labels from training and measuring performance
on both seen and unseen intents at test time.

Protocol:
    1. Split the 10 intents into SEEN (7) and UNSEEN (3) groups.
    2. Build the training set: keep all samples but MASK (zero-out)
       the unseen intent columns during training. The model still
       receives ALL 10 intent label texts in its label encoder — this
       is what makes zero-shot classification possible via LABAN.
    3. Evaluate on a held-out test set with ALL 10 columns active.
    4. Compute:
         - F1-Macro Seen   (only over seen-intent columns)
         - F1-Macro Unseen (only over unseen-intent columns)
         - F1-Macro All    (over all 10 columns)

Multiple seen/unseen splits are evaluated to reduce variance.

Output:
    evaluation/results/seen_unseen_summary.csv       (per-split results)
    evaluation/results/seen_unseen_per_intent.csv    (per-intent per-split)
    evaluation/results/seen_unseen_aggregate.csv     (mean across splits)
"""

import os
import sys
import copy
import time
import warnings
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, precision_score, recall_score

from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ── Path setup ──────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from config import opt

RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Fixed hyper-parameters ──────────────────────────────────────────────────
BATCH_SIZE     = 16
MAX_LEN        = 50
EPOCHS         = 50
LEARNING_RATE  = 2e-5
THRESHOLD      = 0.5
RANDOM_SEED    = 42
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model to evaluate (uses the production model from config.py)
MODEL_NAME  = opt.MODEL_NAME
HIDDEN_SIZE = opt.hidden_size

# ── Dataset path ─────────────────────────────────────────────────────────────
AUG_CSV  = os.path.join(ROOT, "data", "augmentation", "dataset_multiintent_augmented.csv")
ORIG_CSV = os.path.join(ROOT, "data", "dataset_multiintent.csv")
DATASET_CSV = AUG_CSV if os.path.exists(AUG_CSV) else ORIG_CSV

# ── Seen / Unseen Splits ─────────────────────────────────────────────────────
# Each split holds out 3 intents as "unseen".
# We define 3 different splits for robustness; you can add more.
# The model is trained fresh for each split.
UNSEEN_SPLITS = [
    {
        "name": "Split-A",
        "unseen": [
            "Mengisyaratkan Butuh Bantuan Profesional",
            "Mengisyaratkan Gejala Fisik",
            "Menyatakan Reaksi Terkejut dan Tidak Terduga",
        ],
    },
    {
        "name": "Split-B",
        "unseen": [
            "Menyatakan Perasaan Benci dan Jijik",
            "Menyatakan Rasa Syukur dan Apresiasi",
            "Menyatakan Perasaan Sebelum Menghadapi Kejadian",
        ],
    },
    {
        "name": "Split-C",
        "unseen": [
            "Menyatakan Perasaan Marah dan Frustasi",
            "Menyatakan Perasaan Percaya",
            "Menyatakan Perasaan Sedih dan Kehilangan",
        ],
    },
]


# ═══════════════════════════════════════════════════════════════════════════
#  LABAN Architecture (self-contained — mirrors bert_model.py)
# ═══════════════════════════════════════════════════════════════════════════

class LABANModel(nn.Module):
    def __init__(self, model_name, hidden_size, num_labels):
        super().__init__()
        self.hidden_size = hidden_size
        self.bert = AutoModel.from_pretrained(
            model_name, output_hidden_states=True,
            output_attentions=True, ignore_mismatched_sizes=True
        )
        self.bertlabelencoder = AutoModel.from_pretrained(
            model_name, ignore_mismatched_sizes=True
        )

    def forward(self, utterance_ids, utterance_mask, label_ids, label_mask):
        label_out = self.bertlabelencoder(
            input_ids=label_ids, attention_mask=label_mask, return_dict=True
        )
        if hasattr(label_out, 'pooler_output') and label_out.pooler_output is not None:
            clusters = label_out.pooler_output
        else:
            clusters = label_out.last_hidden_state[:, 0, :]

        utt_out = self.bert(
            input_ids=utterance_ids, attention_mask=utterance_mask,
            output_hidden_states=True, output_attentions=True, return_dict=True
        )
        if hasattr(utt_out, 'pooler_output') and utt_out.pooler_output is not None:
            pooled = utt_out.pooler_output
        else:
            pooled = utt_out.last_hidden_state[:, 0, :]

        gram = torch.mm(clusters, clusters.T)
        gram = gram + torch.eye(gram.size(0), device=gram.device) * 1e-4
        weight = torch.mm(pooled, clusters.T)
        logits = torch.mm(weight, torch.inverse(gram)) * np.sqrt(self.hidden_size)
        return logits


# ═══════════════════════════════════════════════════════════════════════════
#  Dataset class
# ═══════════════════════════════════════════════════════════════════════════

class LABANDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len=MAX_LEN):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label_columns = self.data.columns[1:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        encoding = self.tokenizer(
            row["question"],
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.FloatTensor(row[self.label_columns].values.astype("float32")),
        }


# ═══════════════════════════════════════════════════════════════════════════
#  Core evaluation logic
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_model(model, loader, criterion, intent_ids, intent_mask):
    """Run forward pass and return loss + raw preds/targets."""
    model.eval()
    total_loss = 0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            logits = model(ids, mask, intent_ids, intent_mask)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= THRESHOLD).astype(int)
            all_preds.extend(preds)
            all_targets.extend(labels.cpu().numpy())

    return total_loss / len(loader), np.array(all_preds), np.array(all_targets)


def compute_split_metrics(preds, targets, intent_list, seen_intents, unseen_intents):
    """
    Compute F1-Macro for Seen, Unseen, and All intent columns.

    Returns dict with all metrics + per-intent F1 breakdown.
    """
    seen_indices   = [intent_list.index(i) for i in seen_intents]
    unseen_indices = [intent_list.index(i) for i in unseen_intents]

    # F1-Macro All
    f1_all = f1_score(targets, preds, average="macro", zero_division=0)
    p_all  = precision_score(targets, preds, average="macro", zero_division=0)
    r_all  = recall_score(targets, preds, average="macro", zero_division=0)

    # F1-Macro Seen (only seen columns)
    f1_seen = f1_score(
        targets[:, seen_indices], preds[:, seen_indices],
        average="macro", zero_division=0
    )
    p_seen = precision_score(
        targets[:, seen_indices], preds[:, seen_indices],
        average="macro", zero_division=0
    )
    r_seen = recall_score(
        targets[:, seen_indices], preds[:, seen_indices],
        average="macro", zero_division=0
    )

    # F1-Macro Unseen (only unseen columns)
    f1_unseen = f1_score(
        targets[:, unseen_indices], preds[:, unseen_indices],
        average="macro", zero_division=0
    )
    p_unseen = precision_score(
        targets[:, unseen_indices], preds[:, unseen_indices],
        average="macro", zero_division=0
    )
    r_unseen = recall_score(
        targets[:, unseen_indices], preds[:, unseen_indices],
        average="macro", zero_division=0
    )

    # Per-intent F1
    per_intent = {}
    for idx, intent in enumerate(intent_list):
        t_col = targets[:, idx]
        p_col = preds[:, idx]
        per_intent[intent] = {
            "f1": f1_score(t_col, p_col, average="binary", zero_division=0),
            "precision": precision_score(t_col, p_col, average="binary", zero_division=0),
            "recall": recall_score(t_col, p_col, average="binary", zero_division=0),
            "support": int(t_col.sum()),
            "group": "unseen" if intent in unseen_intents else "seen",
        }

    return {
        "f1_macro_all": round(f1_all, 4),
        "precision_macro_all": round(p_all, 4),
        "recall_macro_all": round(r_all, 4),
        "f1_macro_seen": round(f1_seen, 4),
        "precision_macro_seen": round(p_seen, 4),
        "recall_macro_seen": round(r_seen, 4),
        "f1_macro_unseen": round(f1_unseen, 4),
        "precision_macro_unseen": round(p_unseen, 4),
        "recall_macro_unseen": round(r_unseen, 4),
        "per_intent": per_intent,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  Single split training + evaluation
# ═══════════════════════════════════════════════════════════════════════════

def run_one_split(split_name, unseen_intents, df_encoded, intent_list, tokenizer):
    """
    Train LABAN with unseen intent columns masked, evaluate on full test set.
    """
    seen_intents = [i for i in intent_list if i not in unseen_intents]

    print(f"\n{'═'*65}")
    print(f"  {split_name}")
    print(f"  Seen   ({len(seen_intents)}): {seen_intents}")
    print(f"  Unseen ({len(unseen_intents)}): {unseen_intents}")
    print(f"{'═'*65}")

    # ── Split data (same seed for reproducibility) ───────────────────────
    df_train_full, df_temp = train_test_split(
        df_encoded, test_size=0.2, random_state=RANDOM_SEED
    )
    df_val, df_test = train_test_split(
        df_temp, test_size=0.5, random_state=RANDOM_SEED
    )

    # ── Mask unseen columns in training and validation data ──────────────
    # The model still gets ALL 10 intent label texts via the label encoder,
    # but the ground-truth for unseen intents is zeroed out during training.
    # This forces the model to learn ONLY from seen-intent supervision.
    df_train = df_train_full.copy()
    df_val_masked = df_val.copy()
    for intent in unseen_intents:
        df_train[intent] = 0.0
        df_val_masked[intent] = 0.0

    print(f"  Train: {len(df_train)} samples (unseen columns zeroed)")
    print(f"  Val:   {len(df_val_masked)} samples (unseen columns zeroed)")
    print(f"  Test:  {len(df_test)} samples (ALL columns active)")

    # ── Tokenize intent labels (ALL 10 — key for zero-shot) ──────────────
    tokenized_intents = tokenizer(
        intent_list, padding=True, truncation=True, return_tensors="pt"
    )
    intent_ids = tokenized_intents["input_ids"].to(DEVICE)
    intent_mask = tokenized_intents["attention_mask"].to(DEVICE)

    # ── DataLoaders ──────────────────────────────────────────────────────
    train_loader = DataLoader(
        LABANDataset(df_train, tokenizer), batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        LABANDataset(df_val_masked, tokenizer), batch_size=BATCH_SIZE, shuffle=False
    )
    test_loader = DataLoader(
        LABANDataset(df_test, tokenizer), batch_size=BATCH_SIZE, shuffle=False
    )

    # ── Model + training ─────────────────────────────────────────────────
    model = LABANModel(MODEL_NAME, HIDDEN_SIZE, len(intent_list)).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss(reduction="sum").to(DEVICE)

    best_val_f1 = 0.0
    best_state = None

    t_start = time.perf_counter()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        all_preds, all_targets = [], []

        loop = tqdm(train_loader, desc=f"[{split_name}] Epoch {epoch+1}/{EPOCHS}", leave=False)
        for batch in loop:
            optimizer.zero_grad()
            ids = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            logits = model(ids, mask, intent_ids, intent_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            preds = (probs >= THRESHOLD).astype(int)
            all_preds.extend(preds)
            all_targets.extend(labels.cpu().numpy())
            loop.set_postfix(loss=f"{loss.item():.2f}")

        avg_train_loss = total_loss / len(train_loader)
        train_f1 = f1_score(all_targets, all_preds, average="micro", zero_division=0)

        # Validation (on masked val set — only seen columns matter for checkpointing)
        val_loss, val_preds, val_targets = evaluate_model(
            model, val_loader, criterion, intent_ids, intent_mask
        )
        # Compute val F1 only on seen columns for checkpoint selection
        seen_idx = [intent_list.index(i) for i in seen_intents]
        val_f1_seen = f1_score(
            np.array(val_targets)[:, seen_idx],
            np.array(val_preds)[:, seen_idx],
            average="micro", zero_division=0
        )

        if epoch % 10 == 0 or epoch == EPOCHS - 1:
            print(f"  Epoch {epoch+1:2d}  train_loss={avg_train_loss:.4f}"
                  f"  train_F1={train_f1:.4f}  val_F1_seen={val_f1_seen:.4f}")

        if val_f1_seen > best_val_f1:
            best_val_f1 = val_f1_seen
            best_state = copy.deepcopy(model.state_dict())

    train_time = time.perf_counter() - t_start

    # ── Test evaluation (ALL columns active, including unseen) ───────────
    if best_state:
        model.load_state_dict(best_state)

    _, test_preds, test_targets = evaluate_model(
        model, test_loader, criterion, intent_ids, intent_mask
    )

    metrics = compute_split_metrics(
        test_preds, test_targets, intent_list, seen_intents, unseen_intents
    )

    print(f"\n  ── {split_name} TEST RESULTS ──")
    print(f"  F1-Macro All:    {metrics['f1_macro_all']:.4f}")
    print(f"  F1-Macro Seen:   {metrics['f1_macro_seen']:.4f}")
    print(f"  F1-Macro Unseen: {metrics['f1_macro_unseen']:.4f}")
    print(f"  Training time:   {train_time:.1f}s")

    print(f"\n  Per-intent breakdown:")
    for intent, m in metrics["per_intent"].items():
        tag = "[UNSEEN]" if m["group"] == "unseen" else "[seen]  "
        print(f"    {tag} F1={m['f1']:.4f}  P={m['precision']:.4f}"
              f"  R={m['recall']:.4f}  sup={m['support']:3d}  {intent}")

    # Free GPU memory
    del model, optimizer, criterion
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return {
        "split": split_name,
        "seen_intents": "; ".join(seen_intents),
        "unseen_intents": "; ".join(unseen_intents),
        "f1_macro_all": metrics["f1_macro_all"],
        "precision_macro_all": metrics["precision_macro_all"],
        "recall_macro_all": metrics["recall_macro_all"],
        "f1_macro_seen": metrics["f1_macro_seen"],
        "precision_macro_seen": metrics["precision_macro_seen"],
        "recall_macro_seen": metrics["recall_macro_seen"],
        "f1_macro_unseen": metrics["f1_macro_unseen"],
        "precision_macro_unseen": metrics["precision_macro_unseen"],
        "recall_macro_unseen": metrics["recall_macro_unseen"],
        "best_val_f1_seen": round(best_val_f1, 4),
        "training_time_s": round(train_time, 1),
    }, metrics["per_intent"]


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("  LABAN Zero-Shot Evaluation — Seen vs Unseen Labels")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Device: {DEVICE}")
    print("=" * 65)

    # ── Load & encode dataset ────────────────────────────────────────────
    print(f"\nDataset: {DATASET_CSV}")
    df = pd.read_csv(DATASET_CSV, encoding="utf-8-sig")
    df["intent_list"] = df["Intent"].apply(lambda x: x.split("; "))

    mlb = MultiLabelBinarizer()
    encoded = mlb.fit_transform(df["intent_list"])
    intent_list = list(mlb.classes_)
    print(f"Intents ({len(intent_list)}): {intent_list}")

    df_encoded = pd.concat(
        [df[["question"]], pd.DataFrame(encoded, columns=intent_list).astype("float32")],
        axis=1,
    )
    print(f"Total samples: {len(df_encoded)}")

    # ── Tokenizer (shared across all splits, same model) ─────────────────
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # ── Run each split ───────────────────────────────────────────────────
    summary_rows = []
    per_intent_rows = []

    for split_config in UNSEEN_SPLITS:
        split_name = split_config["name"]
        unseen = split_config["unseen"]

        # Validate intent names
        for u in unseen:
            assert u in intent_list, f"Unknown intent '{u}' in {split_name}"

        try:
            result, per_intent = run_one_split(
                split_name, unseen, df_encoded.copy(), intent_list, tokenizer
            )
            summary_rows.append(result)

            for intent_name, m in per_intent.items():
                per_intent_rows.append({
                    "split": split_name,
                    "intent": intent_name,
                    "group": m["group"],
                    "f1": round(m["f1"], 4),
                    "precision": round(m["precision"], 4),
                    "recall": round(m["recall"], 4),
                    "support": m["support"],
                })

        except Exception as e:
            print(f"\n  [ERROR] {split_name} failed: {e}")
            import traceback; traceback.print_exc()
            summary_rows.append({"split": split_name, "status": "FAILED", "error": str(e)})

    # ── Save results ─────────────────────────────────────────────────────
    df_summary = pd.DataFrame(summary_rows)
    df_per_intent = pd.DataFrame(per_intent_rows)

    summary_csv = os.path.join(RESULTS_DIR, "seen_unseen_summary.csv")
    per_intent_csv = os.path.join(RESULTS_DIR, "seen_unseen_per_intent.csv")
    df_summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    df_per_intent.to_csv(per_intent_csv, index=False, encoding="utf-8-sig")

    # ── Aggregate across splits ──────────────────────────────────────────
    ok = df_summary[~df_summary.get("status", pd.Series(dtype=str)).eq("FAILED")].copy()
    if not ok.empty:
        agg = {
            "mean_f1_macro_all":    round(ok["f1_macro_all"].mean(), 4),
            "std_f1_macro_all":     round(ok["f1_macro_all"].std(), 4),
            "mean_f1_macro_seen":   round(ok["f1_macro_seen"].mean(), 4),
            "std_f1_macro_seen":    round(ok["f1_macro_seen"].std(), 4),
            "mean_f1_macro_unseen": round(ok["f1_macro_unseen"].mean(), 4),
            "std_f1_macro_unseen":  round(ok["f1_macro_unseen"].std(), 4),
        }
        agg_csv = os.path.join(RESULTS_DIR, "seen_unseen_aggregate.csv")
        pd.DataFrame([agg]).to_csv(agg_csv, index=False, encoding="utf-8-sig")

        print(f"\n{'='*65}")
        print("  AGGREGATE RESULTS (mean ± std across splits)")
        print(f"{'='*65}")
        print(f"  F1-Macro All:    {agg['mean_f1_macro_all']:.4f} ± {agg['std_f1_macro_all']:.4f}")
        print(f"  F1-Macro Seen:   {agg['mean_f1_macro_seen']:.4f} ± {agg['std_f1_macro_seen']:.4f}")
        print(f"  F1-Macro Unseen: {agg['mean_f1_macro_unseen']:.4f} ± {agg['std_f1_macro_unseen']:.4f}")
        print(f"\n  Saved: {agg_csv}")

    # ── Per-split summary table ──────────────────────────────────────────
    print(f"\n{'='*65}")
    print("  PER-SPLIT SUMMARY")
    print(f"{'='*65}")
    if not ok.empty:
        cols = ["split", "f1_macro_all", "f1_macro_seen", "f1_macro_unseen", "training_time_s"]
        print(ok[[c for c in cols if c in ok.columns]].to_string(index=False))

    print(f"\n  Results saved to:")
    print(f"    {summary_csv}")
    print(f"    {per_intent_csv}")
    print("=" * 65)


if __name__ == "__main__":
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    main()
