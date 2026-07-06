"""
LABAN Multi-Label Classifier — Backbone Comparison
====================================================
Trains the same LABAN architecture with different transformer backbones
and compares them on identical train/val/test splits.

For each backbone:
    1. Tokenise dataset + intent labels
    2. Train LABAN for N epochs (same LR, batch size, seed)
    3. Record best-val-F1 checkpoint
    4. Evaluate on the held-out test set
    5. Produce per-intent classification report

Output:
    evaluation/results/backbone_comparison_summary.csv
    evaluation/results/backbone_comparison_per_intent.csv
    evaluation/results/<model_tag>_training_curve.png  (loss + F1 curves)
"""

import os
import sys
import time
import copy
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    classification_report, accuracy_score
)

from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ── Path setup ──────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Fixed hyper-parameters (shared across ALL models) ────────────────────────
BATCH_SIZE     = 16
MAX_LEN        = 50
EPOCHS         = 50
LEARNING_RATE  = 2e-5
THRESHOLD      = 0.5
RANDOM_SEED    = 42
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Dataset path ─────────────────────────────────────────────────────────────
AUG_CSV  = os.path.join(ROOT, "data", "augmentation", "dataset_multiintent_augmented.csv")
ORIG_CSV = os.path.join(ROOT, "data", "dataset_multiintent.csv")
DATASET_CSV = AUG_CSV if os.path.exists(AUG_CSV) else ORIG_CSV

# ── Candidate backbone models ───────────────────────────────────────────────
# Four paradigms for the academic backbone comparison:
#   1. IndoBERT       — formal Indonesian corpus (SOTA Bahasa Indonesia)
#   2. IndoBERTweet   — informal/social-media Indonesian corpus
#   3. mBERT          — multilingual baseline (standard cross-lingual)
#   4. MiniLM-multi   — lightweight / distilled baseline (computational efficiency)
# (tag, huggingface_id, hidden_size)
CANDIDATE_MODELS = [
    ("IndoBERT",      "indobenchmark/indobert-base-p1",                                  768),
    ("IndoBERTweet",  "indolem/indobertweet-base-uncased",                               768),
    ("mBERT",         "bert-base-multilingual-cased",                                    768),
    ("MiniLM-multi",  "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",     384),
]


# ═══════════════════════════════════════════════════════════════════════════
#  LABAN Architecture (self-contained copy — mirrors bert_model.py)
# ═══════════════════════════════════════════════════════════════════════════

class LABANModel(nn.Module):
    """
    Label-Attention-Based Approach for multi-label classification.
    Uses two transformer encoders (utterance + label) and a closed-form
    projection: logits = weight @ inv(gram) * sqrt(hidden_size)
    """
    def __init__(self, model_name: str, hidden_size: int, num_labels: int):
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
        # Encode intent labels
        label_out = self.bertlabelencoder(
            input_ids=label_ids, attention_mask=label_mask, return_dict=True
        )
        # Use pooler_output for models that have it, else CLS token
        if hasattr(label_out, 'pooler_output') and label_out.pooler_output is not None:
            clusters = label_out.pooler_output
        else:
            clusters = label_out.last_hidden_state[:, 0, :]

        # Encode utterance
        utt_out = self.bert(
            input_ids=utterance_ids, attention_mask=utterance_mask,
            output_hidden_states=True, output_attentions=True, return_dict=True
        )
        if hasattr(utt_out, 'pooler_output') and utt_out.pooler_output is not None:
            pooled = utt_out.pooler_output
        else:
            pooled = utt_out.last_hidden_state[:, 0, :]

        # Gram matrix + Tikhonov regularization
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
#  Evaluation helpers
# ═══════════════════════════════════════════════════════════════════════════

def evaluate(model, loader, criterion, intent_ids, intent_mask):
    """Run one pass over a DataLoader, return loss, F1, preds, targets."""
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

    avg_loss = total_loss / len(loader)
    f1 = f1_score(all_targets, all_preds, average="micro", zero_division=0)
    return avg_loss, f1, np.array(all_preds), np.array(all_targets)


# ═══════════════════════════════════════════════════════════════════════════
#  Single-model training pipeline
# ═══════════════════════════════════════════════════════════════════════════

def train_one_model(tag, model_name, hidden_size, df_train, df_val, df_test,
                    intent_list, mlb):
    """
    Train + evaluate one LABAN backbone. Returns a summary dict.
    """
    print(f"\n{'═'*65}")
    print(f"  MODEL: {tag}")
    print(f"  HF ID: {model_name}")
    print(f"  Hidden: {hidden_size}   Device: {DEVICE}")
    print(f"{'═'*65}")

    # --- Tokenizer & dataloaders ---
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    tokenized_intents = tokenizer(
        intent_list, padding=True, truncation=True, return_tensors="pt"
    )
    intent_ids = tokenized_intents["input_ids"].to(DEVICE)
    intent_mask = tokenized_intents["attention_mask"].to(DEVICE)

    train_loader = DataLoader(
        LABANDataset(df_train, tokenizer), batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        LABANDataset(df_val, tokenizer), batch_size=BATCH_SIZE, shuffle=False
    )
    test_loader = DataLoader(
        LABANDataset(df_test, tokenizer), batch_size=BATCH_SIZE, shuffle=False
    )

    # --- Model ---
    num_labels = len(intent_list)
    model = LABANModel(model_name, hidden_size, num_labels).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss(reduction="sum").to(DEVICE)

    best_val_f1 = 0.0
    best_state = None
    hist = {"train_loss": [], "val_loss": [], "train_f1": [], "val_f1": []}

    t_start = time.perf_counter()

    for epoch in range(EPOCHS):
        # ── Training ──
        model.train()
        total_loss = 0
        all_preds, all_targets = [], []

        loop = tqdm(train_loader, desc=f"[{tag}] Epoch {epoch+1}/{EPOCHS}", leave=False)
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
        hist["train_loss"].append(avg_train_loss)
        hist["train_f1"].append(train_f1)

        # ── Validation ──
        val_loss, val_f1, _, _ = evaluate(model, val_loader, criterion, intent_ids, intent_mask)
        hist["val_loss"].append(val_loss)
        hist["val_f1"].append(val_f1)

        print(f"  Epoch {epoch+1:2d}  train_loss={avg_train_loss:.4f}  train_F1={train_f1:.4f}"
              f"  val_loss={val_loss:.4f}  val_F1={val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = copy.deepcopy(model.state_dict())

    train_time = time.perf_counter() - t_start

    # ── Testing (load best checkpoint) ──
    if best_state:
        model.load_state_dict(best_state)
    test_loss, test_f1, test_preds, test_targets = evaluate(
        model, test_loader, criterion, intent_ids, intent_mask
    )
    test_precision = precision_score(test_targets, test_preds, average="micro", zero_division=0)
    test_recall    = recall_score(test_targets, test_preds, average="micro", zero_division=0)

    # Per-intent report
    report = classification_report(
        test_targets, test_preds,
        target_names=intent_list,
        output_dict=True, zero_division=0,
    )

    print(f"\n  ── TEST RESULTS ({tag}) ──")
    print(f"  Test Loss:      {test_loss:.4f}")
    print(f"  Test F1 (micro): {test_f1:.4f}")
    print(f"  Test Precision:  {test_precision:.4f}")
    print(f"  Test Recall:     {test_recall:.4f}")
    print(f"  Training time:   {train_time:.1f}s")

    # ── Save training curve ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(hist["train_loss"], label="Train Loss")
    axes[0].plot(hist["val_loss"], label="Val Loss")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{tag} — Loss"); axes[0].legend()

    axes[1].plot(hist["train_f1"], label="Train F1")
    axes[1].plot(hist["val_f1"], label="Val F1")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("F1 Score")
    axes[1].set_title(f"{tag} — F1"); axes[1].legend()

    plt.tight_layout()
    curve_path = os.path.join(RESULTS_DIR, f"{tag}_training_curve.png")
    plt.savefig(curve_path, dpi=150)
    plt.close()
    print(f"  Curve saved: {curve_path}")

    # Free GPU memory
    del model, optimizer, criterion
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    return {
        "model": tag,
        "model_id": model_name,
        "hidden_size": hidden_size,
        "best_val_f1": round(best_val_f1, 4),
        "test_loss": round(test_loss, 4),
        "test_f1_micro": round(test_f1, 4),
        "test_precision_micro": round(test_precision, 4),
        "test_recall_micro": round(test_recall, 4),
        "training_time_s": round(train_time, 1),
        "epochs": EPOCHS,
    }, report


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("  LABAN Backbone Comparison")
    print("=" * 65)

    # ── Load & encode dataset (once) ─────────────────────────────────────
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

    # ── Fixed 80/10/10 split (same seed for all models) ──────────────────
    df_train, df_temp = train_test_split(df_encoded, test_size=0.2, random_state=RANDOM_SEED)
    df_val, df_test = train_test_split(df_temp, test_size=0.5, random_state=RANDOM_SEED)
    print(f"Split: train={len(df_train)}, val={len(df_val)}, test={len(df_test)}\n")

    # ── Run each backbone ────────────────────────────────────────────────
    summary_rows = []
    per_intent_rows = []

    for tag, model_id, hidden in CANDIDATE_MODELS:
        try:
            result, report = train_one_model(
                tag, model_id, hidden,
                df_train.copy(), df_val.copy(), df_test.copy(),
                intent_list, mlb
            )
            summary_rows.append(result)

            # Flatten per-intent report
            for intent_name in intent_list:
                if intent_name in report:
                    per_intent_rows.append({
                        "model": tag,
                        "intent": intent_name,
                        "precision": round(report[intent_name].get("precision", 0), 4),
                        "recall": round(report[intent_name].get("recall", 0), 4),
                        "f1-score": round(report[intent_name].get("f1-score", 0), 4),
                        "support": report[intent_name].get("support", 0),
                    })

        except Exception as e:
            print(f"\n  [ERROR] {tag} failed: {e}")
            summary_rows.append({
                "model": tag, "model_id": model_id, "hidden_size": hidden,
                "status": "FAILED", "error": str(e),
            })

    # ── Save results ─────────────────────────────────────────────────────
    df_summary = pd.DataFrame(summary_rows)
    df_per_intent = pd.DataFrame(per_intent_rows)

    summary_csv = os.path.join(RESULTS_DIR, "backbone_comparison_summary.csv")
    per_intent_csv = os.path.join(RESULTS_DIR, "backbone_comparison_per_intent.csv")
    df_summary.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    df_per_intent.to_csv(per_intent_csv, index=False, encoding="utf-8-sig")

    # ── Print summary ────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("  FINAL COMPARISON SUMMARY")
    print(f"{'='*65}")

    if "status" in df_summary.columns:
        ok = df_summary[df_summary["status"].ne("FAILED")].copy()
    else:
        ok = df_summary.copy()  # no failures — all rows are valid
    if not ok.empty:
        cols = ["model", "hidden_size", "test_f1_micro",
                "test_precision_micro", "test_recall_micro",
                "best_val_f1", "training_time_s"]
        print(ok[[c for c in cols if c in ok.columns]].to_string(index=False))

        best = ok.loc[ok["test_f1_micro"].idxmax()]
        print(f"\n  🏆 Best test F1: {best['model']} ({best['test_f1_micro']:.4f})")
        fastest = ok.loc[ok["training_time_s"].idxmin()]
        print(f"  ⚡ Fastest:      {fastest['model']} ({fastest['training_time_s']:.1f}s)")

    print(f"\n  Results saved to:")
    print(f"    {summary_csv}")
    print(f"    {per_intent_csv}")
    print("=" * 65)


if __name__ == "__main__":
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    main()
