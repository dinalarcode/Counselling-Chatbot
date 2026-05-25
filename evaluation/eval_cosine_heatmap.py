"""
LABAN Cosine Similarity Heatmap Evaluation
============================================
Assesses the consistency of the model's ability to establish semantic
connections between input (utterance) embeddings and label embeddings.

This evaluation uses the currently active backbone (IndoBERTweet) and the
trained checkpoint to produce three visualizations:

1. **Label-vs-Label Heatmap** — Cosine similarity between all pairs of
   intent label embeddings from the label encoder. Shows how the model
   separates intent semantics in embedding space.

2. **Utterance Centroid-vs-Label Heatmap** — For each intent, computes the
   centroid (mean) of all test utterance embeddings that have that intent
   as a ground-truth label, then measures cosine similarity against every
   label embedding. High diagonal = good alignment.

3. **Per-Sample Utterance-vs-Label Heatmap** — Averaged cosine similarity
   between individual test utterances (grouped by true intent) and each
   label embedding. Provides a finer-grained view than centroids.

Output:
    evaluation/results/cosine_label_vs_label.png
    evaluation/results/cosine_centroid_vs_label.png
    evaluation/results/cosine_utterance_vs_label.png
    evaluation/results/cosine_similarity_metrics.csv
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle

import torch
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity

from transformers import AutoTokenizer, AutoModel

warnings.filterwarnings("ignore")

# -- Path setup --------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

RESULTS_DIR = os.path.join(ROOT, "evaluation", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# -- Configuration -----------------------------------------------------------
MODEL_NAME   = "indolem/indobertweet-base-uncased"
HIDDEN_SIZE  = 768
MAX_LEN      = 50
BATCH_SIZE   = 16
THRESHOLD    = 0.5
RANDOM_SEED  = 42
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT   = os.path.join(ROOT, "checkpoint", "IndoBERT_multi_label_zsl.pt")

# -- Dataset path -------------------------------------------------------------
AUG_CSV  = os.path.join(ROOT, "data", "augmentation", "dataset_multiintent_augmented.csv")
ORIG_CSV = os.path.join(ROOT, "data", "dataset_multiintent.csv")
DATASET_CSV = AUG_CSV if os.path.exists(AUG_CSV) else ORIG_CSV

# -- Short display names for intent labels (fits heatmap cells) --------------
SHORT_NAMES = {
    "Mengisyaratkan Butuh Bantuan Profesional": "Butuh Bantuan\nProfesional",
    "Mengisyaratkan Gejala Fisik":              "Gejala Fisik",
    "Menyatakan Perasaan Benci dan Jijik":      "Benci & Jijik",
    "Menyatakan Perasaan Marah dan Frustasi":    "Marah &\nFrustasi",
    "Menyatakan Perasaan Percaya":              "Percaya",
    "Menyatakan Perasaan Sebelum Menghadapi Kejadian": "Sebelum\nKejadian",
    "Menyatakan Perasaan Sedih dan Kehilangan": "Sedih &\nKehilangan",
    "Menyatakan Perasaan Takut dan Kecemasan":  "Takut &\nCemas",
    "Menyatakan Rasa Syukur dan Apresiasi":     "Syukur &\nApresiasi",
    "Menyatakan Reaksi Terkejut dan Tidak Terduga": "Terkejut &\nTidak Terduga",
}


# ===========================================================================
#  LABAN Architecture (self-contained — mirrors bert_model.py)
# ===========================================================================

class LABANModel(torch.nn.Module):
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

    @torch.no_grad()
    def encode_labels(self, label_ids, label_mask):
        """Extract label embeddings from the label encoder."""
        label_out = self.bertlabelencoder(
            input_ids=label_ids, attention_mask=label_mask, return_dict=True
        )
        if hasattr(label_out, 'pooler_output') and label_out.pooler_output is not None:
            return label_out.pooler_output
        return label_out.last_hidden_state[:, 0, :]

    @torch.no_grad()
    def encode_utterances(self, utterance_ids, utterance_mask):
        """Extract utterance embeddings from the utterance encoder."""
        utt_out = self.bert(
            input_ids=utterance_ids, attention_mask=utterance_mask,
            output_hidden_states=True, output_attentions=True, return_dict=True
        )
        if hasattr(utt_out, 'pooler_output') and utt_out.pooler_output is not None:
            return utt_out.pooler_output
        return utt_out.last_hidden_state[:, 0, :]


# ===========================================================================
#  Dataset class
# ===========================================================================

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


# ===========================================================================
#  Heatmap plotting helpers
# ===========================================================================

def plot_heatmap(matrix, row_labels, col_labels, title, save_path,
                 cmap="RdYlGn", annotate=True, figsize=None,
                 vmin=None, vmax=None, highlight_diagonal=False):
    """
    Plot a publication-quality heatmap with annotations.
    """
    n_rows, n_cols = matrix.shape
    if figsize is None:
        figsize = (max(10, n_cols * 1.2), max(8, n_rows * 1.0))

    fig, ax = plt.subplots(figsize=figsize)

    if vmin is None:
        vmin = matrix.min()
    if vmax is None:
        vmax = matrix.max()

    im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)

    # Axis ticks
    ax.set_xticks(range(n_cols))
    ax.set_yticks(range(n_rows))
    ax.set_xticklabels(col_labels, fontsize=9, rotation=45, ha="right")
    ax.set_yticklabels(row_labels, fontsize=9)

    # Annotations
    if annotate:
        for i in range(n_rows):
            for j in range(n_cols):
                val = matrix[i, j]
                # Choose text color based on background intensity
                text_color = "white" if val < (vmin + vmax) / 2 else "black"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=8, color=text_color, fontweight="bold")

    # Highlight diagonal cells
    if highlight_diagonal:
        for i in range(min(n_rows, n_cols)):
            rect = Rectangle((i - 0.5, i - 0.5), 1, 1, linewidth=2.5,
                              edgecolor="#222222", facecolor="none")
            ax.add_patch(rect)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Cosine Similarity", fontsize=10)

    ax.set_title(title, fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  [OK] Saved: {save_path}")


# ===========================================================================
#  Main evaluation logic
# ===========================================================================

def main():
    print("=" * 65)
    print("  LABAN Cosine Similarity Heatmap Evaluation")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Device: {DEVICE}")
    print("=" * 65)

    # -- 1. Load and prepare data -----------------------------------------
    print(f"\nDataset: {DATASET_CSV}")
    df = pd.read_csv(DATASET_CSV, encoding="utf-8-sig")
    df["intent_list"] = df["Intent"].apply(lambda x: x.split("; "))

    mlb = MultiLabelBinarizer()
    encoded = mlb.fit_transform(df["intent_list"])
    intent_list = list(mlb.classes_)
    n_intents = len(intent_list)
    print(f"Intents ({n_intents}): {intent_list}")

    df_encoded = pd.concat(
        [df[["question"]], pd.DataFrame(encoded, columns=intent_list).astype("float32")],
        axis=1,
    )

    # Same 80/10/10 split as other evaluations
    _, df_temp = train_test_split(df_encoded, test_size=0.2, random_state=RANDOM_SEED)
    _, df_test = train_test_split(df_temp, test_size=0.5, random_state=RANDOM_SEED)
    print(f"Test set size: {len(df_test)}")

    # -- 2. Load tokenizer and model --------------------------------------
    print(f"\nLoading checkpoint: {CHECKPOINT}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = LABANModel(MODEL_NAME, HIDDEN_SIZE, n_intents).to(DEVICE)
    model.load_state_dict(
        torch.load(CHECKPOINT, map_location=DEVICE, weights_only=True)
    )
    model.eval()
    print("  [OK] Model loaded")

    # -- 3. Tokenize intent labels ----------------------------------------
    tokenized_intents = tokenizer(
        intent_list, padding=True, truncation=True, return_tensors="pt"
    )
    intent_ids = tokenized_intents["input_ids"].to(DEVICE)
    intent_mask = tokenized_intents["attention_mask"].to(DEVICE)

    # -- 4. Extract label embeddings --------------------------------------
    print("\nExtracting label embeddings...")
    label_embeddings = model.encode_labels(intent_ids, intent_mask).cpu().numpy()
    print(f"  Label embeddings shape: {label_embeddings.shape}")

    # -- 5. Extract utterance embeddings from test set --------------------
    print("Extracting utterance embeddings from test set...")
    test_dataset = LABANDataset(df_test.reset_index(drop=True), tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    all_utterance_embeds = []
    all_labels = []

    for batch in test_loader:
        ids = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        embeds = model.encode_utterances(ids, mask).cpu().numpy()
        all_utterance_embeds.append(embeds)
        all_labels.append(batch["labels"].numpy())

    utterance_embeddings = np.concatenate(all_utterance_embeds, axis=0)
    label_matrix = np.concatenate(all_labels, axis=0)
    print(f"  Utterance embeddings shape: {utterance_embeddings.shape}")
    print(f"  Label matrix shape: {label_matrix.shape}")

    # -- Short label names for display ------------------------------------
    short_labels = [SHORT_NAMES.get(name, name) for name in intent_list]

    # ======================================================================
    #  HEATMAP 1: Label-vs-Label Cosine Similarity
    # ======================================================================
    print("\n-- Heatmap 1: Label-vs-Label --")
    label_sim = cosine_similarity(label_embeddings, label_embeddings)

    plot_heatmap(
        label_sim,
        row_labels=short_labels,
        col_labels=short_labels,
        title="Label-vs-Label Cosine Similarity\n(Intent Label Embeddings from LABAN Label Encoder)",
        save_path=os.path.join(RESULTS_DIR, "cosine_label_vs_label.png"),
        cmap="RdYlGn",
        vmin=-1.0, vmax=1.0,
        highlight_diagonal=True,
    )

    # Report label separability metrics
    mask_upper = np.triu(np.ones_like(label_sim, dtype=bool), k=1)
    off_diag = label_sim[mask_upper]
    diag = np.diag(label_sim)
    print(f"  Diagonal (self-sim):  mean={diag.mean():.4f} (should be 1.0)")
    print(f"  Off-diagonal:         mean={off_diag.mean():.4f}, "
          f"min={off_diag.min():.4f}, max={off_diag.max():.4f}")
    print(f"  Separation gap:       {diag.mean() - off_diag.mean():.4f} "
          f"(larger = better label separability)")

    # ======================================================================
    #  HEATMAP 2: Utterance Centroid-vs-Label
    # ======================================================================
    print("\n-- Heatmap 2: Utterance Centroid-vs-Label --")

    # Compute per-intent centroid from test utterances
    centroids = np.zeros((n_intents, utterance_embeddings.shape[1]))
    intent_counts = []

    for i in range(n_intents):
        # Select utterances where this intent is active (multi-label aware)
        mask_i = label_matrix[:, i] == 1.0
        count = mask_i.sum()
        intent_counts.append(int(count))
        if count > 0:
            centroids[i] = utterance_embeddings[mask_i].mean(axis=0)
        else:
            centroids[i] = np.zeros(utterance_embeddings.shape[1])
            print(f"  [WARN] No test samples for intent: {intent_list[i]}")

    centroid_vs_label = cosine_similarity(centroids, label_embeddings)

    # Add sample counts to row labels
    centroid_row_labels = [
        f"{short_labels[i]}\n(n={intent_counts[i]})"
        for i in range(n_intents)
    ]

    plot_heatmap(
        centroid_vs_label,
        row_labels=centroid_row_labels,
        col_labels=short_labels,
        title="Utterance Centroid-vs-Label Cosine Similarity\n(Mean test utterance embedding per intent → Label embedding)",
        save_path=os.path.join(RESULTS_DIR, "cosine_centroid_vs_label.png"),
        cmap="RdYlGn",
        vmin=-1.0, vmax=1.0,
        highlight_diagonal=True,
        figsize=(12, 10),
    )

    # Diagonal analysis
    diag_centroid = np.diag(centroid_vs_label)
    off_diag_means = []
    for i in range(n_intents):
        row = centroid_vs_label[i]
        off = np.delete(row, i)
        off_diag_means.append(off.mean())
    off_diag_means = np.array(off_diag_means)

    print(f"  Diagonal (alignment): mean={diag_centroid.mean():.4f}, "
          f"min={diag_centroid.min():.4f}, max={diag_centroid.max():.4f}")
    print(f"  Off-diagonal:         mean={off_diag_means.mean():.4f}")
    print(f"  Alignment gap:        {diag_centroid.mean() - off_diag_means.mean():.4f} "
          f"(larger = better utterance-label alignment)")

    # ======================================================================
    #  HEATMAP 3: Per-Sample Utterance-vs-Label (averaged by true intent)
    # ======================================================================
    print("\n-- Heatmap 3: Per-Sample Utterance-vs-Label --")

    avg_sim_matrix = np.zeros((n_intents, n_intents))

    for i in range(n_intents):
        mask_i = label_matrix[:, i] == 1.0
        if mask_i.sum() == 0:
            continue
        intent_utterances = utterance_embeddings[mask_i]
        # Cosine sim between each utterance and each label embedding
        sims = cosine_similarity(intent_utterances, label_embeddings)  # (n_samples, n_intents)
        avg_sim_matrix[i] = sims.mean(axis=0)

    plot_heatmap(
        avg_sim_matrix,
        row_labels=centroid_row_labels,
        col_labels=short_labels,
        title="Per-Sample Utterance-vs-Label Cosine Similarity\n(Avg. cosine similarity of individual test utterances grouped by true intent)",
        save_path=os.path.join(RESULTS_DIR, "cosine_utterance_vs_label.png"),
        cmap="RdYlGn",
        vmin=-1.0, vmax=1.0,
        highlight_diagonal=True,
        figsize=(12, 10),
    )

    diag_avg = np.diag(avg_sim_matrix)
    off_diag_avg = []
    for i in range(n_intents):
        row = avg_sim_matrix[i]
        off = np.delete(row, i)
        off_diag_avg.append(off.mean())
    off_diag_avg = np.array(off_diag_avg)

    print(f"  Diagonal (alignment): mean={diag_avg.mean():.4f}, "
          f"min={diag_avg.min():.4f}, max={diag_avg.max():.4f}")
    print(f"  Off-diagonal:         mean={off_diag_avg.mean():.4f}")
    print(f"  Alignment gap:        {diag_avg.mean() - off_diag_avg.mean():.4f}")

    # ======================================================================
    #  Save metrics summary to CSV
    # ======================================================================
    print("\n-- Saving metrics summary --")

    metrics_rows = []
    for i in range(n_intents):
        metrics_rows.append({
            "intent": intent_list[i],
            "test_sample_count": intent_counts[i],
            "label_self_similarity": round(float(label_sim[i, i]), 4),
            "label_avg_off_diagonal": round(float(
                np.delete(label_sim[i], i).mean()
            ), 4),
            "centroid_to_own_label": round(float(centroid_vs_label[i, i]), 4),
            "centroid_avg_to_other_labels": round(float(
                np.delete(centroid_vs_label[i], i).mean()
            ), 4),
            "centroid_alignment_gap": round(float(
                centroid_vs_label[i, i] - np.delete(centroid_vs_label[i], i).mean()
            ), 4),
            "avg_sample_to_own_label": round(float(avg_sim_matrix[i, i]), 4),
            "avg_sample_to_other_labels": round(float(
                np.delete(avg_sim_matrix[i], i).mean()
            ), 4),
            "sample_alignment_gap": round(float(
                avg_sim_matrix[i, i] - np.delete(avg_sim_matrix[i], i).mean()
            ), 4),
        })

    df_metrics = pd.DataFrame(metrics_rows)
    metrics_path = os.path.join(RESULTS_DIR, "cosine_similarity_metrics.csv")
    df_metrics.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    print(f"  [OK] Saved: {metrics_path}")

    # -- Final summary ----------------------------------------------------
    print(f"\n{'='*65}")
    print("  COSINE SIMILARITY EVALUATION SUMMARY")
    print(f"{'='*65}")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Test samples: {len(df_test)}")
    print(f"\n  Label Embedding Space:")
    print(f"    Mean off-diagonal similarity: {off_diag.mean():.4f}")
    print(f"    Label separation gap:         {diag.mean() - off_diag.mean():.4f}")
    print(f"\n  Utterance-Label Alignment (Centroid):")
    print(f"    Mean diagonal (alignment):    {diag_centroid.mean():.4f}")
    print(f"    Alignment gap:                {diag_centroid.mean() - off_diag_means.mean():.4f}")
    print(f"\n  Utterance-Label Alignment (Per-Sample Avg):")
    print(f"    Mean diagonal (alignment):    {diag_avg.mean():.4f}")
    print(f"    Alignment gap:                {diag_avg.mean() - off_diag_avg.mean():.4f}")

    # Identify weakest intents
    weakest_idx = np.argmin(diag_centroid)
    strongest_idx = np.argmax(diag_centroid)
    print(f"\n  Strongest alignment: {intent_list[strongest_idx]}")
    print(f"    (centroid->label cosine = {diag_centroid[strongest_idx]:.4f})")
    print(f"  Weakest alignment:  {intent_list[weakest_idx]}")
    print(f"    (centroid->label cosine = {diag_centroid[weakest_idx]:.4f})")

    # Identify most confusable intent pair
    # Zero out diagonal for finding max off-diagonal
    sim_no_diag = centroid_vs_label.copy()
    np.fill_diagonal(sim_no_diag, -2.0)
    max_idx = np.unravel_index(sim_no_diag.argmax(), sim_no_diag.shape)
    print(f"\n  Most confusable pair:")
    print(f"    {intent_list[max_idx[0]]} <-> {intent_list[max_idx[1]]}")
    print(f"    (cosine = {centroid_vs_label[max_idx[0], max_idx[1]]:.4f})")

    print(f"\n  Output files:")
    print(f"    evaluation/results/cosine_label_vs_label.png")
    print(f"    evaluation/results/cosine_centroid_vs_label.png")
    print(f"    evaluation/results/cosine_utterance_vs_label.png")
    print(f"    evaluation/results/cosine_similarity_metrics.csv")
    print("=" * 65)


if __name__ == "__main__":
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    main()
