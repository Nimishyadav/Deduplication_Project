"""
=============================================================
NLP TEXT DEDUPLICATION — Research Paper Project
=============================================================
Models compared:
  1. Baseline  : TF-IDF + Cosine Similarity
  2. Model 2   : MinHash (Locality-Sensitive Hashing)
  3. Model 3   : Logistic Regression on hand-crafted features

Dataset : Synthetic Quora-style question pairs (self-contained)
Metrics : Precision, Recall, F1, Accuracy, ROC-AUC
Outputs : CSV results + 4 publication-ready plots
=============================================================
"""

import random
import math
import hashlib
import time
import csv
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             accuracy_score, roc_auc_score,
                             confusion_matrix, roc_curve)
from sklearn.preprocessing import StandardScaler

random.seed(42)
np.random.seed(42)

OUTPUT_DIR = "/home/claude/dedup_project/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# STYLE
# ─────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})

COLORS = {
    "tfidf":   "#378ADD",
    "minhash": "#1D9E75",
    "logreg":  "#BA7517",
    "accent":  "#534AB7",
}

# ─────────────────────────────────────────────
# 1. DATASET GENERATION
# ─────────────────────────────────────────────

BASE_QUESTIONS = [
    "What is the best way to learn Python programming?",
    "How do I improve my English speaking skills?",
    "What are the health benefits of drinking green tea?",
    "How can I lose weight quickly and safely?",
    "What is machine learning and how does it work?",
    "How do I start a successful business from scratch?",
    "What are the symptoms of diabetes?",
    "How can I improve my memory and concentration?",
    "What is the difference between AI and machine learning?",
    "How do I cook a perfect pasta dish at home?",
    "What are the best practices for software development?",
    "How can I reduce stress and anxiety in daily life?",
    "What is quantum computing and its applications?",
    "How do I prepare for a job interview?",
    "What are the causes of climate change?",
    "How can I save money on a tight budget?",
    "What is blockchain technology?",
    "How do I learn a new language quickly?",
    "What are the benefits of regular exercise?",
    "How does the stock market work?",
    "What is deep learning and neural networks?",
    "How can I improve my writing skills?",
    "What are the symptoms of depression?",
    "How do I negotiate a salary increase?",
    "What is data science and its applications?",
    "How can I be more productive at work?",
    "What are renewable energy sources?",
    "How do I improve my sleep quality?",
    "What is the difference between Java and Python?",
    "How can I build a strong personal brand online?",
]

PARAPHRASE_TEMPLATES = [
    ("What is the best way to {}", "What is the most effective method to {}"),
    ("How do I {}", "What are the steps to {}"),
    ("How can I {}", "What is the best approach to {}"),
    ("What are the {}", "Can you list the {}"),
    ("What is {}", "Could you explain {}"),
]

def paraphrase(text):
    """Generate a near-duplicate paraphrase."""
    swaps = [
        ("What is the best way to", "What is the most effective approach to"),
        ("How do I", "What are the steps to"),
        ("How can I", "What is the best way to"),
        ("What are the", "Can you describe the"),
        ("What is", "Please explain"),
        ("learn", "study"),
        ("improve", "enhance"),
        ("quickly", "fast"),
        ("skills", "abilities"),
        ("benefits", "advantages"),
        ("symptoms", "signs"),
        ("reduce", "decrease"),
        ("build", "develop"),
        ("start", "begin"),
    ]
    result = text
    changed = False
    for old, new in swaps:
        if old.lower() in result.lower() and not changed:
            result = result.replace(old, new, 1)
            changed = True
    # minor word shuffle for variety
    words = result.split()
    if len(words) > 5 and random.random() < 0.3:
        i = random.randint(1, len(words) - 2)
        words[i], words[i-1] = words[i-1], words[i]
    return " ".join(words)

def make_unrelated(text, all_questions):
    """Pick a clearly different question."""
    candidates = [q for q in all_questions if q != text]
    return random.choice(candidates)

def generate_dataset(n_pairs=800):
    """Create balanced duplicate / non-duplicate pairs."""
    pairs = []
    half = n_pairs // 2

    # Duplicate pairs (label=1)
    for _ in range(half):
        q1 = random.choice(BASE_QUESTIONS)
        q2 = paraphrase(q1)
        pairs.append((q1, q2, 1))

    # Non-duplicate pairs (label=0)
    for _ in range(half):
        q1 = random.choice(BASE_QUESTIONS)
        q2 = make_unrelated(q1, BASE_QUESTIONS)
        while q2 == q1:
            q2 = make_unrelated(q1, BASE_QUESTIONS)
        pairs.append((q1, q2, 0))

    random.shuffle(pairs)
    df = pd.DataFrame(pairs, columns=["question1", "question2", "is_duplicate"])
    return df

print("=" * 60)
print("NLP TEXT DEDUPLICATION — Research Project")
print("=" * 60)
print("\n[1/5] Generating dataset...")
df = generate_dataset(800)
print(f"      Dataset size   : {len(df)} pairs")
print(f"      Duplicates     : {df['is_duplicate'].sum()}")
print(f"      Non-duplicates : {(df['is_duplicate'] == 0).sum()}")

df.to_csv(f"{OUTPUT_DIR}/dataset.csv", index=False)

# Train/test split
train_df, test_df = train_test_split(df, test_size=0.25, random_state=42,
                                     stratify=df["is_duplicate"])

# ─────────────────────────────────────────────
# 2. MODEL 1 — TF-IDF + COSINE SIMILARITY
# ─────────────────────────────────────────────
print("\n[2/5] Running Model 1: TF-IDF + Cosine Similarity...")

def tfidf_predict(train_df, test_df, threshold=0.45):
    all_texts = list(train_df["question1"]) + list(train_df["question2"])
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
    vectorizer.fit(all_texts)

    q1_vecs = vectorizer.transform(test_df["question1"])
    q2_vecs = vectorizer.transform(test_df["question2"])

    scores = []
    for i in range(q1_vecs.shape[0]):
        sim = cosine_similarity(q1_vecs[i], q2_vecs[i])[0][0]
        scores.append(sim)

    scores = np.array(scores)
    preds  = (scores >= threshold).astype(int)
    return preds, scores

t0 = time.time()
tfidf_preds, tfidf_scores = tfidf_predict(train_df, test_df)
tfidf_time = time.time() - t0

y_test = test_df["is_duplicate"].values

tfidf_metrics = {
    "Model": "TF-IDF + Cosine",
    "Accuracy":  round(accuracy_score(y_test, tfidf_preds), 4),
    "Precision": round(precision_score(y_test, tfidf_preds), 4),
    "Recall":    round(recall_score(y_test, tfidf_preds), 4),
    "F1":        round(f1_score(y_test, tfidf_preds), 4),
    "ROC-AUC":   round(roc_auc_score(y_test, tfidf_scores), 4),
    "Time(s)":   round(tfidf_time, 3),
}
print(f"      F1={tfidf_metrics['F1']}  AUC={tfidf_metrics['ROC-AUC']}  ({tfidf_time:.3f}s)")

# ─────────────────────────────────────────────
# 3. MODEL 2 — MINHASH (LSH-style)
# ─────────────────────────────────────────────
print("\n[3/5] Running Model 2: MinHash Similarity...")

def shingle(text, k=3):
    """Character-level k-shingles."""
    text = text.lower().strip()
    return set(text[i:i+k] for i in range(len(text) - k + 1))

def minhash_signature(shingle_set, n_hashes=128):
    """Compute MinHash signature using multiple hash functions."""
    sig = []
    for seed in range(n_hashes):
        min_val = float("inf")
        for s in shingle_set:
            h = int(hashlib.md5((str(seed) + s).encode()).hexdigest(), 16)
            if h < min_val:
                min_val = h
        sig.append(min_val)
    return sig

def jaccard_from_minhash(sig1, sig2):
    """Estimate Jaccard similarity from two MinHash signatures."""
    matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
    return matches / len(sig1)

def minhash_predict(test_df, threshold=0.25, n_hashes=128):
    scores = []
    for _, row in test_df.iterrows():
        sh1 = shingle(row["question1"])
        sh2 = shingle(row["question2"])
        if not sh1 or not sh2:
            scores.append(0.0)
            continue
        sig1 = minhash_signature(sh1, n_hashes)
        sig2 = minhash_signature(sh2, n_hashes)
        scores.append(jaccard_from_minhash(sig1, sig2))
    scores = np.array(scores)
    preds  = (scores >= threshold).astype(int)
    return preds, scores

t0 = time.time()
minhash_preds, minhash_scores = minhash_predict(test_df, n_hashes=64)
minhash_time = time.time() - t0

minhash_metrics = {
    "Model": "MinHash",
    "Accuracy":  round(accuracy_score(y_test, minhash_preds), 4),
    "Precision": round(precision_score(y_test, minhash_preds), 4),
    "Recall":    round(recall_score(y_test, minhash_preds), 4),
    "F1":        round(f1_score(y_test, minhash_preds), 4),
    "ROC-AUC":   round(roc_auc_score(y_test, minhash_scores), 4),
    "Time(s)":   round(minhash_time, 3),
}
print(f"      F1={minhash_metrics['F1']}  AUC={minhash_metrics['ROC-AUC']}  ({minhash_time:.3f}s)")

# ─────────────────────────────────────────────
# 4. MODEL 3 — LOGISTIC REGRESSION (ML classifier)
# ─────────────────────────────────────────────
print("\n[4/5] Running Model 3: Logistic Regression Classifier...")

def extract_features(df, vectorizer=None, fit=False):
    """
    Hand-crafted similarity features for each pair:
    - TF-IDF cosine similarity (unigram + bigram)
    - Jaccard similarity of word sets
    - Length difference ratio
    - Common word ratio
    - Character n-gram overlap
    - Edit distance (normalised)
    """
    if fit:
        all_texts = list(df["question1"]) + list(df["question2"])
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=3000)
        vectorizer.fit(all_texts)

    features = []
    for _, row in df.iterrows():
        q1, q2 = str(row["question1"]).lower(), str(row["question2"]).lower()

        # TF-IDF cosine
        v1 = vectorizer.transform([q1])
        v2 = vectorizer.transform([q2])
        cos_sim = cosine_similarity(v1, v2)[0][0]

        # Jaccard on words
        s1, s2 = set(q1.split()), set(q2.split())
        jac = len(s1 & s2) / max(len(s1 | s2), 1)

        # Length ratio
        len_diff = abs(len(q1) - len(q2)) / max(len(q1), len(q2), 1)

        # Common word ratio
        common = len(s1 & s2) / max(len(s1), len(s2), 1)

        # Character bigram overlap
        cbi1 = set(q1[i:i+2] for i in range(len(q1)-1))
        cbi2 = set(q2[i:i+2] for i in range(len(q2)-1))
        bigram_ov = len(cbi1 & cbi2) / max(len(cbi1 | cbi2), 1)

        # Normalised edit distance (Levenshtein approximation via shared prefix)
        def edit_dist_approx(a, b):
            # Simple DP edit distance (capped for speed)
            a, b = a[:50], b[:50]
            m, n = len(a), len(b)
            dp = list(range(n + 1))
            for i in range(1, m + 1):
                new_dp = [i] + [0] * n
                for j in range(1, n + 1):
                    if a[i-1] == b[j-1]:
                        new_dp[j] = dp[j-1]
                    else:
                        new_dp[j] = 1 + min(dp[j], new_dp[j-1], dp[j-1])
                dp = new_dp
            return dp[n]

        ed = edit_dist_approx(q1, q2)
        norm_ed = 1 - ed / max(len(q1), len(q2), 1)

        features.append([cos_sim, jac, len_diff, common, bigram_ov, norm_ed])

    return np.array(features), vectorizer

t0 = time.time()
X_train, vec = extract_features(train_df, fit=True)
X_test,  _   = extract_features(test_df, vectorizer=vec)
y_train = train_df["is_duplicate"].values

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

clf = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
clf.fit(X_train, y_train)

logreg_preds  = clf.predict(X_test)
logreg_scores = clf.predict_proba(X_test)[:, 1]
logreg_time   = time.time() - t0

logreg_metrics = {
    "Model": "Logistic Regression",
    "Accuracy":  round(accuracy_score(y_test, logreg_preds), 4),
    "Precision": round(precision_score(y_test, logreg_preds), 4),
    "Recall":    round(recall_score(y_test, logreg_preds), 4),
    "F1":        round(f1_score(y_test, logreg_preds), 4),
    "ROC-AUC":   round(roc_auc_score(y_test, logreg_scores), 4),
    "Time(s)":   round(logreg_time, 3),
}
print(f"      F1={logreg_metrics['F1']}  AUC={logreg_metrics['ROC-AUC']}  ({logreg_time:.3f}s)")

# ─────────────────────────────────────────────
# 5. RESULTS TABLE
# ─────────────────────────────────────────────
results = pd.DataFrame([tfidf_metrics, minhash_metrics, logreg_metrics])
results.to_csv(f"{OUTPUT_DIR}/results.csv", index=False)

print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)
print(results.to_string(index=False))

# ─────────────────────────────────────────────
# 6. PLOTS
# ─────────────────────────────────────────────
print("\n[5/5] Generating plots...")

model_names  = ["TF-IDF + Cosine", "MinHash", "Logistic Regression"]
model_colors = [COLORS["tfidf"], COLORS["minhash"], COLORS["logreg"]]
all_preds    = [tfidf_preds,  minhash_preds,  logreg_preds]
all_scores   = [tfidf_scores, minhash_scores, logreg_scores]
all_metrics  = [tfidf_metrics, minhash_metrics, logreg_metrics]

# ── Plot 1: Grouped bar chart of metrics ──
fig, ax = plt.subplots(figsize=(10, 5))
metric_keys = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
x = np.arange(len(metric_keys))
width = 0.25

for i, (name, color, m) in enumerate(zip(model_names, model_colors, all_metrics)):
    vals = [m[k] for k in metric_keys]
    bars = ax.bar(x + i * width, vals, width, label=name, color=color, alpha=0.88,
                  edgecolor="white", linewidth=0.8)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.2f}", ha="center", va="bottom", fontsize=8.5)

ax.set_xticks(x + width)
ax.set_xticklabels(metric_keys)
ax.set_ylim(0, 1.12)
ax.set_ylabel("Score")
ax.set_title("Model Comparison — All Metrics")
ax.legend(frameon=False)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/plot1_metrics_comparison.png", bbox_inches="tight")
plt.close()

# ── Plot 2: ROC Curves ──
fig, ax = plt.subplots(figsize=(7, 6))
for name, color, scores in zip(model_names, model_colors, all_scores):
    fpr, tpr, _ = roc_curve(y_test, scores)
    auc = roc_auc_score(y_test, scores)
    ax.plot(fpr, tpr, color=color, linewidth=2, label=f"{name} (AUC={auc:.3f})")

ax.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves — All Models")
ax.legend(frameon=False, loc="lower right")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/plot2_roc_curves.png", bbox_inches="tight")
plt.close()

# ── Plot 3: Confusion Matrices ──
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, name, color, preds in zip(axes, model_names, model_colors, all_preds):
    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap=sns.light_palette(color, as_cmap=True),
                xticklabels=["Not Dup", "Duplicate"],
                yticklabels=["Not Dup", "Duplicate"],
                ax=ax, linewidths=0.5, cbar=False)
    ax.set_title(name, fontsize=11)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
plt.suptitle("Confusion Matrices", fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/plot3_confusion_matrices.png", bbox_inches="tight")
plt.close()

# ── Plot 4: Score distributions ──
fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)
for ax, name, color, scores in zip(axes, model_names, model_colors, all_scores):
    dup_scores    = scores[y_test == 1]
    nondup_scores = scores[y_test == 0]
    ax.hist(nondup_scores, bins=25, alpha=0.65, color="#888780", label="Not Duplicate",
            edgecolor="white")
    ax.hist(dup_scores,    bins=25, alpha=0.75, color=color,    label="Duplicate",
            edgecolor="white")
    ax.set_title(name, fontsize=11)
    ax.set_xlabel("Similarity Score")
    ax.set_ylabel("Count")
    ax.legend(frameon=False, fontsize=9)
plt.suptitle("Score Distributions by Class", fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/plot4_score_distributions.png", bbox_inches="tight")
plt.close()

print(f"\n  Plots saved to {OUTPUT_DIR}/")
print("\n" + "=" * 60)
print("PROJECT COMPLETE")
print("=" * 60)
print(f"  dataset.csv                   — raw data ({len(df)} pairs)")
print(f"  results.csv                   — model comparison table")
print(f"  plot1_metrics_comparison.png  — grouped bar chart")
print(f"  plot2_roc_curves.png          — ROC curves")
print(f"  plot3_confusion_matrices.png  — confusion matrices (3 models)")
print(f"  plot4_score_distributions.png — similarity score histograms")
print("=" * 60)
