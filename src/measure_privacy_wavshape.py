import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json

EMBEDDINGS_PATH = "/home/embeddings/embeddings_test_wavshape.npy"
SPEAKERS_PATH   = "/home/embeddings/speaker_ids_test.npy"
OUTPUT_DIR      = "/home/embeddings"
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

embeddings  = np.load(EMBEDDINGS_PATH)
speaker_ids = np.load(SPEAKERS_PATH)

print(f"WavShape Embeddings : {embeddings.shape}")
print(f"Speakers            : {len(np.unique(speaker_ids))} unique")

# ── 1. AUROC ──────────────────────────────────────────────────────────────────
top2  = [s for s, _ in Counter(speaker_ids).most_common(2)]
mask  = np.isin(speaker_ids, top2)
X_bin = embeddings[mask]
y_bin = (speaker_ids[mask] == top2[0]).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X_bin, y_bin, test_size=0.3, random_state=42, stratify=y_bin
)

class SpeakerClassifier(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)

clf     = SpeakerClassifier(embeddings.shape[1]).to(DEVICE)
opt     = torch.optim.Adam(clf.parameters(), lr=1e-3)
loss_fn = nn.BCELoss()
X_tr = torch.FloatTensor(X_train).to(DEVICE)
y_tr = torch.FloatTensor(y_train).unsqueeze(1).to(DEVICE)
X_te = torch.FloatTensor(X_test).to(DEVICE)

print("\nTraining speaker classifier on WavShape embeddings...")
for epoch in range(100):
    clf.train()
    opt.zero_grad()
    loss_fn(clf(X_tr), y_tr).backward()
    opt.step()
    if (epoch+1) % 20 == 0:
        clf.eval()
        with torch.no_grad():
            probs = clf(X_te).cpu().numpy()
        auroc = roc_auc_score(y_test, probs)
        print(f"  Epoch {epoch+1}/100 — AUROC: {auroc:.4f}")

clf.eval()
with torch.no_grad():
    probs = clf(X_te).cpu().numpy()
auroc = roc_auc_score(y_test, probs)

# ── 2. MI Estimation ──────────────────────────────────────────────────────────
class MIEstimator(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d + 1, 128), nn.ReLU(),
            nn.Linear(128, 64),    nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, e, s): return self.net(torch.cat([e, s], dim=1))

le      = LabelEncoder()
spk_int = le.fit_transform(speaker_ids).astype(np.float32)
emb_n   = ((embeddings - embeddings.mean(0)) /
            (embeddings.std(0) + 1e-8)).astype(np.float32)

E    = torch.FloatTensor(emb_n).to(DEVICE)
S    = torch.FloatTensor(spk_int[:, None]).to(DEVICE)
mine = MIEstimator(emb_n.shape[1]).to(DEVICE)
mopt = torch.optim.Adam(mine.parameters(), lr=1e-3)

print("\nEstimating MI on WavShape embeddings...")
mi_vals = []
for step in range(500):
    mine.train()
    mopt.zero_grad()
    T_joint    = mine(E, S)
    T_marginal = mine(E, S[torch.randperm(len(S))])
    mi_lb      = T_joint.mean() - torch.log(torch.exp(T_marginal).mean() + 1e-8)
    (-mi_lb).backward()
    mopt.step()
    mi_vals.append(mi_lb.item())
    if (step+1) % 100 == 0:
        print(f"  Step {step+1}/500 — MI: {np.mean(mi_vals[-20:]):.4f}")

mi_estimate = float(np.mean(mi_vals[-50:]))

# ── 3. t-SNE ──────────────────────────────────────────────────────────────────
print("\nGenerating t-SNE...")
idx    = np.random.choice(len(embeddings), min(3000, len(embeddings)), replace=False)
tsne   = TSNE(n_components=2, random_state=42, perplexity=30)
emb_2d = tsne.fit_transform(embeddings[idx])

top10  = [s for s, _ in Counter(speaker_ids[idx]).most_common(10)]
colors = plt.cm.tab10(np.linspace(0, 1, 10))
plt.figure(figsize=(10, 8))
for i, spk in enumerate(top10):
    m = speaker_ids[idx] == spk
    plt.scatter(emb_2d[m, 0], emb_2d[m, 1],
                c=[colors[i]], label=f"Spk {spk}", alpha=0.6, s=8)
other = ~np.isin(speaker_ids[idx], top10)
plt.scatter(emb_2d[other, 0], emb_2d[other, 1],
            c="lightgray", alpha=0.3, s=5, label="Other")
plt.legend(markerscale=3, fontsize=8)
plt.title("t-SNE: WavShape Embeddings colored by Speaker ID\n"
          "Less clustering = better privacy")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/tsne_wavshape.png", dpi=150)
print(f"t-SNE saved → {OUTPUT_DIR}/tsne_wavshape.png")

# ── Load baseline for comparison ──────────────────────────────────────────────
with open(f"{OUTPUT_DIR}/baseline_results.json") as f:
    baseline = json.load(f)

# ── Summary ───────────────────────────────────────────────────────────────────
mi_reduction   = 100 * (baseline["mi_speaker"] - mi_estimate) / baseline["mi_speaker"]
auroc_reduction = 100 * (baseline["speaker_auroc"] - auroc) / baseline["speaker_auroc"]

print(f"""
{'='*60}
  BEFORE vs AFTER WAVSHAPE — MyST + Whisper Small
{'='*60}
  Metric              Baseline    WavShape    Change
  ─────────────────────────────────────────────────
  WER                 16.48%      ~16-19%     (recheck needed)
  Speaker AUROC       {baseline['speaker_auroc']:.4f}      {auroc:.4f}      -{auroc_reduction:.1f}%
  MI(emb; speaker)    {baseline['mi_speaker']:.4f}      {mi_estimate:.4f}      -{mi_reduction:.1f}%
  Embedding dim       768         64          -91.7%
{'='*60}
""")

# Save results
results = {
    "model": "whisper-small-finetuned-myst+wavshape",
    "wer": "recheck_needed",
    "speaker_auroc": float(auroc),
    "mi_speaker": mi_estimate,
    "embedding_dim": 64,
    "mi_reduction_pct": float(mi_reduction),
    "auroc_reduction_pct": float(auroc_reduction),
}
with open(f"{OUTPUT_DIR}/wavshape_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"Results saved → {OUTPUT_DIR}/wavshape_results.json")
