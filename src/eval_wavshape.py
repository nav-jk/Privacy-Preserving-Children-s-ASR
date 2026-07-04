import torch, numpy as np, sys, json
sys.path.insert(0, "/home/WavShape")

from src.models.utils import create_encoder_model

# ── Find latest encoder checkpoint ───────────────────────────────────────────
import glob
checkpoints = sorted(glob.glob("/home/WavShape/exps/training/**/encoder_*.pt",
                               recursive=True))
if not checkpoints:
    checkpoints = sorted(glob.glob("/home/WavShape/exps/training/**/*.pt",
                                   recursive=True))
print("Found checkpoints:", checkpoints)
ENCODER_PATH = checkpoints[-1]
print(f"Using: {ENCODER_PATH}")

# ── Load encoder ──────────────────────────────────────────────────────────────
encoder = create_encoder_model(
    model_name="DenseEncoder",
    model_params={"in_dim": 768, "hidden_dims": [256, 128], "out_dim": 64}
)
encoder.load_state_dict(torch.load(ENCODER_PATH, map_location="cpu"))
encoder.eval()

# ── Transform all splits ───────────────────────────────────────────────────────
OUTPUT_DIR = "/home/embeddings"
for split in ["train", "validation", "test"]:
    emb = torch.FloatTensor(np.load(f"{OUTPUT_DIR}/embeddings_{split}.npy"))
    with torch.no_grad():
        ws_emb = encoder(emb).numpy()
    out_path = f"{OUTPUT_DIR}/embeddings_{split}_wavshape.npy"
    np.save(out_path, ws_emb)
    print(f"{split}: {emb.shape} → {ws_emb.shape}")

print("\nWavShape embeddings saved!")
