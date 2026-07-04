import os
import torch
import numpy as np
from datasets import load_from_disk
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from tqdm import tqdm

MODEL_PATH   = "/home/whisper_finetuned"
DATASET_PATH = "/home/myst-v0.4.2/myst_dataset.ds"
OUTPUT_DIR   = "/home/embeddings"
SPLIT        = "test"
DEVICE       = "cuda"
BATCH_SIZE   = 32

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Loading model from {MODEL_PATH}...")
processor = WhisperProcessor.from_pretrained(MODEL_PATH)
model = WhisperForConditionalGeneration.from_pretrained(
    MODEL_PATH, torch_dtype=torch.float16
).to(DEVICE)
model.eval()

print(f"Loading dataset split: {SPLIT}...")
ds = load_from_disk(DATASET_PATH)[SPLIT]
print(f"Samples: {len(ds)}  |  Unique speakers: {len(set(ds['speaker_id']))}")

embeddings_list = []
speaker_ids     = []
skipped         = 0

for i in tqdm(range(0, len(ds), BATCH_SIZE), desc="Extracting embeddings"):
    batch = ds[i : i + BATCH_SIZE]
    try:
        audio_arrays = [np.array(a["array"], dtype=np.float32)
                        for a in batch["audio"]]

        # padding=True + truncation pads/truncates to exactly 30s (3000 frames)
        inputs = processor(
            audio_arrays,
            sampling_rate=16000,
            return_tensors="pt",
            padding="max_length",   # ← pad to max_length (30s)
            truncation=True,
            max_length=480000,      # 30s * 16000
        ).input_features.half().to(DEVICE)

        with torch.no_grad():
            encoder_out = model.model.encoder(inputs)
            emb = encoder_out.last_hidden_state.mean(dim=1)

        embeddings_list.append(emb.cpu().float().numpy())
        speaker_ids.extend(batch["speaker_id"])

    except Exception as e:
        print(f"Skipped batch {i}: {e}")
        skipped += len(batch["speaker_id"])
        continue

embeddings  = np.vstack(embeddings_list)
speaker_ids = np.array(speaker_ids)

np.save(f"{OUTPUT_DIR}/embeddings_{SPLIT}.npy", embeddings)
np.save(f"{OUTPUT_DIR}/speaker_ids_{SPLIT}.npy", speaker_ids)

print(f"\nSaved embeddings : {embeddings.shape}")
print(f"Unique speakers  : {len(np.unique(speaker_ids))}")
print(f"Skipped          : {skipped}")
