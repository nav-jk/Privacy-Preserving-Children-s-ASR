import os, sys, torch, numpy as np
from datasets import load_from_disk
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from tqdm import tqdm

MODEL_PATH   = "/home/whisper_finetuned"
PROCESSOR_PATH = "openai/whisper-small"
DATASET_PATH = "/home/myst-v0.4.2/myst_dataset.ds"
OUTPUT_DIR   = "/home/embeddings"
DEVICE       = "cuda"
BATCH_SIZE   = 32

os.makedirs(OUTPUT_DIR, exist_ok=True)
processor = WhisperProcessor.from_pretrained(PROCESSOR_PATH, language="English", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained(
    MODEL_PATH, torch_dtype=torch.float16
).to(DEVICE)
model.eval()

full_ds = load_from_disk(DATASET_PATH)

for SPLIT in ["train", "validation", "test"]:
    # Skip if already done
    if os.path.exists(f"{OUTPUT_DIR}/embeddings_{SPLIT}.npy"):
        print(f"Skipping {SPLIT} — already exists")
        continue

    ds = full_ds[SPLIT]
    print(f"\nProcessing {SPLIT}: {len(ds)} samples")

    embeddings_list, speaker_ids = [], []

    for i in tqdm(range(0, len(ds), BATCH_SIZE), desc=SPLIT):
        batch = ds[i : i + BATCH_SIZE]
        try:
            audio_arrays = [np.array(a["array"], dtype=np.float32) for a in batch["audio"]]
            inputs = processor(
                audio_arrays, sampling_rate=16000, return_tensors="pt",
                padding="max_length", truncation=True, max_length=480000,
            ).input_features.half().to(DEVICE)
            with torch.no_grad():
                emb = model.model.encoder(inputs).last_hidden_state.mean(dim=1)
            embeddings_list.append(emb.cpu().float().numpy())
            speaker_ids.extend(batch["speaker_id"])
        except Exception as e:
            print(f"  Skipped batch {i}: {e}")

    embeddings = np.vstack(embeddings_list)
    speaker_ids = np.array(speaker_ids)
    np.save(f"{OUTPUT_DIR}/embeddings_{SPLIT}.npy", embeddings)
    np.save(f"{OUTPUT_DIR}/speaker_ids_{SPLIT}.npy", speaker_ids)
    print(f"Saved {SPLIT}: {embeddings.shape}")
