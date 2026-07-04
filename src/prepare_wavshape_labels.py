import numpy as np
from sklearn.preprocessing import LabelEncoder

OUTPUT_DIR = "/home/embeddings"
le = LabelEncoder()

# Fit on all splits combined for consistent encoding
all_spk = np.concatenate([
    np.load(f"{OUTPUT_DIR}/speaker_ids_train.npy"),
    np.load(f"{OUTPUT_DIR}/speaker_ids_validation.npy"),
    np.load(f"{OUTPUT_DIR}/speaker_ids_test.npy"),
])
le.fit(all_spk)

for split in ["train", "validation", "test"]:
    spk = np.load(f"{OUTPUT_DIR}/speaker_ids_{split}.npy")
    spk_int = le.transform(spk).astype(np.float32)
    # Normalize to [0,1] as WavShape expects
    spk_norm = spk_int / (len(le.classes_) - 1)
    np.save(f"{OUTPUT_DIR}/speaker_label_{split}.npy", spk_norm)
    # Task label: use speaker count as proxy (ones = all same utility)
    # Since we don't have per-utterance transcription labels as floats,
    # use a binary label: 1.0 for all (preserve everything = utility)
    task_label = np.ones(len(spk), dtype=np.float32)
    np.save(f"{OUTPUT_DIR}/task_label_{split}.npy", task_label)
    print(f"{split}: embeddings={len(spk)}, speakers={len(np.unique(spk))}")

print("\nLabel files saved:")
print("  speaker_label_train.npy    ← sensitive S(x)")
print("  task_label_train.npy       ← utility T(x)")
