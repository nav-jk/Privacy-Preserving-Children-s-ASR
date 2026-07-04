# Privacy-Preserving Children's Speech Recognition
## Research Progress

> **Last updated:** July 4, 2026  
> **Model:** Whisper Small | **Dataset:** MyST v0.4.2

---

## Phase 1: ASR Baseline 

### Dataset
| Split | Samples |
|-------|---------|
| Train | 36,544 |
| Validation | 6,013 |
| Test | 6,321 |
| **Total in corpus** | **102,331** (91 speakers, MyST v0.4.2) |

### Training Configuration
| Parameter | Value |
|-----------|-------|
| Base model | openai/whisper-small |
| GPU | NVIDIA H200 (141 GB VRAM) |
| Precision | BF16 |
| Learning rate | 1e-6 |
| Warmup steps | 100 |
| Max steps | 1,000 |
| Batch size | 64 |
| Framework | HuggingFace Seq2SeqTrainer |

### Training Results
| Step | Train Loss | Val Loss | WER |
|------|-----------|----------|-----|
| 0 (base) | — | — | ~19% |
| 500 | 0.3370 | 0.4411 | 17.26% |
| 1000 | 0.3131 | 0.4309 | **16.48%** |

**Final WER: 16.48%** — 2.5 percentage point improvement over base Whisper Small on children's speech.

---

## Phase 2: Privacy Baseline Measurement 

### Method
- Extracted Whisper encoder embeddings for all 13,169 test samples → `(13169, 768)`
- Trained lightweight DNN classifier to predict speaker ID from embeddings (AUROC)
- Estimated mutual information using Donsker-Varadhan / MINE formulation
- Generated t-SNE visualization colored by speaker ID

### Results
| Metric | Value | Interpretation |
|--------|-------|----------------|
| WER | 16.48% | ASR utility |
| Speaker AUROC | 0.9953 | Near-perfect speaker recovery = severe leakage |
| MI(emb; speaker) | 1.5556 | Very high identity information in embeddings |

---

## Phase 3: WavShape Privacy Layer 

### Setup
- Cloned WavShape repo (UTAustin-SwarmLab/WavShape)
- Fed pre-extracted Whisper embeddings `(76924, 768)` as input
- Sensitive attribute S(x): speaker_id (91 unique speakers)
- Task attribute T(x): utility label (ones — preserve all content)
- WavShape projection: 768 → 64 dimensions

### WavShape Configuration
| Parameter | Value |
|-----------|-------|
| Encoder | DenseEncoder 768→256→128→64 |
| Encoder epochs | 10 |
| MINE epochs | 100 (utility + privacy) |
| Early stopping patience | 20 |
| Beta (privacy weight) | 1.0 |
| Batch size | 1,024 |
| GPU | NVIDIA A30 |

### Final MI after training (epoch 9)
```
Utility MI  I(embedding; task):    ~0.0
Privacy MI  I(embedding; speaker):  0.190  ← down from 1.5556
```

---

## Phase 4: Privacy Evaluation After WavShape 

### Results — Before vs After

| Metric | Baseline (Whisper) | After WavShape | Reduction |
|--------|-------------------|----------------|-----------|
| **WER** | **16.48%** | 18.45% | 1.97% |
| **Speaker AUROC** | **0.9953** | **0.7160** | **-28.1%** |
| **MI(emb; speaker)** | **1.5556** | **0.4933** | **-68.3%** |
| Embedding dim | 768 | 64 | -91.7% |

### Comparison with WavShape paper
| Dataset | MI Reduction | AUROC Reduction |
|---------|-------------|-----------------|
| Common Voice (paper) | ~81% | ~25.8% |
| VCTK (paper) | ~83% | ~38.4% |
| **MyST children's speech (ours)** | **68.3%** | **28.1%** |

Results are consistent with the paper's findings on a novel children's speech dataset not used in the original work.

---

## Saved Artifacts
```
/home/embeddings/
├── embeddings_{train,validation,test}.npy          # Whisper embeddings (768-dim)
├── embeddings_{train,validation,test}_wavshape.npy # WavShape embeddings (64-dim)
├── speaker_ids_{train,validation,test}.npy         # Speaker labels
├── baseline_results.json                           # Before WavShape metrics
├── wavshape_results.json                           # After WavShape metrics
├── tsne_baseline.png                               # t-SNE before
└── tsne_wavshape.png                               # t-SNE after

/home/WavShape/exps/training/2026-07-04_07-42-02/
└── encoder_weights/model_9.pt                      # Trained WavShape encoder
```

---

## Summary of All Results

```
Base Whisper Small (no fine-tuning):   WER  ~19%
After fine-tuning on MyST:             WER   16.48%
Speaker leakage AUROC  (baseline):           0.9953
Speaker leakage AUROC  (WavShape):           0.7160   (-28.1%)
MI(emb; speaker)       (baseline):           1.5556
MI(emb; speaker)       (WavShape):           0.4933   (-68.3%)
Embedding compression: 768 → 64             (-91.7%)
```
