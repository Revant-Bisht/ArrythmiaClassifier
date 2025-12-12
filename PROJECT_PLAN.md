# Arrhythmia Classifier — Project Plan

**Author:** Revant Bisht
**Dataset:** PTB-XL (PhysioNet) — 21,837 12-lead clinical ECG records, 10s each, 500 Hz
**Goal:** A production-quality ML project demonstrating EE + ML depth, deployed as an interactive web experience for recruiters.

---

## Vision: The End Product

### Website Structure

```
portfolio.revantbisht.com/
├── (main site)  About Me, Projects, Contact
└── /arrhythmia-classifier  ← standalone sub-experience
```

### Arrhythmia Classifier Page — Landing (Page 1)

The user lands on a **single, cinematic page**:

1. **Hero Section — Animated Model Visualization**
   - A live, canvas/WebGL animation showing:
     - A 12-lead ECG waveform feeding into the model from the left
     - Model architecture decomposed — convolutional layers shown as filter banks, activations lighting up as data flows through
     - Output: a clean "Arrhythmia Report" card appearing on the right
   - This plays on loop, auto-pausing on hover
   - Typography: "Detecting Arrhythmia with Deep Learning on 12-Lead ECG"

2. **Try It Here — Interactive Demo**
   - Input panel (left side):
     - Patient metadata: Age, Sex
     - ECG input: Upload a `.csv`/`.hea/.dat` file OR select a random sample from the test set
     - A real-time ECG waveform preview (rendered with D3.js or Plotly)
   - Output panel (right side):
     - Predicted class (NORM / MI / STTC / CD / HYP) with confidence bar
     - Grad-CAM heatmap overlaid on the ECG — highlights which part of the signal drove the prediction
     - Plain-English clinical summary for each class
   - Inference runs via a lightweight FastAPI backend (or ONNX in-browser for speed)

3. **Scroll CTA**
   - At the bottom: "See How We Built This ↓"

---

### Arrhythmia Classifier Page — Technical Blog (Page 2)

Scrollable long-form page with clean section anchors:

| Section | Content |
|---|---|
| The Problem | What arrhythmia is, clinical stakes, why automated detection matters |
| The Data | PTB-XL overview, label hierarchy (71 → 5 superclasses), class imbalance, signal properties |
| EDA | Distribution plots, lead correlations, signal quality analysis, sample waveform gallery |
| Feature Engineering | Why raw waveforms > hand-crafted features for DL; R-peak detection for baseline |
| Model Architecture | SOTA choice (see below), architecture diagram, design rationale |
| Training | Loss function (BCE + class weighting), optimizer, augmentation strategy, hardware |
| Results | AUC-ROC per class, confusion matrix, comparison to PTB-XL benchmark baselines |
| Explainability | Grad-CAM adapted for 1D CNN, clinical validation of highlighted regions |
| Deployment | ONNX export, FastAPI, latency benchmarks |

---

## SOTA for PTB-XL Arrhythmia Classification

### The Benchmark Paper
Strodthoff et al. (2020), *"Deep Learning for ECG Analysis: Benchmarks and Insights from PTB-XL"*
This paper is the canonical reference — it establishes baselines on the exact dataset you're using.

### Top Performing Architectures (5-class superdiagnostic AUC)

| Model | Macro AUC | Notes |
|---|---|---|
| **xresnet1d101** | ~0.931 | 1D ResNet adapted for time-series. Best single model. |
| **inception1d** (InceptionTime) | ~0.925 | Multi-scale conv, fast inference |
| **LSTM + Attention** | ~0.907 | Good baseline, interpretable attention weights |
| **1D CNN (simple)** | ~0.890 | Solid floor, easiest to explain |
| Transformer (ECG-specific) | ~0.935 | Cutting-edge but heavy; overkill for portfolio |

### Recommended Architecture: **InceptionTime + Attention**

**Why this choice:**
- Strong AUC (~0.925) — competitive with the very best
- Multi-scale convolutions (kernel sizes 10, 20, 40) mirror what EEs intuitively know: different arrhythmias manifest at different timescales
- Attention layer gives you a natural, explainable "which timestep mattered" signal
- Smaller and faster than xresnet1d101 — critical for demo latency
- Visualizing the multi-scale filters is architecturally beautiful for the animation

**Architecture sketch:**
```
Input: (batch, 12, 1000)  ← 12 leads × 1000 timesteps at 100Hz
  │
  ├─ InceptionModule(k=10, 20, 40) × 3 blocks
  │     Each block: parallel convs + MaxPool branch + BN + ReLU
  │     Shortcut connections (residual)
  │
  ├─ Temporal Attention (soft attention over time axis)
  │
  ├─ GlobalAveragePool → (batch, 128)
  │
  └─ Linear(128 → 5) + Sigmoid  ← multi-label output
```

**Loss:** Binary Cross-Entropy with label smoothing + class-frequency weighting
**Augmentation:** Gaussian noise, lead dropout, random time-shift
**Explainability:** Grad-CAM on last inception block → heatmap on ECG signal

---

## Development Stages

### Stage 1 — Data & Environment Setup
- [ ] Download PTB-XL from PhysioNet (`wget` + `wfdb`)
- [ ] Parse records with `wfdb` library; extract 12-lead arrays at 100 Hz
- [ ] Map 71 SCP codes → 5 superclasses per the PTB-XL label schema
- [ ] Train/val/test split using the pre-defined `strat_fold` column (stratified 10-fold)
- [ ] Build a `torch.Dataset` class for ECG tensors + labels
- **Deliverable:** Clean dataset loader, class distribution plot, sample waveform gallery

### Stage 2 — EDA & Signal Analysis
- [ ] Plot all 12 leads for NORM vs. each pathology class
- [ ] Analyze class imbalance (NORM ~28%, MI ~20%, STTC ~24%, CD ~22%, HYP ~14%)
- [ ] Signal quality check: SNR estimation, artifact detection
- [ ] R-peak detection (Pan-Tompkins) for sanity check and potential feature
- [ ] Correlation between leads; PCA on frequency domain features
- **Deliverable:** EDA notebook → figures used directly in the blog section

### Stage 3 — Model Implementation
- [ ] Implement InceptionTime in PyTorch (from scratch, not a library — shows skill)
- [ ] Add temporal soft-attention head
- [ ] Implement training loop with AMP (mixed precision), cosine LR schedule
- [ ] Implement Grad-CAM for 1D convolutions
- [ ] Train baseline (simple 1D CNN) first to validate pipeline
- [ ] Train full InceptionTime + Attention model
- **Deliverable:** Trained model checkpoint, training curves, AUC-ROC ≥ 0.92

### Stage 4 — Evaluation & Explainability
- [ ] Per-class AUC-ROC, PR curves, confusion matrix on held-out test set
- [ ] Compare to PTB-XL paper baselines
- [ ] Generate Grad-CAM heatmaps for 20 representative test samples per class
- [ ] Validate heatmaps clinically: do they highlight ST segments for STTC? QRS for CD?
- **Deliverable:** Results table, heatmap gallery, model card

### Stage 5 — Export & Backend
- [ ] Export model to ONNX (`torch.onnx.export`)
- [ ] Build FastAPI endpoint: accepts ECG array + age/sex → returns class probs + Grad-CAM array
- [ ] Quantize model (INT8) for faster inference if needed
- [ ] Containerize with Docker
- [ ] Deploy to Fly.io or Railway (free tier, fast cold starts)
- **Deliverable:** Live API endpoint, <500ms inference latency

### Stage 6 — Frontend (The Showpiece)
- [ ] Framework: **Next.js** (fast, SEO-friendly, easy to integrate with portfolio)
- [ ] Animation: **Three.js** or **Framer Motion** for the model visualization hero
- [ ] ECG rendering: **D3.js** for the waveform preview and Grad-CAM overlay
- [ ] Build the interactive demo panel (file upload + random sample)
- [ ] Build the blog section with code snippets, figures from EDA, architecture diagrams
- [ ] Responsive design (mobile-friendly)
- [ ] Lighthouse score ≥ 90 on performance
- **Deliverable:** Deployed website at a custom subdomain

### Stage 7 — Polish & Portfolio Integration
- [ ] Record a 30-second screen capture for LinkedIn/resume
- [ ] Write a 3-paragraph project summary for the main portfolio page
- [ ] Add GitHub repo with clean README, model card, and reproducibility instructions
- [ ] Link from main portfolio site

---

## Tech Stack

| Layer | Choice | Reason |
|---|---|---|
| Data | `wfdb`, `pandas`, `numpy` | Standard PhysioNet tools |
| ML | PyTorch, scikit-learn | Industry standard |
| Training infra | Lightning or plain PyTorch + W&B | Experiment tracking |
| Export | ONNX Runtime | Fast cross-platform inference |
| Backend | FastAPI + Uvicorn | Lightweight, async |
| Container | Docker | Reproducible deploy |
| Hosting | Fly.io / Railway | Free tier, fast, no cold start penalty |
| Frontend | Next.js + TypeScript | Performance + DX |
| Visualization | D3.js, Three.js / Framer Motion | ECG rendering + animations |
| Styling | Tailwind CSS | Fast, consistent |

---

## Success Metrics

| Metric | Target |
|---|---|
| Macro AUC-ROC (5-class) | ≥ 0.92 |
| Demo inference latency | < 500ms end-to-end |
| Lighthouse performance score | ≥ 90 |
| Grad-CAM clinical plausibility | Heatmaps highlight known diagnostic regions |
| Recruiter legibility | Non-technical person understands the demo in < 30s |

---

## Key References

- **PTB-XL dataset:** Wagner et al. (2020). *PTB-XL, a large publicly available electrocardiography dataset.* Scientific Data.
- **Benchmark paper:** Strodthoff et al. (2020). *Deep Learning for ECG Analysis: Benchmarks and Insights from PTB-XL.* IEEE JBHI.
- **InceptionTime:** Ismail Fawaz et al. (2020). *InceptionTime: Finding AlexNet for Time Series Classification.* Data Mining and Knowledge Discovery.
- **Grad-CAM:** Selvaraju et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.* ICCV.
- **PTB-XL code repo:** https://github.com/helme/ecg_ptbxl_benchmarking (official benchmark code)
