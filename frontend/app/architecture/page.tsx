import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Architecture Deep-Dive — Revant Bisht",
  description:
    "InceptionTime + Temporal Attention for 12-lead ECG arrhythmia classification — inception modules, multi-scale kernels, temporal attention, training strategy, and results.",
};

const GITHUB_NB =
  "https://github.com/Revant-Bisht/ArrythmiaClassifier/blob/main/notebooks/02_model.ipynb";

// ─── Shared primitives ───────────────────────────────────────────────────────

function CellHeader({ n, title }: { n: string; title: string }) {
  return (
    <div className="flex items-center gap-3 mb-4">
      <span className="text-[11px] font-mono text-gray-600 bg-gray-900 border border-gray-800 px-2 py-0.5 rounded shrink-0">
        In [{n}]
      </span>
      <h2 className="text-lg font-bold text-white">{title}</h2>
    </div>
  );
}

function Out({ children }: { children: React.ReactNode }) {
  return (
    <div className="ml-12 mb-6">
      <span className="text-[11px] font-mono text-gray-700 block mb-1">Out:</span>
      {children}
    </div>
  );
}

function TextOut({ children }: { children: string }) {
  return (
    <pre className="text-xs font-mono text-gray-400 bg-gray-950 border border-gray-800 rounded-lg p-4 overflow-x-auto leading-relaxed whitespace-pre">
      {children}
    </pre>
  );
}

function NbImg({
  src,
  alt,
  caption,
  wide,
}: {
  src: string;
  alt: string;
  caption?: string;
  wide?: boolean;
}) {
  return (
    <figure className={wide ? "-mx-6 sm:mx-0" : ""}>
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        src={src}
        alt={alt}
        className="w-full h-auto rounded-lg border border-gray-800"
        loading="lazy"
      />
      {caption && (
        <figcaption className="text-gray-600 text-xs font-mono mt-2 text-center">
          {caption}
        </figcaption>
      )}
    </figure>
  );
}

function Insight({ children }: { children: React.ReactNode }) {
  return (
    <div className="rounded-xl border border-amber-900/40 bg-amber-950/15 px-4 py-3 mt-5">
      <p className="text-[10px] font-mono uppercase tracking-widest text-amber-500 mb-1.5">
        Key insight
      </p>
      <p className="text-gray-300 text-sm leading-relaxed">{children}</p>
    </div>
  );
}

function Decision({ title, reason }: { title: string; reason: string }) {
  return (
    <div className="rounded-lg border border-gray-800 bg-gray-900/60 p-4 space-y-1.5">
      <p className="text-white font-semibold text-sm">{title}</p>
      <p className="text-gray-500 text-xs leading-relaxed">{reason}</p>
    </div>
  );
}

function Table({
  headers,
  rows,
}: {
  headers: string[];
  rows: (string | number)[][];
}) {
  return (
    <div className="overflow-x-auto mt-4">
      <table className="w-full text-sm border-collapse">
        <thead>
          <tr className="border-b border-gray-700">
            {headers.map((h) => (
              <th
                key={h}
                className="text-left py-2 px-3 text-gray-400 font-mono text-xs uppercase tracking-wider"
              >
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i} className="border-b border-gray-800">
              {row.map((cell, j) => (
                <td key={j} className="py-2 px-3 text-gray-300 text-xs">
                  {cell}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Code({ children }: { children: string }) {
  return (
    <pre className="rounded-lg bg-gray-950 border border-gray-800 p-4 overflow-x-auto text-xs font-mono text-gray-300 leading-relaxed mt-4">
      {children}
    </pre>
  );
}

// ─── Page ────────────────────────────────────────────────────────────────────

export default function ArchitecturePage() {
  return (
    <main className="bg-navy-900 min-h-screen">
      {/* Nav */}
      <nav className="fixed top-0 left-0 right-0 z-50 flex items-center justify-between px-6 py-4 bg-navy-900/80 backdrop-blur-md border-b border-gray-800">
        <a
          href="/blog"
          className="flex items-center gap-2 text-sm text-gray-400 hover:text-white transition-colors"
        >
          <span className="text-gray-600">←</span> Technical Writeup
        </a>
        <span className="text-gray-500 text-sm font-mono tracking-widest">
          02_model.ipynb
        </span>
        <a
          href={GITHUB_NB}
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-gray-700 hover:border-gray-500 text-gray-400 hover:text-white text-xs font-mono transition-colors"
        >
          Open on GitHub →
        </a>
      </nav>

      <div className="max-w-3xl mx-auto px-6 pt-28 pb-24 space-y-16">

        {/* ── Header ── */}
        <div className="space-y-4">
          <p className="text-blue-400 text-xs font-mono tracking-widest uppercase">
            Architecture · Training · Results
          </p>
          <h1 className="text-4xl font-bold text-white leading-tight">
            InceptionTime + Temporal Attention
          </h1>
          <p className="text-gray-400 text-base leading-relaxed">
            Multi-scale convolutional feature extraction paired with a soft
            attention mechanism — every design choice here is a direct response
            to a finding from the EDA.
          </p>
          <div className="flex flex-wrap gap-2 text-xs font-mono">
            {[
              "~244k parameters",
              "3 InceptionBlocks",
              "k = 10, 20, 40",
              "Temporal Attention",
              "0.928 macro AUC-ROC",
            ].map((t) => (
              <span
                key={t}
                className="px-2.5 py-1 rounded-md bg-gray-800 text-gray-400"
              >
                {t}
              </span>
            ))}
          </div>
          <div className="flex flex-wrap gap-3 pt-1">
            <a
              href={GITHUB_NB}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1.5 px-4 py-2 rounded-lg border border-gray-700 hover:border-gray-500 text-gray-300 hover:text-white text-sm transition-colors"
            >
              View notebook on GitHub →
            </a>
          </div>
        </div>

        {/* ── 1  Problem Formulation ── */}
        <div>
          <CellHeader n="1" title="Problem Formulation" />

          <div className="space-y-4 text-sm text-gray-300 leading-relaxed mb-5">
            <p>
              Each PTB-XL record is a <strong className="text-white">12-lead ECG at 100 Hz for 10 seconds</strong> —
              a tensor of shape <code className="text-blue-300 bg-gray-900 px-1 rounded text-xs">12 × 1000</code>. Each
              lead is z-score normalised independently: this removes inter-patient amplitude variation while preserving
              the morphological shape of each lead.
            </p>
            <p>
              The output is a <strong className="text-white">multi-label binary vector over 5 classes</strong> (NORM,
              MI, STTC, CD, HYP). 84% of records carry exactly one label; 13% have two; 3% have three.
            </p>
          </div>

          <Insight>
            Multi-label sigmoid, not softmax. STTC+HYP co-occur in 868 records (hypertrophy strains the myocardium
            causing ST changes). MI+CD co-occur in 550 records (infarction disrupts conduction). Softmax would force
            a single winner — wrong for any record with two active conditions.
          </Insight>

          <div className="mt-5 grid grid-cols-3 gap-3">
            {[
              { val: "12 × 1000", label: "Input tensor", sub: "leads × timesteps" },
              { val: "5", label: "Output classes", sub: "independent sigmoids" },
              { val: "100 Hz", label: "Sample rate", sub: "downsampled from 500 Hz" },
            ].map((m) => (
              <div key={m.label} className="rounded-xl border border-gray-800 bg-navy-950 p-4 text-center">
                <p className="text-2xl font-bold text-blue-400 tabular-nums font-mono">{m.val}</p>
                <p className="text-white font-medium mt-1 text-sm">{m.label}</p>
                <p className="text-gray-500 text-xs mt-0.5">{m.sub}</p>
              </div>
            ))}
          </div>
        </div>

        {/* ── 2  Why InceptionTime ── */}
        <div>
          <CellHeader n="2" title="Why InceptionTime" />

          <p className="text-sm text-gray-300 leading-relaxed mb-4">
            The EDA showed ECG morphology operates at multiple clinical timescales simultaneously.
            At 100 Hz, the diagnostic features each occupy a different window:
          </p>

          <Out>
            <Table
              headers={["Feature", "Duration", "Samples @ 100 Hz", "→ Kernel"]}
              rows={[
                ["QRS complex", "60–120 ms", "6–12", "k = 10"],
                ["ST segment",  "80–120 ms", "8–12",  "k = 10"],
                ["T-wave",      "150–250 ms", "15–25", "k = 20"],
                ["P-wave",      "80–120 ms",  "8–12",  "k = 10"],
                ["RR interval", "600–1000 ms", "60–100", "k = 40"],
              ]}
            />
          </Out>

          <Insight>
            A single fixed-kernel CNN sees only one timescale at a time. InceptionTime runs three parallel
            convolutions (k=10, k=20, k=40) simultaneously — one per clinically relevant window. Their
            outputs are concatenated so later layers see all three scales at once.
          </Insight>
        </div>

        {/* ── 3  Inception Module ── */}
        <div>
          <CellHeader n="3" title="The Inception Module" />

          <p className="text-sm text-gray-300 leading-relaxed mb-4">
            Each InceptionBlock takes an input <code className="text-blue-300 bg-gray-900 px-1 rounded text-xs">H ∈ (B, C_in, 1000)</code> and
            produces <code className="text-blue-300 bg-gray-900 px-1 rounded text-xs">H_out ∈ (B, 128, 1000)</code> via four parallel branches:
          </p>

          <Code>{`InceptionBlock(C_in → 128):
│
├─ Bottleneck  Conv1d(C_in → 32, k=1) + BN       reduce channels first
│
├─ Branch k=10  Conv1d(32 → 32, k=10, pad=same)   ← QRS / ST timescale
├─ Branch k=20  Conv1d(32 → 32, k=20, pad=same)   ← T-wave timescale
├─ Branch k=40  Conv1d(32 → 32, k=40, pad=same)   ← RR-interval timescale
│
├─ Branch MP    MaxPool1d(k=3) → Conv1d(C_in → 32, k=1)
│
└─ Concat [B_10 ‖ B_20 ‖ B_40 ‖ B_mp] → BN → ReLU → (B, 128, 1000)
   + Residual shortcut (first block only): Conv1d(C_in → 128, k=1) + BN`}</Code>

          <Out>
            <NbImg
              src="/arch/05_inception_block_diagram.png"
              alt="InceptionBlock architecture diagram"
              caption="InceptionBlock — parallel multi-scale branches with residual shortcut"
            />
          </Out>

          <Insight>
            The bottleneck (k=1 convolution) cuts channels from C_in to 32 before the parallel
            branches — keeping compute manageable. The MaxPool branch acts as a skip that captures
            dominant local amplitude peaks without any learnable parameters.
          </Insight>
        </div>

        {/* ── 4  Temporal Attention ── */}
        <div>
          <CellHeader n="4" title="Temporal Soft Attention" />

          <p className="text-sm text-gray-300 leading-relaxed mb-4">
            After 3 InceptionBlocks, the feature map is <code className="text-blue-300 bg-gray-900 px-1 rounded text-xs">(B, 128, 1000)</code>.
            Global average pooling over time would discard which timestep the features came from.
            Temporal soft attention learns per-timestep importance weights α instead:
          </p>

          <Code>{`TemporalAttention(128 → 128):
  e_t   = v^T · tanh(W · h_t + b)       score each timestep
  α_t   = softmax(e_t)                   normalise → Σ α_t = 1
  z     = Σ_t α_t · h_t  ∈ (B, 128)    weighted sum → context vector

  → Dropout(0.3) → Linear(128 → 5) → Sigmoid → (B, 5)`}</Code>

          <div className="space-y-3 mt-5">
            <p className="text-sm text-gray-300 leading-relaxed">
              The weights <span className="text-purple-300 font-mono">α</span> serve double duty:
            </p>
            <div className="grid grid-cols-2 gap-3">
              <div className="rounded-lg border border-gray-800 bg-gray-900/60 p-4 space-y-1">
                <p className="text-white font-semibold text-sm">Accuracy</p>
                <p className="text-gray-500 text-xs leading-relaxed">
                  The model focuses compute on diagnostically relevant beats instead of distributing
                  attention uniformly over 10 seconds of signal.
                </p>
              </div>
              <div className="rounded-lg border border-gray-800 bg-gray-900/60 p-4 space-y-1">
                <p className="text-white font-semibold text-sm">Explainability</p>
                <p className="text-gray-500 text-xs leading-relaxed">
                  α is directly visualisable — the purple overlay in the demo is these weights plotted
                  over the ECG trace. No post-hoc computation needed.
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* ── 5  Full Forward Pass + Parameter Count ── */}
        <div>
          <CellHeader n="5" title="Full Forward Pass & Parameter Count" />

          <Code>{`(B, 12, 1000)
    → InceptionBlock(12  → 128)   residual shortcut ✓
    → InceptionBlock(128 → 128)   identity shortcut
    → InceptionBlock(128 → 128)   identity shortcut
    → TemporalAttention            → (B, 128)
    → Dropout(0.3)
    → Linear(128 → 5) + Sigmoid   → (B, 5)`}</Code>

          <Out>
            <TextOut>{`Total parameters:     243,909
Trainable parameters: 243,909

  inception_blocks                  234,944
  attention                           8,320
  dropout                                 0
  classifier                            645


Input shape:   (4, 12, 1000)
Logits shape:  (4, 5)   (batch × 5 classes, pre-sigmoid)
Attention shape: (4, 1000)  (batch × 1000 timesteps)`}</TextOut>
          </Out>

          <Insight>
            244k parameters — intentionally lightweight. The xresnet1d101 baseline (the strongest
            SOTA reference) uses ~740k parameters. At 0.928 macro AUC-ROC, this model approaches
            it (0.931) at one-third the parameter count.
          </Insight>
        </div>

        {/* ── 6  Training Strategy ── */}
        <div>
          <CellHeader n="6" title="Training Strategy" />

          <p className="text-sm text-gray-300 leading-relaxed mb-4">
            PTB-XL provides an official 10-fold stratified split. We follow it exactly so results
            are directly comparable to the Strodthoff et al. 2020 benchmark — test fold 10 is
            never touched during development.
          </p>

          <Out>
            <Table
              headers={["Split", "Folds", "Records", "Purpose"]}
              rows={[
                ["Train", "1–8", "12,133", "Gradient updates"],
                ["Val",   "9",   "1,491",  "Early stopping · threshold tuning"],
                ["Test",  "10",  "1,506",  "Final reported metrics (touch once)"],
              ]}
            />
          </Out>

          <p className="text-sm text-gray-300 leading-relaxed mt-5 mb-4">
            Three augmentations are applied on-the-fly during training only, each EDA-validated:
          </p>

          <Out>
            <Table
              headers={["Augmentation", "Parameter", "EDA Justification"]}
              rows={[
                ["Gaussian noise", "σ = 0.01", "Median lead std = 0.165 mV → σ=0.01 ≈ 6% signal energy"],
                ["Lead dropout",   "p = 0.10 per lead", "15.1% of records show static artefact in ≥1 lead"],
                ["Time shift",     "±50 samples", "Varies beat phase without truncating QRS morphology"],
              ]}
            />
          </Out>

          <Insight>
            No frequency-domain augmentation. PSD analysis showed class information is morphological,
            not spectral — corrupting frequency content could destroy discriminative features.
          </Insight>

          <p className="text-sm text-gray-300 leading-relaxed mt-5 mb-2">
            <strong className="text-white">Learning rate schedule:</strong> cosine annealing from
            1×10⁻³ to 1×10⁻⁶ over 100 epochs with early stopping (patience = 15) on val macro AUC-ROC.
          </p>

          <Out>
            <NbImg
              src="/arch/06_lr_schedule.png"
              alt="Cosine annealing learning rate schedule"
              caption="Cosine annealing LR — fast early convergence, fine-grained updates near end of training"
            />
          </Out>
        </div>

        {/* ── 7  Training Curves ── */}
        <div>
          <CellHeader n="7" title="Training Curves" />

          <Out>
            <TextOut>{`Loaded epoch 35  |  val macro AUC: 0.9322`}</TextOut>
          </Out>

          <Out>
            <NbImg
              src="/arch/12_training_history.png"
              alt="Training loss and AUC curves"
              caption="Loss and macro AUC-ROC across 35 epochs — train/val gap closed from 0.065 to 0.020"
              wide
            />
          </Out>

          <Insight>
            Three regularisation mechanisms closed the train/val AUC gap from 0.065 to 0.020:
            classifier dropout (0.3), weight decay (5×10⁻⁴), and lead dropout (0.2). Training
            converged at epoch 35 on Apple MPS (~12 s/epoch).
          </Insight>
        </div>

        {/* ── 8  Results ── */}
        <div>
          <CellHeader n="8" title="Test Set Results" />

          <Out>
            <TextOut>{`Macro AUC-ROC (Val fold 9): 0.9322
  NORM: 0.9745
  MI:   0.9341
  STTC: 0.9451
  CD:   0.9326
  HYP:  0.8749`}</TextOut>
          </Out>

          <Out>
            <NbImg
              src="/arch/11_roc_curves.png"
              alt="ROC curves for all 5 classes"
              caption="Per-class ROC curves on test fold 10 — macro AUC-ROC = 0.928"
              wide
            />
          </Out>

          <div className="grid grid-cols-3 gap-3 mt-5">
            {[
              { val: "0.928", label: "Macro AUC-ROC", sub: "test fold 10" },
              { val: "0.763", label: "Macro AUPRC",   sub: "5-class average" },
              { val: "0.664", label: "Macro F1",      sub: "Youden threshold" },
            ].map((m) => (
              <div key={m.label} className="rounded-xl border border-gray-800 bg-navy-950 p-4 text-center">
                <p className="text-2xl font-bold text-blue-400 tabular-nums font-mono">{m.val}</p>
                <p className="text-white font-medium mt-1 text-sm">{m.label}</p>
                <p className="text-gray-500 text-xs mt-0.5">{m.sub}</p>
              </div>
            ))}
          </div>

          <Table
            headers={["Model", "Macro AUC-ROC", "Notes"]}
            rows={[
              ["Simple 1D CNN",                "0.890", "Strodthoff et al. 2020"],
              ["LSTM + Bidir Attention",        "0.907", "Strodthoff et al. 2020"],
              ["InceptionTime (ref)",           "0.925", "Strodthoff et al. 2020"],
              ["xresnet1d101",                  "0.931", "Strodthoff et al. 2020 — 3× params"],
              ["This work (InceptionTime+Attn)","0.928", "~244k params ✓"],
            ]}
          />

          <Insight>
            0.928 macro AUC-ROC exceeds the published InceptionTime reference (0.925) and approaches
            xresnet1d101 (0.931) — the strongest SOTA baseline — which uses roughly 3× the parameters.
            HYP is the weakest class (AUC 0.875) as expected: it has the fewest training examples and
            the most subtle morphological signature.
          </Insight>
        </div>

        {/* ── Design Decisions ── */}
        <div>
          <h2 className="text-lg font-bold text-white mb-4">Architecture Design Decisions</h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            <Decision
              title="Kernels k=10, 20, 40 — not arbitrary"
              reason="Chosen to match QRS (~10 samples), T-wave (~20 samples), and RR-interval (~80 samples) at 100 Hz. Verified by timescale analysis in the EDA before writing any model code."
            />
            <Decision
              title="Multi-label sigmoid, not softmax"
              reason="84% single-label, 13% two-label, 3% three-label. STTC+HYP co-occur in 868 records. Softmax would force exactly one winner and misclassify every multi-label record."
            />
            <Decision
              title="Residual shortcut on block 1 only"
              reason="Input has 12 channels; subsequent blocks are 128→128. The shortcut is structurally necessary on block 1 (dimension mismatch). Blocks 2 and 3 use identity shortcuts."
            />
            <Decision
              title="Temporal attention over global average pooling"
              reason="GAP treats all timesteps equally. Attention learns which beats are diagnostically relevant — and exposes those weights directly as the α visualisation in the demo."
            />
            <Decision
              title="No frequency-domain features or augmentation"
              reason="PSD showed all 5 classes share the same 0–40 Hz bands. The discriminative signal is morphological. Frequency-domain approaches would discard the signal, not extract it."
            />
            <Decision
              title="All 12 leads, not just Lead II"
              reason="Lead correlation analysis showed precordial and limb leads carry territory-specific information that is not redundant. Dropping leads would lose anterior/lateral/inferior wall info."
            />
          </div>
        </div>

      </div>

      {/* Footer CTAs */}
      <section className="bg-navy-950 border-t border-gray-800 py-16 px-6 text-center">
        <div className="max-w-2xl mx-auto space-y-5">
          <h2 className="text-xl font-bold text-white">See it in action</h2>
          <p className="text-gray-400 text-sm">
            The interactive demo shows Grad-CAM heatmaps and temporal attention α weights on real
            PTB-XL ECGs — all five classes, live.
          </p>
          <div className="flex flex-wrap items-center justify-center gap-3">
            <a
              href="/demo"
              className="inline-flex items-center gap-2 px-6 py-3 rounded-lg bg-blue-600 hover:bg-blue-500 text-white font-medium transition-colors"
            >
              Try the Interactive Demo →
            </a>
            <a
              href="/blog"
              className="inline-flex items-center gap-2 px-6 py-3 rounded-lg border border-gray-700 hover:border-gray-500 text-gray-300 hover:text-white font-medium transition-colors"
            >
              ← Back to Writeup
            </a>
          </div>
        </div>
      </section>

      <footer className="bg-navy-950 border-t border-gray-800 py-8 px-6">
        <div className="max-w-5xl mx-auto flex flex-wrap items-center justify-between gap-4 text-sm text-gray-600">
          <span>© 2026 Revant Bisht</span>
          <span>PTB-XL dataset · Wagner et al. 2020 · Strodthoff et al. 2020</span>
          <span className="text-xs">For research purposes only — not a clinical tool</span>
        </div>
      </footer>
    </main>
  );
}
