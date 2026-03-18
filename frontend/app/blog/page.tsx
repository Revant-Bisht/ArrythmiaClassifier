import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "How I Built an ECG Arrhythmia Classifier — Revant Bisht",
  description:
    "Technical writeup: dataset, EDA, InceptionTime + Attention architecture, training, results (0.928 AUC), Grad-CAM explainability, and ONNX deployment.",
};

function Section({
  id,
  tag,
  title,
  children,
}: {
  id: string;
  tag: string;
  title: string;
  children: React.ReactNode;
}) {
  return (
    <section id={id} className="py-16 border-b border-gray-800">
      <p className="text-blue-400 text-xs font-mono tracking-widest uppercase mb-2">
        {tag}
      </p>
      <h2 className="text-2xl font-bold text-white mb-8">{title}</h2>
      <div className="prose prose-invert prose-sm max-w-none text-gray-300 leading-relaxed space-y-4">
        {children}
      </div>
    </section>
  );
}

function Metric({
  value,
  label,
  sub,
}: {
  value: string;
  label: string;
  sub?: string;
}) {
  return (
    <div className="rounded-xl border border-gray-800 bg-navy-950 p-5 text-center">
      <p className="text-3xl font-bold text-blue-400 tabular-nums">{value}</p>
      <p className="text-white font-medium mt-1">{label}</p>
      {sub && <p className="text-gray-500 text-xs mt-0.5">{sub}</p>}
    </div>
  );
}

function Code({ children }: { children: string }) {
  return (
    <pre className="rounded-lg bg-navy-950 border border-gray-800 p-4 overflow-x-auto text-xs font-mono text-gray-300 leading-relaxed">
      {children}
    </pre>
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
    <div className="overflow-x-auto">
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
                <td key={j} className="py-2 px-3 text-gray-300">
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

export default function BlogPage() {
  return (
    <main className="bg-navy-900 min-h-screen">
      <nav className="fixed top-0 left-0 right-0 z-50 flex items-center justify-between px-6 py-4 bg-navy-900/80 backdrop-blur-md border-b border-gray-800">
        <a href="/" className="flex items-center gap-3 text-sm">
          <span className="text-gray-500">←</span>
          <span className="text-gray-400 hover:text-white transition-colors">
            Back to Demo
          </span>
        </a>
        <span className="text-gray-500 text-sm">Technical Writeup</span>
      </nav>

      <div className="max-w-3xl mx-auto px-6 pt-28 pb-24">
        <div className="mb-16 space-y-4">
          <p className="text-blue-400 text-sm font-mono tracking-widest uppercase">
            Deep Learning · Cardiology · MLOps
          </p>
          <h1 className="text-4xl font-bold text-white leading-tight">
            Building a 12-Lead ECG Arrhythmia Classifier
          </h1>
          <p className="text-gray-400 text-lg">
            From raw PTB-XL waveforms to a live explainable demo — architecture
            decisions backed by data, 0.928 macro AUC-ROC, and Grad-CAM
            heatmaps that highlight clinically known diagnostic regions.
          </p>
          <div className="flex flex-wrap gap-2 pt-2">
            {[
              "PyTorch",
              "InceptionTime",
              "Temporal Attention",
              "Grad-CAM",
              "ONNX Runtime",
              "FastAPI",
              "PTB-XL",
            ].map((tag) => (
              <span
                key={tag}
                className="px-2.5 py-1 rounded-md bg-gray-800 text-gray-400 text-xs font-mono"
              >
                {tag}
              </span>
            ))}
          </div>
        </div>

        <Section id="problem" tag="01 · The Problem" title="Why Automated ECG Analysis Matters">
          <p>
            Cardiovascular disease is the leading cause of death globally.
            A 12-lead ECG is the first-line diagnostic tool — inexpensive,
            non-invasive, and available in every clinic. But interpreting
            one correctly requires years of specialist training, and demand
            far outstrips supply in many healthcare systems.
          </p>
          <p>
            The PTB-XL benchmark (Strodthoff et al., 2020) established a
            rigorous evaluation framework: given 10 seconds of 12-lead ECG
            data, classify each record into five clinically meaningful
            superdiagnostic categories simultaneously. The multi-label setup
            reflects real clinical reality — a patient can present with both
            conduction disturbance and hypertrophy.
          </p>
        </Section>

        <Section id="data" tag="02 · The Data" title="PTB-XL Dataset">
          <p>
            PTB-XL (Wagner et al., 2020) contains 21,799 clinical ECG records
            from 18,869 patients, each 10 seconds long at 500 Hz
            (downsampled to 100 Hz for this project). Records are annotated by
            up to two cardiologists with SCP-ECG codes, which I mapped to five
            superdiagnostic classes.
          </p>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 my-6">
            <Metric value="15,130" label="Labeled records" sub="min_likelihood=100%" />
            <Metric value="12,133" label="Training" sub="folds 1–8" />
            <Metric value="1,491" label="Validation" sub="fold 9" />
            <Metric value="1,506" label="Test" sub="fold 10" />
          </div>
          <p>
            Applying a 100% likelihood filter removes ambiguous annotations,
            leaving 15,130 records. Class distribution is significantly
            imbalanced: NORM (47.4%) dominates while HYP (9.7%) is rare.
            84% of records carry a single label; 13% have two; 3% have three —
            confirming that independent sigmoid outputs are the correct
            formulation.
          </p>
          <Table
            headers={["Class", "Description", "Count", "% of test"]}
            rows={[
              ["NORM", "Normal sinus rhythm", "721", "47.9%"],
              ["CD", "Conduction disturbance", "456", "30.3%"],
              ["STTC", "ST/T-wave change", "271", "18.0%"],
              ["MI", "Myocardial infarction", "171", "11.4%"],
              ["HYP", "Hypertrophy", "132", "8.8%"],
            ]}
          />
        </Section>

        <Section id="eda" tag="03 · EDA" title="What the Data Told Me">
          <p>
            Before writing a single model line, I ran a 10-section exploratory
            analysis to let the data drive architecture decisions.
          </p>
          <p>
            <strong className="text-white">Power spectral density:</strong>{" "}
            Classes are not separable by frequency content. All five classes
            share the same dominant frequency bands (0–40 Hz). This ruled out
            frequency-domain features and confirmed that the discriminative
            signal is morphological — the shape of the PQRST complex matters,
            not which frequencies are present.
          </p>
          <p>
            <strong className="text-white">PCA:</strong> Complete class overlap
            in the first 20 principal components. Linear separation is
            impossible, requiring a deep non-linear model.
          </p>
          <p>
            <strong className="text-white">Timescale analysis:</strong> At
            100 Hz, QRS complexes span ~10 samples, T-waves ~20 samples, and
            RR intervals ~80 samples. This directly motivated the three
            parallel kernel sizes in InceptionTime: 10, 20, and 40.
          </p>
          <p>
            <strong className="text-white">Lead correlation:</strong> Limb
            leads and precordial leads form distinct correlated blocks, but all
            12 leads carry territory-specific information not redundant with
            others. All 12 leads should be processed together.
          </p>
        </Section>

        <Section id="architecture" tag="04 · Architecture" title="InceptionTime + Temporal Attention">
          <p>
            The architecture is a direct response to the EDA findings: parallel
            convolutional kernels sized to clinically relevant timescales,
            processing all 12 leads simultaneously, with a soft-attention
            mechanism to identify which temporal regions matter most.
          </p>
          <Code>{`Input: (B, 12, 1000)
│
├─ InceptionBlock × 3
│   Bottleneck:  Conv1D(C_in → 32, k=1) + BN
│   Branch 1:   Conv1D(32 → 32, k=10)  ← QRS timescale
│   Branch 2:   Conv1D(32 → 32, k=20)  ← T-wave timescale
│   Branch 3:   Conv1D(32 → 32, k=40)  ← RR-interval timescale
│   Branch 4:   MaxPool(k=3) → Conv1D(C_in → 32, k=1)
│   Concat → (B, 128, 1000) → BN → ReLU
│   Residual shortcut: Conv1D(C_in → 128, k=1) + BN
│
├─ Temporal Soft Attention
│   Linear(128 → 64) → Tanh → Linear(64 → 1) → Softmax
│   Context: weighted sum → (B, 128)
│
└─ Dropout(0.3) → Linear(128 → 5) → Sigmoid

Parameters: ~400k`}</Code>
          <p>
            The bottleneck projection reduces dimensionality before the parallel
            convolutions, keeping compute manageable. The residual shortcut on
            the first block prevents vanishing gradients. The temporal attention
            both improves accuracy and provides a natural saliency signal for
            the explainability visualisation.
          </p>
        </Section>

        <Section id="training" tag="05 · Training" title="Loss, Regularisation, and Augmentation">
          <p>
            Multi-label classification with imbalanced classes requires careful
            loss design. I used weighted binary cross-entropy with label
            smoothing (ε=0.05) to prevent overconfident predictions:
          </p>
          <Code>{`L = -Σ_c  w_c [ ỹ_c · log(p_c) + (1 - ỹ_c) · log(1 - p_c) ]

ỹ_c = y_c · (1 - ε) + ε/2          (label smoothing)

Class weights (inverse frequency, training set):
  NORM = 0.42   CD = 0.67   STTC = 1.12
  MI   = 1.48   HYP = 2.02`}</Code>
          <Table
            headers={["Hyperparameter", "Value", "Justification"]}
            rows={[
              ["Optimizer", "Adam", "Standard for time-series"],
              ["Learning rate", "1e-3 → 1e-6", "Cosine annealing"],
              ["Weight decay", "5e-4", "L2 regularisation"],
              ["Batch size", "64", "Fits comfortably in MPS memory"],
              ["Early stopping", "patience=15", "Val macro AUC-ROC"],
              ["Gaussian noise", "σ=0.01", "1% of signal energy — EDA-validated"],
              ["Lead dropout", "p=0.20", "Matches ~15% electrode artifact rate"],
              ["Time shift", "±50 samples", "Preserves QRS, varies beat phase"],
            ]}
          />
          <p>
            Three regularisation mechanisms worked together to close the
            train/val AUC gap from 0.065 to 0.020: classifier dropout (0.3),
            increased weight decay (5e-4), and lead dropout (0.2). Training
            converged at epoch 35 on Apple MPS (~12s/epoch).
          </p>
        </Section>

        <Section id="results" tag="06 · Results" title="Test Set Performance">
          <div className="grid grid-cols-3 gap-4 my-6">
            <Metric value="0.928" label="Macro AUC-ROC" sub="PTB-XL test fold 10" />
            <Metric value="0.763" label="Macro AUPRC" sub="5-class average" />
            <Metric value="0.664" label="Macro F1" sub="Youden threshold" />
          </div>
          <Table
            headers={["Model", "Macro AUC", "Notes"]}
            rows={[
              ["Simple 1D CNN", "0.890", "Strodthoff et al. 2020"],
              ["LSTM + Bidir Attention", "0.907", "Strodthoff et al. 2020"],
              ["InceptionTime (ref)", "0.925", "Strodthoff et al. 2020"],
              ["xresnet1d101", "0.931", "Strodthoff et al. 2020 — 3× params"],
              ["This work (InceptionTime+Attn)", "0.928", "~400k params ✓"],
            ]}
          />
          <p>
            At 0.928 macro AUC-ROC, this model exceeds the published
            InceptionTime reference (0.925) despite having a similar parameter
            count, and approaches xresnet1d101 (0.931) which has roughly 3×
            the parameters. HYP is the weakest class (AUC 0.860) — expected
            given it has only 132 test samples.
          </p>
          <Table
            headers={["Class", "AUC-ROC", "AUPRC", "F1 (Youden)"]}
            rows={[
              ["NORM", "0.971", "0.963", "0.911"],
              ["MI", "0.941", "0.725", "0.560"],
              ["STTC", "0.947", "0.817", "0.727"],
              ["CD", "0.921", "0.866", "0.761"],
              ["HYP", "0.860", "0.444", "0.364"],
            ]}
          />
        </Section>

        <Section id="explainability" tag="07 · Explainability" title="Grad-CAM Heatmaps">
          <p>
            Grad-CAM (Selvaraju et al., 2017) is adapted for 1D convolutions
            by hooking into the last InceptionBlock, computing the gradient of
            the target class logit with respect to the feature map, and
            global-average-pooling the gradients over time to get per-channel
            importance weights.
          </p>
          <Code>{`A = activations[last_block]   # (C, T)
G = gradients[last_block]     # (C, T)
w = G.mean(dim=-1)            # (C,) — channel importance
cam = ReLU(w · A)             # (T,) — temporal saliency
cam = cam / cam.max()         # normalise to [0, 1]`}</Code>
          <p>
            The heatmaps are clinically plausible without any supervision on
            waveform segments:
          </p>
          <Table
            headers={["Class", "Expected region", "Grad-CAM finding"]}
            rows={[
              ["MI", "Q-wave, ST elevation", "✓ Peaks on early QRS + ST segment"],
              ["STTC", "ST segment, T-wave", "✓ Activation on ST-T region"],
              ["CD", "Entire QRS complex", "✓ Broad activation over widened QRS"],
              ["HYP", "Tall R-wave peak", "✓ Sharp peak at R-wave maximum"],
              ["NORM", "No focal region", "✓ Diffuse low-amplitude activation"],
            ]}
          />
          <p>
            The interactive demo shows all five Grad-CAM maps simultaneously
            — selecting a class tab switches the overlay without re-fetching,
            because all five maps are pre-computed and cached server-side.
          </p>
        </Section>

        <Section id="deployment" tag="08 · Deployment" title="ONNX, FastAPI, and Fly.io">
          <p>
            The production server runs ONNX Runtime — no PyTorch dependency in
            the Docker image. PyTorch is used only locally to generate the
            pre-cached sample responses at build time.
          </p>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 my-6">
            <Metric value="1.01 MB" label="ONNX model" sub="opset 17" />
            <Metric value="2.6×" label="ONNX speedup" sub="vs PyTorch CPU" />
            <Metric value="7 ms" label="API latency" sub="preloaded sample" />
            <Metric value="0 ms" label="Cold start" sub="Fly.io always-on" />
          </div>
          <p>
            The five curated samples (highest-confidence correct prediction per
            class) are pre-computed with full Grad-CAM at server startup and
            held in memory. A recruiter clicking "Run" receives a complete
            response — signal, five Grad-CAM maps, attention weights, clinical
            report, flagged regions — in under 10ms.
          </p>
          <Code>{`GET /predict/preloaded/MI
→ {
    predicted_class: "MI",
    confidence: 0.997,
    probs: { NORM: 0.004, MI: 0.997, ... },
    gradcam_per_class: { NORM: [1000 floats], MI: [...], ... },
    attention: [1000 floats],
    signal_lead2: [1000 floats],
    report: { headline, summary, flagged_regions, ... }
  }`}</Code>
        </Section>

        <div className="pt-10 text-center space-y-4">
          <a
            href="/"
            className="inline-flex items-center gap-2 px-6 py-3 rounded-lg bg-blue-600 hover:bg-blue-500 text-white font-medium transition-colors"
          >
            ← Try the Interactive Demo
          </a>
          <p className="text-gray-600 text-xs">
            All source code available on{" "}
            <a
              href="https://github.com/Revant-Bisht/ArrythmiaClassifier"
              className="underline hover:text-gray-400 transition-colors"
              target="_blank"
              rel="noopener noreferrer"
            >
              GitHub
            </a>
          </p>
        </div>
      </div>
    </main>
  );
}
