import type { Metadata } from "next";
import { HeartDiagram } from "@/components/HeartDiagram";

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

/** Inline callout box — blue for definitions, amber for "why it matters" insights */
function InfoBox({
  accent = "blue",
  label,
  children,
}: {
  accent?: "blue" | "amber";
  label: string;
  children: React.ReactNode;
}) {
  const border =
    accent === "amber"
      ? "border-amber-900/50 bg-amber-950/15"
      : "border-blue-900/50 bg-blue-950/15";
  const labelClr = accent === "amber" ? "text-amber-400" : "text-blue-400";
  return (
    <div className={`not-prose rounded-xl border p-4 my-5 ${border}`}>
      <p className={`text-xs font-mono uppercase tracking-widest mb-3 ${labelClr}`}>
        {label}
      </p>
      <div className="text-gray-300 text-sm leading-relaxed space-y-2">{children}</div>
    </div>
  );
}

/** Likelihood confidence spectrum visual */
function LikelihoodSpectrum() {
  const bands = [
    { pct: "100%", label: "Definitive", sub: "both cardiologists agree", cls: "border-green-800/70 bg-green-950/25 text-green-400", dim: false },
    { pct: "80%",  label: "Probable",   sub: "some uncertainty",          cls: "border-gray-700 bg-gray-900/40 text-gray-500",        dim: true },
    { pct: "50%",  label: "Possible",   sub: "significant uncertainty",   cls: "border-gray-700 bg-gray-900/40 text-gray-500",        dim: true },
  ];
  return (
    <div className="grid grid-cols-3 gap-2 mt-3">
      {bands.map((b) => (
        <div
          key={b.pct}
          className={`rounded-lg border px-3 py-2.5 ${b.cls} ${b.dim ? "opacity-40" : ""}`}
        >
          <div className="flex items-baseline gap-1.5">
            <span className="text-lg font-bold tabular-nums">{b.pct}</span>
            {!b.dim && <span className="text-xs text-green-500 font-mono">✓ kept</span>}
            {b.dim && <span className="text-xs text-gray-600 font-mono">✗ removed</span>}
          </div>
          <p className="text-xs font-medium mt-0.5">{b.label}</p>
          <p className="text-[10px] text-gray-500 mt-0.5">{b.sub}</p>
        </div>
      ))}
    </div>
  );
}

/** "Explore the data" CTA — links to the (future) EDA page */
function EDAExploreLink() {
  return (
    <a
      href="/eda"
      className="group not-prose flex items-center justify-between rounded-xl border border-dashed border-blue-800/50 bg-blue-950/10 hover:border-blue-600/70 hover:bg-blue-950/20 px-5 py-4 transition-all mt-6"
    >
      <div>
        <p className="text-blue-400 font-semibold text-sm group-hover:text-blue-300 transition-colors">
          Explore the data with me
        </p>
        <p className="text-gray-500 text-xs mt-0.5">
          Signal distributions · PCA · Per-class morphology · Lead correlations · Label co-occurrence
        </p>
      </div>
      <span className="text-blue-600 group-hover:text-blue-400 group-hover:translate-x-0.5 transition-all text-xl shrink-0 ml-4">
        →
      </span>
    </a>
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
            Electrical Engineering · Deep Learning · Clinical AI
          </p>
          <h1 className="text-4xl font-bold text-white leading-tight">
            Building a 12-Lead ECG Arrhythmia Classifier
          </h1>
          <p className="text-gray-400 text-lg">
            The heart generates a signal. ECG measures it. The diagnosis is
            in the waveform shape — not the frequency content. That single EE
            insight drove every architectural decision.{" "}
            <span className="text-blue-400 font-medium">
              0.928 macro AUC-ROC on PTB-XL.
            </span>
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
          <div className="not-prose grid grid-cols-3 gap-4 my-6">
            <Metric value="18M"  label="CVD deaths per year"  sub="#1 cause of death globally (WHO)" />
            <Metric value="300M" label="ECGs recorded annually" sub="fastest, cheapest cardiac test" />
            <Metric value="~10 min" label="specialist read time" sub="per ECG — if one is available" />
          </div>

          <InfoBox label="This is an electrical engineering problem">
            <p>
              The heart generates electrical impulses at the{" "}
              <strong className="text-white">SA node</strong>; a 12-lead ECG
              captures the resulting voltage field from{" "}
              <strong className="text-white">
                10 electrode positions × 12 leads × 1000 time steps
              </strong>
              . The diagnostic signal is{" "}
              <strong className="text-white">morphological</strong> — it lives
              in the <em>shape</em> of the PQRST complex, not its frequency
              content. All five classes share the same 0–40 Hz bands; what
              differs is waveform geometry.
            </p>
            <HeartDiagram />
          </InfoBox>

          <p>
            The problem is not the sensor — it&apos;s the{" "}
            <strong className="text-white">interpretation bottleneck</strong>.
            There are fewer than 1 cardiologist per 10,000 people in most of
            the world. A single specialist reviews 50–100 ECGs per day; at that
            pace, backlogs accumulate and{" "}
            <strong className="text-white">
              time-critical conditions go undetected
            </strong>{" "}
            — myocardial infarction being the clearest example, where outcome
            degrades measurably with every hour of delay.
          </p>
          <div className="not-prose grid grid-cols-2 gap-3 my-5">
            <div className="rounded-lg border border-red-900/50 bg-red-950/15 p-4">
              <p className="text-red-400 font-mono text-xs uppercase tracking-widest mb-1.5">
                Abnormal
              </p>
              <p className="text-white font-semibold text-sm">
                Flag for immediate review
              </p>
              <p className="text-gray-500 text-xs mt-1">
                High-confidence detections → cardiologist queue, no delay
              </p>
            </div>
            <div className="rounded-lg border border-green-900/50 bg-green-950/15 p-4">
              <p className="text-green-400 font-mono text-xs uppercase tracking-widest mb-1.5">
                Normal
              </p>
              <p className="text-white font-semibold text-sm">
                Lighter triage path
              </p>
              <p className="text-gray-500 text-xs mt-1">
                Routine records flow through a fast-lane — backlog cleared
              </p>
            </div>
          </div>
          <p>
            The model doesn&apos;t replace the cardiologist.{" "}
            <strong className="text-blue-400">It scales their attention.</strong>
          </p>
        </Section>

        <Section id="data" tag="02 · The Data" title="Exploring the PTB-XL Dataset">
          <p>
            PTB-XL (Wagner et al., 2020) is the largest openly available
            clinical 12-lead ECG dataset: <strong className="text-white">21,799 recordings</strong>{" "}
            from 18,869 patients, each 10 seconds at 500 Hz — downsampled to
            100 Hz for this project. Each record was annotated by{" "}
            <strong className="text-white">up to two cardiologists</strong>{" "}
            using SCP-ECG codes, the European standard for machine-readable
            ECG diagnoses.
          </p>

          <InfoBox label="What is SCP-ECG?">
            <p>
              <strong className="text-white">SCP-ECG</strong> is the ISO
              standard for machine-readable ECG diagnoses — think of it as a
              structured code system for cardiologist findings. PTB-XL contains{" "}
              <strong className="text-white">71 distinct codes</strong>. For
              this project I collapsed them into{" "}
              <strong className="text-white">5 superclasses</strong> (NORM, MI,
              STTC, CD, HYP) — the same grouping used in the published Strodthoff
              et al. benchmark.
            </p>
          </InfoBox>

          <div className="not-prose grid grid-cols-2 md:grid-cols-4 gap-4 my-6">
            <Metric value="21,799" label="Total records"    sub="10 s · 12 leads · 100 Hz" />
            <Metric value="71"     label="SCP-ECG codes"   sub="→ 5 superclasses" />
            <Metric value="15,130" label="After filtering"  sub="100% likelihood only" />
            <Metric value="2"      label="Cardiologists"    sub="per annotation" />
          </div>

          <InfoBox accent="amber" label="The 100% likelihood filter — and why it matters">
            <p>
              Every annotation in PTB-XL carries a{" "}
              <strong className="text-white">likelihood score</strong> —
              the annotating cardiologist&apos;s confidence in the diagnosis:
            </p>
            <LikelihoodSpectrum />
            <p className="mt-3">
              Training on ambiguous examples (80% or 50% likelihood) teaches
              the model to reproduce{" "}
              <strong className="text-white">cardiologist uncertainty</strong>,
              not diagnostic ground truth. By keeping only 100%-likelihood
              annotations, we remove 6,669 records but are left with{" "}
              <strong className="text-white">
                15,130 records where the diagnosis was definitive
              </strong>{" "}
              — the cleanest possible training signal for the model.
            </p>
          </InfoBox>

          <InfoBox accent="amber" label="The class imbalance challenge">
            <p>
              Real clinical data mirrors real life: most people who get an ECG
              are — thankfully — healthy.{" "}
              <strong className="text-white">NORM accounts for nearly half</strong>{" "}
              the dataset simply because disease prevalence is low. This is not
              a dataset flaw; it&apos;s an accurate reflection of the world.
            </p>
            <p>
              The challenge: the model must be{" "}
              <strong className="text-white">equally correct on all five classes</strong>.
              An algorithm that misses 11% of MIs to pad its overall accuracy
              figure is not a medical tool — it&apos;s a liability. This is why
              per-class AUC-ROC, not overall accuracy, is the right metric,
              and why class-weighted loss is essential during training.
            </p>
          </InfoBox>

          <Table
            headers={["Class", "Description", "Test records", "% of test"]}
            rows={[
              ["NORM", "Normal sinus rhythm",    "721", "47.9%"],
              ["CD",   "Conduction disturbance", "456", "30.3%"],
              ["STTC", "ST/T-wave change",       "271", "18.0%"],
              ["MI",   "Myocardial infarction",  "171", "11.4%"],
              ["HYP",  "Hypertrophy",            "132",  "8.8%"],
            ]}
          />

          <EDAExploreLink />
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
