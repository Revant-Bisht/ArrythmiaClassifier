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

/** "Full evaluation results" CTA — links to the results deep-dive page */
function ResultsExploreLink() {
  return (
    <a
      href="/results"
      className="group not-prose flex items-center justify-between rounded-xl border border-dashed border-blue-800/50 bg-blue-950/10 hover:border-blue-600/70 hover:bg-blue-950/20 px-5 py-4 transition-all mt-6"
    >
      <div>
        <p className="text-blue-400 font-semibold text-sm group-hover:text-blue-300 transition-colors">
          Full evaluation deep-dive
        </p>
        <p className="text-gray-500 text-xs mt-0.5">
          ROC curves · PR curves · Confusion matrix · Training dynamics · Grad-CAM gallery
        </p>
      </div>
      <span className="text-blue-600 group-hover:text-blue-400 group-hover:translate-x-0.5 transition-all text-xl shrink-0 ml-4">
        →
      </span>
    </a>
  );
}

/** "Dive into the architecture" CTA — links to the architecture deep-dive page */
function ArchExploreLink() {
  return (
    <a
      href="/architecture"
      className="group not-prose flex items-center justify-between rounded-xl border border-dashed border-blue-800/50 bg-blue-950/10 hover:border-blue-600/70 hover:bg-blue-950/20 px-5 py-4 transition-all mt-6"
    >
      <div>
        <p className="text-blue-400 font-semibold text-sm group-hover:text-blue-300 transition-colors">
          Dive into the architecture
        </p>
        <p className="text-gray-500 text-xs mt-0.5">
          Inception modules · Temporal attention · Parameter count · Training curves · ROC curves
        </p>
      </div>
      <span className="text-blue-600 group-hover:text-blue-400 group-hover:translate-x-0.5 transition-all text-xl shrink-0 ml-4">
        →
      </span>
    </a>
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

        <Section id="architecture" tag="03 · Architecture" title="InceptionTime + Temporal Attention">
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
          <ArchExploreLink />
        </Section>

        <Section id="results" tag="04 · Training & Results" title="How It Was Trained — and How It Did">
          <p>
            Training used <strong className="text-white">weighted binary cross-entropy</strong> with
            label smoothing (ε=0.05). Because HYP has 4.8× fewer examples than NORM, each class gets
            an inverse-frequency weight so the model can&apos;t cheat by ignoring rare conditions.
            A cosine annealing schedule dropped the learning rate from 1×10⁻³ to 1×10⁻⁶ over 100
            epochs; early stopping fired at <strong className="text-white">epoch 35</strong>.
          </p>

          {/* Metric definitions */}
          <div className="not-prose grid grid-cols-1 sm:grid-cols-3 gap-3 my-6">
            {[
              {
                label: "AUC-ROC",
                val: "0.928",
                sub: "macro, test fold 10",
                def: "Probability the model ranks a sick patient above a healthy one. 1.0 = perfect, 0.5 = coin flip.",
                color: "border-blue-900/50 bg-blue-950/10",
                valColor: "text-blue-400",
              },
              {
                label: "AUPRC",
                val: "0.763",
                sub: "macro, 5-class avg",
                def: "Like AUC-ROC but penalises false alarms more heavily — the right metric when disease is rare.",
                color: "border-purple-900/50 bg-purple-950/10",
                valColor: "text-purple-400",
              },
              {
                label: "Macro F1",
                val: "0.664",
                sub: "Youden threshold",
                def: "Balances precision (few false alarms) and recall (few missed cases). Youden picks the optimal threshold per class.",
                color: "border-emerald-900/50 bg-emerald-950/10",
                valColor: "text-emerald-400",
              },
            ].map((m) => (
              <div key={m.label} className={`rounded-xl border p-4 ${m.color}`}>
                <p className={`text-2xl font-bold tabular-nums font-mono ${m.valColor}`}>{m.val}</p>
                <p className="text-white font-semibold text-sm mt-1">{m.label}</p>
                <p className="text-gray-500 text-[10px] font-mono mb-2">{m.sub}</p>
                <p className="text-gray-400 text-xs leading-relaxed">{m.def}</p>
              </div>
            ))}
          </div>

          {/* Per-class results */}
          <p className="text-gray-300 text-sm mb-3">
            Per-class breakdown on the <strong className="text-white">1,506-record test fold</strong>:
          </p>
          <div className="not-prose overflow-x-auto">
            <table className="w-full text-sm border-collapse">
              <thead>
                <tr className="border-b border-gray-700">
                  {["Class", "Full name", "AUC-ROC", "AUPRC", "F1", "Comment"].map((h) => (
                    <th key={h} className="text-left py-2 px-3 text-gray-400 font-mono text-xs uppercase tracking-wider">
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {[
                  { cls: "NORM", name: "Normal Sinus Rhythm",    auc: "0.971", pr: "0.963", f1: "0.911", note: "Strongest class — normal patterns are consistent and plentiful.", hi: true },
                  { cls: "STTC", name: "ST/T-Wave Change",       auc: "0.947", pr: "0.817", f1: "0.727", note: "ST segment deflections are morphologically distinct and easy to locate.", hi: false },
                  { cls: "MI",   name: "Myocardial Infarction",  auc: "0.941", pr: "0.725", f1: "0.560", note: "Q-waves and ST elevation are clear; lower F1 reflects fewer test samples.", hi: false },
                  { cls: "CD",   name: "Conduction Disturbance", auc: "0.921", pr: "0.866", f1: "0.761", note: "Wide QRS is a reliable marker; second largest test class.", hi: false },
                  { cls: "HYP",  name: "Hypertrophy",            auc: "0.860", pr: "0.444", f1: "0.364", note: "Weakest — voltage criteria are subtle and only 132 test samples.", hi: false },
                ].map((r) => (
                  <tr key={r.cls} className={`border-b border-gray-800 ${r.hi ? "bg-blue-950/10" : ""}`}>
                    <td className="py-2.5 px-3 font-mono text-xs text-blue-300">{r.cls}</td>
                    <td className="py-2.5 px-3 text-gray-300 text-xs">{r.name}</td>
                    <td className="py-2.5 px-3 text-white font-semibold tabular-nums text-xs">{r.auc}</td>
                    <td className="py-2.5 px-3 text-gray-300 tabular-nums text-xs">{r.pr}</td>
                    <td className="py-2.5 px-3 text-gray-300 tabular-nums text-xs">{r.f1}</td>
                    <td className="py-2.5 px-3 text-gray-500 text-xs">{r.note}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Literature comparison */}
          <div className="not-prose rounded-xl border border-green-900/40 bg-green-950/10 px-4 py-4 mt-6">
            <p className="text-[10px] font-mono uppercase tracking-widest text-green-500 mb-2">
              Literature benchmark
            </p>
            <p className="text-gray-300 text-sm leading-relaxed">
              The Strodthoff et al. 2020 benchmark puts the InceptionTime reference at{" "}
              <strong className="text-white">0.925 macro AUC-ROC</strong> and the strongest model
              (xresnet1d101) at <strong className="text-white">0.931</strong> — using ~3× the
              parameters. This project scores <strong className="text-green-400">0.928</strong> —
              above the InceptionTime reference and within 0.003 of the largest published baseline.
            </p>
          </div>

          <ResultsExploreLink />
        </Section>

        <Section id="summary" tag="05 · Summary" title="What Was Built — and What Comes Next">

          {/* What was achieved */}
          <p>
            A 12-lead ECG arrhythmia classifier that matches published literature on the largest
            open clinical ECG dataset — in a full end-to-end stack from raw signal to a live
            web demo. Every component was justified by data, not convention.
          </p>

          <div className="not-prose grid grid-cols-2 md:grid-cols-4 gap-3 my-6">
            {[
              { val: "0.928", label: "Macro AUC-ROC", sub: "literature-equivalent" },
              { val: "244k",  label: "Parameters",    sub: "3× lighter than SOTA" },
              { val: "1.01 MB", label: "ONNX model",  sub: "opset 17" },
              { val: "7 ms",  label: "API latency",   sub: "Fly.io always-on" },
            ].map((m) => (
              <div key={m.label} className="rounded-xl border border-gray-800 bg-navy-950 p-4 text-center">
                <p className="text-xl font-bold text-blue-400 tabular-nums font-mono">{m.val}</p>
                <p className="text-white font-medium mt-1 text-sm">{m.label}</p>
                <p className="text-gray-500 text-xs mt-0.5">{m.sub}</p>
              </div>
            ))}
          </div>

          {/* EE principles */}
          <div className="not-prose rounded-xl border border-blue-900/40 bg-blue-950/10 px-5 py-5 my-6">
            <p className="text-[10px] font-mono uppercase tracking-widest text-blue-400 mb-4">
              Electrical engineering principles applied
            </p>
            <div className="space-y-3">
              {[
                {
                  principle: "Signal is morphological, not spectral",
                  detail: "PSD showed all 5 classes share the same 0–40 Hz bands. The diagnostic information lives in waveform shape — this ruled out frequency-domain approaches and motivated a purely time-domain convolutional architecture.",
                },
                {
                  principle: "Multi-scale sensing matches the physical timescales",
                  detail: "At 100 Hz, QRS complexes span ~10 samples, T-waves ~20, RR intervals ~80. The k=10/20/40 kernels were sized directly to these measured timescales — not picked by grid search.",
                },
                {
                  principle: "Z-score normalisation preserves morphology, removes amplitude bias",
                  detail: "Each lead is normalised independently (μ, σ per lead per record). This removes inter-patient height/weight amplitude variation while keeping the relative shape of the PQRST complex intact.",
                },
                {
                  principle: "Soft attention as a learned filter bank in time",
                  detail: "The temporal attention mechanism α_t is analogous to a time-varying matched filter — it up-weights samples that best match the learned diagnostic templates and down-weights baseline noise.",
                },
              ].map((p) => (
                <div key={p.principle} className="flex gap-3">
                  <span className="text-blue-500 mt-0.5 shrink-0">→</span>
                  <div>
                    <p className="text-white font-semibold text-sm">{p.principle}</p>
                    <p className="text-gray-400 text-xs leading-relaxed mt-0.5">{p.detail}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Practical use */}
          <p className="font-semibold text-white">Practical application</p>
          <p>
            This model is designed as a <strong className="text-white">triage assistant</strong>,
            not a replacement for clinical judgement. The natural deployment is a two-stage
            pipeline: the model flags high-confidence abnormals for immediate specialist review
            and routes clear normals through a fast-lane — compressing the backlog without
            removing the cardiologist from any final decision. Grad-CAM and attention overlays
            are included precisely so the specialist can see{" "}
            <em>why</em> the model flagged a record, not just that it did.
          </p>

          {/* Improvements */}
          <p className="font-semibold text-white mt-4">Where it can go further</p>
          <div className="not-prose grid grid-cols-1 sm:grid-cols-2 gap-3 mt-3">
            {[
              {
                label: "Richer label set",
                text: "Expand from 5 superclasses back toward the full 71 SCP-ECG codes. This requires more data per code — a multi-institution federation could help.",
              },
              {
                label: "Patient-level context",
                text: "The model currently sees one 10-second snapshot. Adding age, sex, and prior ECG as conditioning signals could substantially improve HYP performance, where demographic voltage norms matter.",
              },
              {
                label: "Beat-level supervision",
                text: "Augmenting training with cardiologist-annotated beat boundaries (P onset, QRS, T offset) would sharpen Grad-CAM localisation and reduce reliance on the full-signal average.",
              },
              {
                label: "Uncertainty quantification",
                text: "MC Dropout or deep ensembles would give calibrated confidence intervals — essential before any clinical deployment, so the model can flag ambiguous cases rather than forcing a decision.",
              },
            ].map((item) => (
              <div key={item.label} className="rounded-lg border border-gray-800 bg-gray-900/60 p-4 space-y-1.5">
                <p className="text-white font-semibold text-sm">{item.label}</p>
                <p className="text-gray-500 text-xs leading-relaxed">{item.text}</p>
              </div>
            ))}
          </div>
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
