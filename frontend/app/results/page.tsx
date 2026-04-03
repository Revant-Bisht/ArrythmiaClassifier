import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Evaluation & Results — Revant Bisht",
  description:
    "Full evaluation of the InceptionTime + Temporal Attention arrhythmia classifier — ROC curves, precision-recall curves, confusion matrix, benchmark comparison, training dynamics, and Grad-CAM gallery.",
};

const GITHUB_NB =
  "https://github.com/Revant-Bisht/ArrythmiaClassifier/blob/main/notebooks/03_evaluation.ipynb";

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
        Key finding
      </p>
      <p className="text-gray-300 text-sm leading-relaxed">{children}</p>
    </div>
  );
}

// ─── Page ────────────────────────────────────────────────────────────────────

export default function ResultsPage() {
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
          03_evaluation.ipynb
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
            Evaluation · Explainability · Benchmark Comparison
          </p>
          <h1 className="text-4xl font-bold text-white leading-tight">
            Test Set Evaluation
          </h1>
          <p className="text-gray-400 text-base leading-relaxed">
            All metrics computed on <strong className="text-white">PTB-XL test fold 10</strong> —
            1,506 records never seen during training or threshold tuning.
          </p>
          <div className="flex flex-wrap gap-2 text-xs font-mono">
            {[
              "1,506 test records",
              "Macro AUC-ROC 0.928",
              "Macro AUPRC 0.763",
              "Macro F1 0.664",
              "Epoch 35",
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

        {/* ── 1  Summary Metrics ── */}
        <div>
          <CellHeader n="1" title="Test Set Metrics Summary" />

          <Out>
            <TextOut>{`Test samples: 1506
Macro AUC-ROC: 0.9280
Macro AUPRC:   0.7628
Macro F1:      0.6644`}</TextOut>
          </Out>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 mt-2">
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

          <div className="overflow-x-auto mt-6">
            <table className="w-full text-sm border-collapse">
              <thead>
                <tr className="border-b border-gray-700">
                  {["Class", "Full Name", "Test N", "AUC-ROC", "AUPRC", "F1 (Youden)"].map((h) => (
                    <th key={h} className="text-left py-2 px-3 text-gray-400 font-mono text-xs uppercase tracking-wider">
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {[
                  { cls: "NORM", name: "Normal Sinus Rhythm",    n: "721", auc: "0.971", pr: "0.963", f1: "0.911" },
                  { cls: "STTC", name: "ST/T-Wave Change",       n: "271", auc: "0.947", pr: "0.817", f1: "0.727" },
                  { cls: "MI",   name: "Myocardial Infarction",  n: "171", auc: "0.941", pr: "0.725", f1: "0.560" },
                  { cls: "CD",   name: "Conduction Disturbance", n: "456", auc: "0.921", pr: "0.866", f1: "0.761" },
                  { cls: "HYP",  name: "Hypertrophy",            n: "132", auc: "0.860", pr: "0.444", f1: "0.364" },
                ].map((r) => (
                  <tr key={r.cls} className="border-b border-gray-800">
                    <td className="py-2.5 px-3 font-mono text-xs text-blue-300">{r.cls}</td>
                    <td className="py-2.5 px-3 text-gray-300 text-xs">{r.name}</td>
                    <td className="py-2.5 px-3 text-gray-500 tabular-nums text-xs">{r.n}</td>
                    <td className="py-2.5 px-3 text-white font-semibold tabular-nums text-xs">{r.auc}</td>
                    <td className="py-2.5 px-3 text-gray-300 tabular-nums text-xs">{r.pr}</td>
                    <td className="py-2.5 px-3 text-gray-300 tabular-nums text-xs">{r.f1}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <Insight>
            HYP (Hypertrophy) is the weakest class across all three metrics — AUC 0.860, AUPRC 0.444,
            F1 0.364. It has the fewest test samples (132) and the most subtle diagnostic signature:
            hypertrophy is diagnosed by voltage amplitude criteria, which overlap heavily with normal
            variation across body types. All other classes exceed 0.921 AUC.
          </Insight>
        </div>

        {/* ── 2  ROC Curves ── */}
        <div>
          <CellHeader n="2" title="Per-Class ROC Curves" />
          <p className="text-sm text-gray-300 leading-relaxed mb-4">
            ROC curves plot true positive rate vs. false positive rate at every threshold.
            Area under the curve (AUC) summarises discrimination across all operating points.
          </p>
          <Out>
            <NbImg
              src="/eval/05_roc_curves.png"
              alt="Per-class ROC curves on test fold 10"
              caption="ROC curves — test fold 10. Each curve is one class vs. all others."
              wide
            />
          </Out>
          <Insight>
            NORM and STTC both achieve AUC ≥ 0.947. The HYP curve drops noticeably below the
            others — visually confirming it is the hardest class. Despite this, 0.860 is still
            well above random (0.5) and comparable to published baselines for hypertrophy.
          </Insight>
        </div>

        {/* ── 3  PR Curves ── */}
        <div>
          <CellHeader n="3" title="Precision-Recall Curves" />
          <p className="text-sm text-gray-300 leading-relaxed mb-4">
            Precision-recall curves are more informative than ROC curves for imbalanced classes.
            A random classifier achieves AUPRC equal to the class prevalence — not 0.5.
          </p>
          <Out>
            <NbImg
              src="/eval/07_pr_curves.png"
              alt="Precision-Recall curves on test fold 10"
              caption="Precision-Recall curves — test fold 10. Dashed baseline = class prevalence (random classifier)."
              wide
            />
          </Out>
          <Insight>
            HYP (8.8% of test set) has the lowest baseline (0.088) and the lowest AUPRC (0.444) —
            still 5× better than random. MI and STTC trade off precision and recall differently:
            MI favours recall (catching every infarction matters more than false alarms).
          </Insight>
        </div>

        {/* ── 4  Confusion Matrix ── */}
        <div>
          <CellHeader n="4" title="Confusion Matrix" />
          <p className="text-sm text-gray-300 leading-relaxed mb-4">
            Computed on single-label records only, using per-class Youden thresholds.
            Rows = true label, columns = predicted label.
          </p>
          <Out>
            <NbImg
              src="/eval/09_confusion_matrix.png"
              alt="Confusion matrix on test fold 10"
              caption="Confusion matrix — single-label records, Youden thresholds"
              wide
            />
          </Out>
          <Insight>
            The diagonal is dominant across all classes. The most common error is HYP being
            misclassified as NORM — expected, since hypertrophy&apos;s voltage criterion falls
            within the normal range for tall or muscular patients. CD and STTC occasionally
            confuse each other because both affect the QRS/ST region.
          </Insight>
        </div>

        {/* ── 5  Benchmark Comparison ── */}
        <div>
          <CellHeader n="5" title="Benchmark Comparison" />
          <p className="text-sm text-gray-300 leading-relaxed mb-4">
            Compared against the Strodthoff et al. 2020 PTB-XL benchmark — the standard reference
            for this dataset. All models evaluated on the same test fold 10.
          </p>
          <Out>
            <NbImg
              src="/eval/11_benchmark_comparison.png"
              alt="Benchmark comparison bar chart"
              caption="Macro AUC-ROC comparison — Strodthoff et al. 2020 baselines vs. this work"
              wide
            />
          </Out>

          <div className="overflow-x-auto mt-4">
            <table className="w-full text-sm border-collapse">
              <thead>
                <tr className="border-b border-gray-700">
                  {["Model", "Macro AUC-ROC", "Notes"].map((h) => (
                    <th key={h} className="text-left py-2 px-3 text-gray-400 font-mono text-xs uppercase tracking-wider">
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {[
                  { model: "Simple 1D CNN",                 auc: "0.890", note: "Strodthoff et al. 2020",              hi: false },
                  { model: "LSTM + Bidir Attention",         auc: "0.907", note: "Strodthoff et al. 2020",              hi: false },
                  { model: "InceptionTime (reference)",      auc: "0.925", note: "Strodthoff et al. 2020 — same arch",  hi: false },
                  { model: "This work (InceptionTime+Attn)", auc: "0.928", note: "~244k params — above reference",       hi: true  },
                  { model: "xresnet1d101",                   auc: "0.931", note: "Strodthoff et al. 2020 — 3× params",  hi: false },
                ].map((r) => (
                  <tr key={r.model} className={`border-b border-gray-800 ${r.hi ? "bg-green-950/15" : ""}`}>
                    <td className={`py-2.5 px-3 text-xs ${r.hi ? "text-green-300 font-semibold" : "text-gray-300"}`}>{r.model}</td>
                    <td className={`py-2.5 px-3 tabular-nums text-xs font-semibold ${r.hi ? "text-green-400" : "text-gray-300"}`}>{r.auc}</td>
                    <td className="py-2.5 px-3 text-gray-500 text-xs">{r.note}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <Insight>
            At 0.928 macro AUC-ROC, this project exceeds the published InceptionTime reference (0.925)
            and sits within 0.003 of the strongest published baseline (xresnet1d101, 0.931) — which uses
            approximately 3× more parameters. Literature-equivalent performance at a fraction of the model size.
          </Insight>
        </div>

        {/* ── 6  Training Dynamics ── */}
        <div>
          <CellHeader n="6" title="Training Dynamics" />

          <Out>
            <TextOut>{`Best epoch 35 | train AUC 0.9518 | val AUC 0.9322 | gap 0.0196`}</TextOut>
          </Out>

          <Out>
            <NbImg
              src="/eval/13_training_dynamics.png"
              alt="Training loss and AUC curves across epochs"
              caption="Loss and macro AUC-ROC — train (solid) vs. validation (dashed)"
              wide
            />
          </Out>

          <Insight>
            The train/val AUC gap at convergence is 0.020 — indicating mild overfitting that was
            well-controlled by the combination of classifier dropout (0.3), weight decay (5×10⁻⁴),
            and lead dropout (0.2). The gap closed from an initial 0.065 with these three mechanisms.
          </Insight>
        </div>

        {/* ── 7  Grad-CAM Gallery ── */}
        <div>
          <CellHeader n="7" title="Grad-CAM Explainability Gallery" />

          <p className="text-sm text-gray-300 leading-relaxed mb-4">
            For each class, the highest-confidence correct prediction from the test set is shown.
            Lead II is used — the most diagnostically informative standard lead.
          </p>

          <div className="rounded-xl border border-gray-800 bg-gray-900/40 p-4 text-sm space-y-1.5 mb-4">
            <p className="text-gray-400 font-medium text-xs font-mono uppercase tracking-widest mb-2">Reading the overlay</p>
            <p className="text-gray-500 text-xs leading-relaxed">
              <span className="text-red-400 font-medium">Red background</span> — Grad-CAM activation:
              regions whose gradient magnitude most strongly influenced the prediction for that class.
            </p>
            <p className="text-gray-500 text-xs leading-relaxed">
              <span className="text-purple-400 font-medium">Purple line</span> — temporal attention α:
              the learned per-timestep focus weight from the soft-attention head.
            </p>
          </div>

          <Out>
            <NbImg
              src="/eval/16_gradcam_gallery.png"
              alt="Grad-CAM gallery — 5 classes"
              caption="Grad-CAM heatmaps — highest-confidence correct prediction per class, Lead II"
              wide
            />
          </Out>

          <div className="overflow-x-auto mt-5">
            <table className="w-full text-sm border-collapse">
              <thead>
                <tr className="border-b border-gray-700">
                  {["Class", "Expected diagnostic region", "Grad-CAM finding"].map((h) => (
                    <th key={h} className="text-left py-2 px-3 text-gray-400 font-mono text-xs uppercase tracking-wider">
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {[
                  { cls: "NORM", expected: "No focal abnormality — broad, low activation", found: "✓ Diffuse low-amplitude baseline activation" },
                  { cls: "MI",   expected: "Q-wave (0–80 ms), ST elevation post-QRS",     found: "✓ Peaks on early QRS & ST segment" },
                  { cls: "STTC", expected: "ST segment & T-wave (150–350 ms post-QRS)",   found: "✓ Activation concentrated on ST/T region" },
                  { cls: "CD",   expected: "Entire QRS complex widened (>120 ms)",         found: "✓ Broad activation spanning full QRS" },
                  { cls: "HYP",  expected: "Tall R or deep S peak (voltage criterion)",    found: "✓ Sharp peak at R-wave maximum" },
                ].map((r) => (
                  <tr key={r.cls} className="border-b border-gray-800">
                    <td className="py-2.5 px-3 font-mono text-xs text-blue-300">{r.cls}</td>
                    <td className="py-2.5 px-3 text-gray-400 text-xs">{r.expected}</td>
                    <td className="py-2.5 px-3 text-emerald-400 text-xs">{r.found}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <Insight>
            The Grad-CAM heatmaps are clinically plausible without any explicit supervision on waveform
            segments. The model learned to look at Q-waves for MI, the ST region for STTC, and R-wave
            peaks for HYP — consistent with AHA/ACC 12-lead ECG interpretation guidelines
            (Surawicz et al., 2009).
          </Insight>
        </div>

      </div>

      {/* Footer CTAs */}
      <section className="bg-navy-950 border-t border-gray-800 py-16 px-6 text-center">
        <div className="max-w-2xl mx-auto space-y-5">
          <h2 className="text-xl font-bold text-white">See the predictions live</h2>
          <p className="text-gray-400 text-sm">
            The interactive demo lets you select any of the 5 classes and see the model&apos;s
            Grad-CAM heatmaps and attention weights on a real PTB-XL ECG.
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
