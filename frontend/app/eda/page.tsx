import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "PTB-XL EDA — Revant Bisht",
  description:
    "Exploratory data analysis of the PTB-XL ECG dataset — class distribution, waveform morphology, frequency domain, timescale analysis, lead correlation, and design decisions.",
};

const GITHUB_NB =
  "https://github.com/Revant-Bisht/ArrythmiaClassifier/blob/main/notebooks/01_eda.ipynb";

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

function Decision({ title, reason }: { title: string; reason: string }) {
  return (
    <div className="rounded-lg border border-gray-800 bg-gray-900/60 p-4 space-y-1.5">
      <p className="text-white font-semibold text-sm">{title}</p>
      <p className="text-gray-500 text-xs leading-relaxed">{reason}</p>
    </div>
  );
}

// ─── Page ────────────────────────────────────────────────────────────────────

export default function EDAPage() {
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
          01_eda.ipynb
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
            Exploratory Data Analysis · PTB-XL v1.0.3
          </p>
          <h1 className="text-4xl font-bold text-white leading-tight">
            PTB-XL Exploratory Data Analysis
          </h1>
          <p className="text-gray-400 text-base leading-relaxed">
            12-lead ECG arrhythmia classification — dataset characterisation,
            label analysis, signal quality, clinical feature extraction, and
            design decisions for model training.
          </p>
          <div className="flex flex-wrap gap-2 text-xs font-mono">
            {[
              "PTB-XL v1.0.3",
              "21,799 records",
              "18,885 patients",
              "10 s @ 100 Hz",
              "12 leads",
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
              className="inline-flex items-center gap-2 px-5 py-2.5 rounded-lg bg-blue-600 hover:bg-blue-500 text-white font-medium text-sm transition-colors"
            >
              View full notebook on GitHub →
            </a>
          </div>
        </div>

        {/* ── Section divider style ── */}
        <div className="space-y-16 divide-y divide-gray-800/60">

          {/* ═══ 1 Dataset Overview ═══════════════════════════════════════ */}
          <div className="pt-10 space-y-5">
            <CellHeader n="3" title="1  Dataset Overview" />

            <Out>
              <TextOut>{`Total records:          21,799
Unique patients:        18,869
Records with ≥1 label:  15,130
Records with 0 labels:   6,669  (no confident superdiagnostic code)

Split sizes (strat_fold):
  train  folds [1–8]:  12,133 records
  val    fold  [9]:     1,491 records
  test   fold  [10]:    1,506 records`}</TextOut>
            </Out>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              {[
                { v: "21,799", l: "Total records",   s: "PTB-XL v1.0.3" },
                { v: "15,130", l: "Labeled subset",  s: "100% likelihood only" },
                { v: "12,133", l: "Training records", s: "folds 1–8" },
                { v: "1,506",  l: "Test records",     s: "fold 10, held out" },
              ].map(({ v, l, s }) => (
                <div key={l} className="rounded-xl border border-gray-800 bg-navy-950 p-4 text-center">
                  <p className="text-2xl font-bold text-blue-400 tabular-nums">{v}</p>
                  <p className="text-white text-xs font-medium mt-1">{l}</p>
                  <p className="text-gray-600 text-[10px] mt-0.5">{s}</p>
                </div>
              ))}
            </div>
          </div>

          {/* ═══ 2 Label Analysis ═════════════════════════════════════════ */}
          <div className="pt-10 space-y-5">
            <CellHeader n="5–8" title="2  Label Analysis" />

            <p className="text-gray-400 text-sm leading-relaxed">
              Per-class counts after the 100%-likelihood filter. Multi-label records
              (15.2%) carry two or more simultaneous superclass diagnoses.
            </p>

            <Out>
              <TextOut>{`class  count       pct
 NORM   7172    47.4 %
   CD   4508    29.8 %
 STTC   2716    18.0 %
   MI   1997    13.2 %
  HYP   1468     9.7 %

Labels per record:
  1 label   12,833  (84.8 %)
  2 labels   1,898  (12.5 %)
  3 labels     364   (2.4 %)
  4 labels      35   (0.2 %)

Multi-label records: 2,297  (15.2 %)`}</TextOut>
            </Out>

            <div className="space-y-4">
              <Out>
                <NbImg
                  src="/eda/02_class_distribution.png"
                  alt="Class distribution bar chart"
                  caption="Fig 1 — Per-class record counts. NORM dominates; HYP is rarest."
                />
              </Out>
              <Out>
                <NbImg
                  src="/eda/02_multilabel_counts.png"
                  alt="Multi-label distribution bar chart"
                  caption="Fig 2 — Distribution of labels per record. 84.8% are single-label."
                />
              </Out>
              <Out>
                <NbImg
                  src="/eda/02_cooccurrence_matrix.png"
                  alt="Class co-occurrence heatmap"
                  caption="Fig 3 — Co-occurrence matrix. NORM rarely appears with MI; STTC and CD frequently co-occur."
                />
              </Out>
            </div>

            <Out>
              <TextOut>{`Training set class weights (inverse frequency):
  N=12,133   C=5

  NORM   n=5,743   weight=0.4225  (majority — down-weighted)
  CD     n=3,600   weight=0.6741
  STTC   n=2,174   weight=1.1162
  MI     n=1,637   weight=1.4823
  HYP    n=1,202   weight=2.0188

Weight ratio max/min = 4.78×
Label smoothing ε=0.05 is safe at this imbalance ratio.`}</TextOut>
            </Out>

            <Insight>
              Multi-label co-occurrence rules out softmax — five{" "}
              <strong>independent sigmoid outputs</strong> with weighted BCE loss.
              Class weights span 4.78×, making per-class AUC-ROC the only fair
              evaluation metric.
            </Insight>
          </div>

          {/* ═══ 3 Demographics ════════════════════════════════════════════ */}
          <div className="pt-10 space-y-5">
            <CellHeader n="10" title="3  Patient Demographics" />

            <Out>
              <TextOut>{`Missing values:
  age     :     0   (0.0 %)
  sex     :     0   (0.0 %)
  height  : 10,341  (68.3 %)
  weight  :  8,452  (55.9 %)

Age  — mean 62.3  std 33.4  range [2, 300]
Sex  — female: 7,920   male: 7,210`}</TextOut>
            </Out>

            <Out>
              <NbImg
                src="/eda/03_demographics.png"
                alt="Demographic distributions: age and sex"
                caption="Fig 4 — Age and sex distribution. Skewed older; near-equal sex split. Height and weight are largely missing and excluded from features."
              />
            </Out>

            <Insight>
              Height and weight are missing for the majority of records — they cannot
              be used as features. Age and sex are complete and could serve as
              auxiliary inputs, but the model was trained on signal alone to avoid
              demographic shortcuts.
            </Insight>
          </div>

          {/* ═══ 4 Signal Quality ══════════════════════════════════════════ */}
          <div className="pt-10 space-y-5">
            <CellHeader n="12" title="4  Signal Quality Audit" />

            <Out>
              <TextOut>{`Signal shape verification:
  All 1000 samples:  ✓
  All 12 leads:      ✓

NaN rate:  mean=0.0000 %   max=0.000 %
Flat lead: mean=0.00 %

Quality flags (% of labeled records affected):
  baseline_drift      : 1,135  (7.5 %)
  static_noise        : 2,286  (15.1 %)
  burst_noise         :   412  (2.7 %)
  electrodes_problems :    21  (0.1 %)
  extra_beats         : 1,226  (8.1 %)
  pacemaker           :     7  (0.0 %)`}</TextOut>
            </Out>

            <Insight>
              No NaN or flat-lead issues — the 100%-likelihood filter already removed
              the lowest-quality annotations. Static noise affects 15% of records,
              motivating the <strong>Gaussian noise augmentation</strong> (σ=0.01) and{" "}
              <strong>lead dropout</strong> (p=0.2) used during training.
            </Insight>
          </div>

          {/* ═══ 5 Waveform Gallery ════════════════════════════════════════ */}
          <div className="pt-10 space-y-5">
            <CellHeader n="14" title="5  Waveform Visualisation" />
            <p className="text-gray-400 text-sm leading-relaxed">
              One representative 12-lead ECG per class (pure single-label records,
              selected at random with a fixed seed). Each row is one class; each
              column is one of the 12 standard leads.
            </p>

            <Out>
              <NbImg
                src="/eda/05_waveform_gallery.png"
                alt="5×12 ECG waveform gallery — one row per class, one column per lead"
                caption="Fig 5 — 5 classes × 12 leads. Morphological differences are visible: ST elevation in MI (row 2), widened QRS in CD (row 4), tall R-waves in HYP (row 5)."
                wide
              />
            </Out>

            <Insight>
              The discriminative signal is <strong>morphological</strong> — it lives in
              the shape of the PQRST complex. All five classes are visually
              distinguishable by waveform geometry, not by frequency content.
              This directly rules out FFT-based features and motivates a
              convolutional architecture.
            </Insight>
          </div>

          {/* ═══ 6 Amplitude & Noise ═══════════════════════════════════════ */}
          <div className="pt-10 space-y-5">
            <CellHeader n="16" title="6  Amplitude & Noise Analysis" />
            <p className="text-gray-400 text-sm leading-relaxed">
              Per-lead signal standard deviation in raw mV (pre-normalisation),
              computed on a 300-record sample. Used to calibrate the Gaussian noise
              augmentation magnitude.
            </p>

            <Out>
              <TextOut>{`Per-lead signal std (mV) — raw, pre-normalisation:
  Lead      mean   median      p5      p95
  I        0.144   0.128    0.072   0.266
  II       0.155   0.137    0.068   0.300
  III      0.143   0.111    0.051   0.342
  aVR      0.131   0.125    0.067   0.212
  aVL      0.121   0.098    0.045   0.273
  aVF      0.130   0.109    0.045   0.291
  V1       0.206   0.164    0.083   0.475
  V2       0.308   0.244    0.121   0.690
  V3       0.296   0.246    0.137   0.715
  V4       0.264   0.242    0.130   0.465
  V5       0.255   0.223    0.113   0.501
  V6       0.213   0.187    0.096   0.419`}</TextOut>
            </Out>

            <Insight>
              Precordial leads (V1–V6) have 2–3× higher amplitude than limb leads —
              a known clinical property. Gaussian noise at σ=0.01 represents roughly
              1% of the median lead-II amplitude (0.137 mV), which is small enough
              to be realistic and large enough to act as regularisation.
            </Insight>
          </div>

          {/* ═══ 7 Clinical Features ═══════════════════════════════════════ */}
          <div className="pt-10 space-y-5">
            <CellHeader n="18–19" title="7  Clinical Feature Analysis — R-peaks & Heart Rate" />

            <Out>
              <NbImg
                src="/eda/07_rpeak_detection.png"
                alt="R-peak detection on NORM representative"
                caption="Fig 6 — R-peak detection (neurokit2) on a NORM representative. Heart rate: 78.0 ± 1.8 bpm."
              />
            </Out>

            <Out>
              <NbImg
                src="/eda/07_hr_distribution.png"
                alt="Heart rate distribution per class"
                caption="Fig 7 — Heart rate distribution per class (50 records per class). HYP patients have notably higher resting HR."
              />
            </Out>

            <Out>
              <TextOut>{`Median HR per class:
  NORM:  67.5 bpm
  CD:    71.5 bpm
  MI:    76.0 bpm
  STTC:  78.2 bpm
  HYP:   85.7 bpm`}</TextOut>
            </Out>

            <Insight>
              HYP patients have the highest resting heart rate (85.7 bpm vs 67.5 for
              NORM). Beat-to-beat rhythm information — captured by the k=40 kernel at
              the RR-interval timescale — carries class-discriminative signal beyond
              just waveform shape.
            </Insight>
          </div>

          {/* ═══ 8 Frequency Domain ════════════════════════════════════════ */}
          <div className="pt-10 space-y-5">
            <CellHeader n="21" title="8  Frequency Domain Analysis" />
            <p className="text-gray-400 text-sm leading-relaxed">
              Average power spectral density per class on Lead II (30 records
              each). If classes had distinct frequency signatures, spectral features
              would work.
            </p>

            <Out>
              <NbImg
                src="/eda/08_psd_comparison.png"
                alt="Power spectral density per class"
                caption="Fig 8 — Average PSD (Lead II). All five classes share essentially identical frequency content in 0–40 Hz."
              />
            </Out>

            <Insight>
              <strong>The classes are not separable by frequency content.</strong> All
              five PSD curves overlap in the same 0–40 Hz band. This definitively
              rules out FFT features, mel-spectrograms, or any frequency-domain
              approach. The information is in <em>when</em> and <em>how</em> the
              signal changes shape — not in which frequencies are present.
            </Insight>
          </div>

          {/* ═══ 9 Lead Correlation & PCA ═════════════════════════════════ */}
          <div className="pt-10 space-y-5">
            <CellHeader n="23–24" title="9  Lead Analysis — Correlation & PCA" />

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="space-y-3">
                <Out>
                  <NbImg
                    src="/eda/09_lead_correlation.png"
                    alt="12×12 lead correlation matrix"
                    caption="Fig 9 — Lead-lead correlation (averaged over 200 records). Limb leads (I–aVF) and precordial leads (V1–V6) form distinct correlated blocks."
                  />
                </Out>
              </div>
              <div className="space-y-3">
                <Out>
                  <NbImg
                    src="/eda/09_pca_scatter.png"
                    alt="PCA scatter plot of 5 classes"
                    caption="Fig 10 — PCA projection (Lead II, 40 pure records per class). Complete class overlap in the first two principal components."
                  />
                </Out>
              </div>
            </div>

            <Insight>
              <strong>Lead correlation:</strong> Limb and precordial leads form correlated
              blocks, but each lead captures unique spatial territory — none is fully
              redundant. All 12 leads are fed to the model as a (12 × 1000) tensor.
              <br /><br />
              <strong>PCA:</strong> Complete class overlap in the first 20 principal
              components. Linear separation is impossible. A deep non-linear model is
              necessary — this rules out SVMs with linear kernels, logistic regression,
              and classical feature engineering.
            </Insight>
          </div>

          {/* ═══ 10 Design Decisions ══════════════════════════════════════ */}
          <div className="pt-10 space-y-5">
            <CellHeader n="26" title="10  Design Decisions — What the EDA Decided" />
            <p className="text-gray-400 text-sm leading-relaxed">
              Every entry below traces directly to an EDA finding above.
            </p>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              <Decision
                title="Multi-label BCE loss (not softmax)"
                reason="15.2% of records carry ≥2 simultaneous labels. Co-occurrence matrix shows MI+CD and STTC+HYP are common. Softmax would force mutual exclusion — BCE treats each class independently."
              />
              <Decision
                title="Class-weighted loss (4.78× max ratio)"
                reason="NORM is 4.9× more frequent than HYP. Without weights, the model can ignore HYP and achieve high overall accuracy. Weights are exact inverse-frequency from the training split."
              />
              <Decision
                title="Convolutional architecture"
                reason="PSD shows no frequency separation (§8). The discriminative signal is morphological (§5). Convolutions learn waveform shape directly from the time-series."
              />
              <Decision
                title="Kernel sizes k=10, 20, 40"
                reason="QRS ≈ 10 samples (~100 ms), T-wave ≈ 20 samples (~200 ms), RR interval ≈ 80 samples (~400 ms) at 100 Hz. One filter per physiological timescale."
              />
              <Decision
                title="All 12 leads as input (12 × 1000)"
                reason="Lead correlation (§9) shows each lead carries unique spatial territory. Dropping any lead loses information. Input tensor is (B, 12, 1000)."
              />
              <Decision
                title="Gaussian noise σ=0.01 augmentation"
                reason="Amplitude analysis (§6) shows lead-II median std ≈ 0.137 mV. σ=0.01 is ~7% of that — realistic SNR perturbation. Static noise affects 15% of records (§4)."
              />
              <Decision
                title="Lead dropout p=0.2"
                reason="Signal quality audit (§4) shows electrode problems affect ~0.1% of records, but static/burst noise affects ~18%. Lead dropout simulates realistic missing-lead scenarios."
              />
              <Decision
                title="Macro AUC-ROC as primary metric"
                reason="4.78× class imbalance means overall accuracy is misleading. AUC-ROC is threshold-free and treats each class equally regardless of prevalence."
              />
            </div>

            <Out>
              <TextOut>{`TASK FORMULATION
  Multi-label binary classification — 5 independent sigmoid outputs.
  BCE loss, NOT softmax/CE. Records carry 1–3 simultaneous labels.
  84.8% of records have exactly 1 label; 12.5% have 2; 2.4% have 3.

CLASS WEIGHTS  (training split, N=12,133)
  NORM  n=5,743   w=0.4225   (majority — down-weighted)
  CD    n=3,600   w=0.6741
  STTC  n=2,174   w=1.1162
  MI    n=1,637   w=1.4823
  HYP   n=1,202   w=2.0188   (rarest — up-weighted 4.78×)

ARCHITECTURE
  InceptionTime — parallel Conv1D kernels k=10, 20, 40
  Motivated by QRS/T-wave/RR timescales from §7 clinical analysis.
  All 12 leads processed jointly as (B, 12, 1000) input tensor.

NORMALISATION
  Z-score per lead per record (zero-mean, unit-variance).
  Applied at load time; augmentation added on top.

AUGMENTATION
  Gaussian noise:  σ=0.01  (calibrated from §6 amplitude analysis)
  Lead dropout:    p=0.20  (calibrated from §4 quality audit)
  Time shift:      ±50 samples  (preserves QRS, varies beat phase)`}</TextOut>
            </Out>
          </div>

        </div>

        {/* ── Notebook CTA ── */}
        <div className="rounded-2xl border border-gray-800 bg-navy-950 p-8 text-center space-y-4 mt-8">
          <p className="text-blue-400 text-xs font-mono tracking-widest uppercase">
            01_eda.ipynb · 27 cells · 10 sections
          </p>
          <h2 className="text-xl font-bold text-white">
            View the full notebook on GitHub
          </h2>
          <p className="text-gray-400 text-sm max-w-md mx-auto">
            Includes all computed outputs — interactive plots, waveform galleries,
            R-peak detection, and the full design-decisions printout.
          </p>
          <a
            href={GITHUB_NB}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-2 px-7 py-3 rounded-lg bg-blue-600 hover:bg-blue-500 text-white font-medium transition-colors"
          >
            Open 01_eda.ipynb →
          </a>
        </div>

        {/* ── Footer nav ── */}
        <div className="flex flex-wrap items-center justify-between gap-4 pt-4 border-t border-gray-800 text-sm">
          <a href="/blog" className="text-gray-400 hover:text-white transition-colors">
            ← Technical writeup
          </a>
          <a href="/blog#architecture" className="text-gray-400 hover:text-white transition-colors">
            Architecture →
          </a>
        </div>
      </div>

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
