import { Hero } from "@/components/Hero";

// ── Inline SVG illustrations ────────────────────────────────────────────────

function MultiScaleSVG() {
  return (
    <svg viewBox="0 0 210 72" className="w-full h-auto">
      {/* Shaded scanning windows (widest first so narrower ones paint on top) */}
      <rect x="22" y="2" width="82" height="30" rx="3"
        fill="#34d399" fillOpacity={0.06} stroke="#34d399" strokeOpacity={0.3} strokeWidth={0.7} strokeDasharray="3,2" />
      <rect x="40" y="2" width="46" height="30" rx="3"
        fill="#a78bfa" fillOpacity={0.08} stroke="#a78bfa" strokeOpacity={0.4} strokeWidth={0.7} strokeDasharray="3,2" />
      <rect x="53" y="2" width="20" height="30" rx="3"
        fill="#60a5fa" fillOpacity={0.13} stroke="#60a5fa" strokeOpacity={0.55} strokeWidth={0.9} />

      {/* ECG line  baseline y=18  QRS centred at x=63 */}
      <path
        d="M0,18 L20,18 C22,18 24,11 28,11 C32,11 34,18 36,18 L50,18 L52,24 L56,2 L60,28 L63,18
           L110,18 C113,18 117,8 123,8 C129,8 132,18 135,18 L210,18"
        stroke="#9ca3af" strokeWidth={1.5} fill="none" strokeLinecap="round"
      />

      {/* Bracket spans */}
      {/* k=10 */}
      <line x1="53" y1="38" x2="73" y2="38" stroke="#60a5fa" strokeWidth={2} strokeLinecap="round" />
      <line x1="53" y1="35" x2="53" y2="41" stroke="#60a5fa" strokeWidth={1.2} />
      <line x1="73" y1="35" x2="73" y2="41" stroke="#60a5fa" strokeWidth={1.2} />
      <text x="77" y="41" fontSize={8} fill="#60a5fa" fontFamily="ui-monospace,monospace" dominantBaseline="middle">k=10 · QRS spike</text>

      {/* k=20 */}
      <line x1="40" y1="51" x2="86" y2="51" stroke="#a78bfa" strokeWidth={2} strokeLinecap="round" />
      <line x1="40" y1="48" x2="40" y2="54" stroke="#a78bfa" strokeWidth={1.2} />
      <line x1="86" y1="48" x2="86" y2="54" stroke="#a78bfa" strokeWidth={1.2} />
      <text x="90" y="54" fontSize={8} fill="#a78bfa" fontFamily="ui-monospace,monospace" dominantBaseline="middle">k=20 · T-wave</text>

      {/* k=40 */}
      <line x1="22" y1="64" x2="104" y2="64" stroke="#34d399" strokeWidth={2} strokeLinecap="round" />
      <line x1="22" y1="61" x2="22" y2="67" stroke="#34d399" strokeWidth={1.2} />
      <line x1="104" y1="61" x2="104" y2="67" stroke="#34d399" strokeWidth={1.2} />
      <text x="108" y="67" fontSize={8} fill="#34d399" fontFamily="ui-monospace,monospace" dominantBaseline="middle">k=40 · beat interval</text>
    </svg>
  );
}

function InceptionSVG() {
  return (
    <svg viewBox="0 0 210 58" className="w-full h-auto">
      {/* Input boxes */}
      <rect x="2" y="4"  width="28" height="13" rx="3" fill="rgb(15,23,42)" stroke="#60a5fa" strokeWidth={0.9} />
      <text x="16" y="12" fontSize={7.5} fill="#60a5fa" textAnchor="middle" fontFamily="ui-monospace,monospace">k=10</text>

      <rect x="2" y="22" width="28" height="13" rx="3" fill="rgb(15,23,42)" stroke="#a78bfa" strokeWidth={0.9} />
      <text x="16" y="30" fontSize={7.5} fill="#a78bfa" textAnchor="middle" fontFamily="ui-monospace,monospace">k=20</text>

      <rect x="2" y="40" width="28" height="13" rx="3" fill="rgb(15,23,42)" stroke="#34d399" strokeWidth={0.9} />
      <text x="16" y="48" fontSize={7.5} fill="#34d399" textAnchor="middle" fontFamily="ui-monospace,monospace">k=40</text>

      {/* Converging lines */}
      <line x1="30" y1="10" x2="50" y2="29" stroke="#4b5563" strokeWidth={0.9} />
      <line x1="30" y1="29" x2="50" y2="29" stroke="#4b5563" strokeWidth={0.9} />
      <line x1="30" y1="47" x2="50" y2="29" stroke="#4b5563" strokeWidth={0.9} />

      {/* Block 1 */}
      <rect x="52" y="18" width="38" height="22" rx="4" fill="rgb(23,37,63)" stroke="#60a5fa" strokeWidth={1} />
      <text x="71" y="30" fontSize={7.5} fill="white" textAnchor="middle" fontFamily="ui-monospace,monospace">Block 1</text>

      <line x1="90" y1="29" x2="98" y2="29" stroke="#4b5563" strokeWidth={0.9} />

      {/* Block 2 */}
      <rect x="100" y="18" width="38" height="22" rx="4" fill="rgb(23,37,63)" stroke="#60a5fa" strokeWidth={1} strokeOpacity={0.65} />
      <text x="119" y="30" fontSize={7.5} fill="#d1d5db" textAnchor="middle" fontFamily="ui-monospace,monospace">Block 2</text>

      <line x1="138" y1="29" x2="146" y2="29" stroke="#4b5563" strokeWidth={0.9} />

      {/* Block 3 */}
      <rect x="148" y="18" width="38" height="22" rx="4" fill="rgb(23,37,63)" stroke="#60a5fa" strokeWidth={1} strokeOpacity={0.35} />
      <text x="167" y="30" fontSize={7.5} fill="#9ca3af" textAnchor="middle" fontFamily="ui-monospace,monospace">Block 3</text>

      <line x1="186" y1="29" x2="194" y2="29" stroke="#4b5563" strokeWidth={0.9} />

      {/* Output */}
      <text x="203" y="26" fontSize={7} fill="#60a5fa" textAnchor="middle" fontFamily="ui-monospace,monospace">5</text>
      <text x="203" y="34" fontSize={7} fill="#60a5fa" textAnchor="middle" fontFamily="ui-monospace,monospace">cls</text>
    </svg>
  );
}

function GradCAMSVG() {
  return (
    <svg viewBox="0 0 210 58" className="w-full h-auto">
      <defs>
        <linearGradient id="gc-qrs" x1="0" x2="1" y1="0" y2="0">
          <stop offset="0%"   stopColor="#ef4444" stopOpacity={0} />
          <stop offset="35%"  stopColor="#f97316" stopOpacity={0.55} />
          <stop offset="55%"  stopColor="#ef4444" stopOpacity={0.85} />
          <stop offset="75%"  stopColor="#f97316" stopOpacity={0.5} />
          <stop offset="100%" stopColor="#ef4444" stopOpacity={0} />
        </linearGradient>
        <linearGradient id="gc-t" x1="0" x2="1" y1="0" y2="0">
          <stop offset="0%"   stopColor="#f59e0b" stopOpacity={0} />
          <stop offset="40%"  stopColor="#fbbf24" stopOpacity={0.35} />
          <stop offset="60%"  stopColor="#f59e0b" stopOpacity={0.45} />
          <stop offset="100%" stopColor="#f59e0b" stopOpacity={0} />
        </linearGradient>
      </defs>

      {/* Heatmap overlays */}
      <rect x="40" y="4"  width="52" height="36" rx="3" fill="url(#gc-qrs)" />
      <rect x="108" y="6" width="58" height="32" rx="3" fill="url(#gc-t)"   />

      {/* ECG line  baseline y=26 */}
      <path
        d="M0,26 L20,26 C22,26 24,19 28,19 C32,19 34,26 36,26 L46,26 L48,33 L52,4 L56,38 L59,26
           L103,26 C106,26 110,14 116,14 C122,14 125,26 128,26 L210,26"
        stroke="#e5e7eb" strokeWidth={1.5} fill="none" strokeLinecap="round"
      />

      {/* Annotation labels */}
      <text x="64"  y="50" fontSize={8} fill="#ef4444" textAnchor="middle" fontFamily="ui-monospace,monospace">high activation</text>
      <text x="137" y="50" fontSize={8} fill="#f59e0b" textAnchor="middle" fontFamily="ui-monospace,monospace">moderate</text>
    </svg>
  );
}

function AttentionSVG() {
  const bars = [2, 3, 2, 4, 3, 4, 3, 4, 20, 30, 24, 7, 4, 3, 14, 18, 11, 4, 3, 2];
  const maxH = Math.max(...bars);
  const barW = 7, gap = 3, baseY = 44;
  const totalW = bars.length * (barW + gap) - gap;
  const startX = (210 - totalW) / 2;

  // QRS peak = index 9 (h=30), T peak = index 15 (h=18)
  const qrsCx = startX + 9 * (barW + gap) + barW / 2;   // ≈ 100
  const tCx   = startX + 15 * (barW + gap) + barW / 2;  // ≈ 160

  return (
    <svg viewBox="0 0 210 64" className="w-full h-auto">
      {/* Baseline */}
      <line x1="0" y1={baseY} x2="210" y2={baseY} stroke="#374151" strokeWidth={0.5} />

      {bars.map((h, i) => {
        const x = startX + i * (barW + gap);
        const isHigh = h >= maxH * 0.45;
        return (
          <rect
            key={i}
            x={x} y={baseY - h}
            width={barW} height={h}
            rx={1.5}
            fill={isHigh ? "#60a5fa" : "#374151"}
            opacity={isHigh ? 1 : 0.55}
          />
        );
      })}

      {/* "high α" labels above each peak — avoids bottom overlap */}
      <text x={qrsCx} y={baseY - 30 - 4} fontSize={7.5} fill="#60a5fa"
        textAnchor="middle" fontFamily="ui-monospace,monospace">high α</text>
      <text x={tCx}   y={baseY - 18 - 4} fontSize={7.5} fill="#60a5fa"
        textAnchor="middle" fontFamily="ui-monospace,monospace">high α</text>

      {/* Region labels below baseline */}
      <text x={qrsCx} y={baseY + 10} fontSize={7.5} fill="#6b7280"
        textAnchor="middle" fontFamily="ui-monospace,monospace">QRS</text>
      <text x={tCx}   y={baseY + 10} fontSize={7.5} fill="#6b7280"
        textAnchor="middle" fontFamily="ui-monospace,monospace">T-wave</text>
    </svg>
  );
}

// ── Page ────────────────────────────────────────────────────────────────────

export default function Home() {
  const cards = [
    {
      figure: <MultiScaleSVG />,
      title: "(1) Multi-scale filters — k=10, k=20, k=40",
      text: "The 3 filters each capture a different width of the signal — narrow for QRS spikes, medium for T-waves, wide for beat-to-beat intervals. All three run in parallel.",
    },
    {
      figure: <InceptionSVG />,
      title: "(2) InceptionTime — stacking the filters",
      text: "We combine the output of the 3 filters using Inception blocks. Stack three of these and the network builds up increasingly complex beat-level patterns.",
    },
    {
      figure: <GradCAMSVG />,
      title: "(3) Visualising model I: Grad-CAM",
      text: "After the model is trained, we can visualise the importance of different regions of the ECG using Grad-CAM — brighter regions drove the prediction more strongly.",
    },
    {
      figure: <AttentionSVG />,
      title: "(4) Visualising model II: Temporal Attention",
      text: "Similarly, the model learns a weight α for every time step. High α marks a diagnostically important moment — independently of how loud the signal is there.",
    },
  ];

  return (
    <main className="bg-navy-900 min-h-screen">
      <nav className="fixed top-0 left-0 right-0 z-50 flex items-center justify-between px-6 py-4 bg-navy-900/80 backdrop-blur-md border-b border-gray-800">
        <div className="flex items-center gap-3">
          <span className="text-white font-semibold">Revant Bisht</span>
          <span className="text-gray-600">·</span>
          <span className="text-gray-400 text-sm">Arrhythmia Classifier</span>
        </div>
        <div className="flex items-center gap-5 text-sm">
          <a href="/demo" className="text-gray-400 hover:text-white transition-colors">
            Demo
          </a>
          <a href="/blog" className="text-gray-400 hover:text-white transition-colors">
            How I Built It
          </a>
          <a
            href="https://github.com/Revant-Bisht/ArrythmiaClassifier"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-1.5 text-gray-400 hover:text-white transition-colors"
          >
            <svg className="w-4 h-4" viewBox="0 0 24 24" fill="currentColor">
              <path d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z" />
            </svg>
            GitHub
          </a>
        </div>
      </nav>

      <div className="pt-16">
        <Hero />

        {/* ── Model explainer ── */}
        <section className="bg-navy-900 border-t border-gray-800 py-16 px-6">
          <div className="max-w-5xl mx-auto">
            <p className="text-blue-400 text-xs font-mono tracking-widest uppercase mb-2">
              What you just watched
            </p>
            <h2 className="text-2xl font-bold text-white mb-8">
              How the model reads an ECG
            </h2>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-12">
              {cards.map(({ figure, title, text }) => (
                <div
                  key={title}
                  className="rounded-xl border border-gray-800 bg-navy-950 overflow-hidden"
                >
                  <div className="bg-gray-900/60 border-b border-gray-800 px-4 py-4">
                    {figure}
                  </div>
                  <div className="p-5 space-y-1.5">
                    <p className="text-white font-semibold text-sm">{title}</p>
                    <p className="text-gray-400 text-sm leading-relaxed">{text}</p>
                  </div>
                </div>
              ))}
            </div>

            {/* CTA */}
            <div className="flex flex-col items-center gap-3 text-center">
              <a
                href="/demo"
                className="inline-flex items-center gap-3 px-8 py-4 rounded-xl bg-blue-600 hover:bg-blue-500 text-white text-lg font-semibold transition-colors shadow-lg shadow-blue-900/30"
              >
                Try it yourself
                <span className="text-xl">→</span>
              </a>
              <p className="text-gray-600 text-xs">
                Select a condition · see Grad-CAM · read the model&apos;s clinical report
              </p>
            </div>
          </div>
        </section>
      </div>

      <section className="bg-navy-950 border-t border-gray-800 py-20 px-6 text-center">
        <div className="max-w-2xl mx-auto space-y-6">
          <h2 className="text-2xl font-bold text-white">How I Built This</h2>
          <p className="text-gray-400">
            A deep-dive into the dataset, architecture decisions backed by EDA,
            training strategy, explainability, and deployment — written for
            engineers and researchers.
          </p>
          <a
            href="/blog"
            className="inline-flex items-center gap-2 px-6 py-3 rounded-lg bg-blue-600 hover:bg-blue-500 text-white font-medium transition-colors"
          >
            Read the Technical Writeup →
          </a>
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
