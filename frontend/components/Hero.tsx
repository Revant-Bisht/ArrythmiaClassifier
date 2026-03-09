"use client";

import { useEffect } from "react";
import { motion, useAnimation, type AnimationControls } from "framer-motion";

const VW = 330;
const VH = 90;
const N_BARS = 42;

// Simplified Lead II ECG shape — two QRS complexes
const ECG_PATH =
  "M0,45 L30,45 C36,44 40,38 47,45 L64,45 L68,56 L71,8 L75,45 L86,45 C96,32 111,32 121,45 L165,45 L195,45 C201,44 205,38 212,45 L229,45 L233,56 L236,8 L240,45 L251,45 C261,32 276,32 286,45 L330,45";

// Activation shapes that mimic what an inception conv responds to in an MI ECG
function syntheticBars(seed: number) {
  return Array.from({ length: N_BARS }, (_, i) => {
    const x = i / N_BARS;
    return Math.min(100, Math.max(6,
      Math.exp(-((x - 0.19) ** 2) / 0.006) * 90 +
      Math.exp(-((x - 0.71) ** 2) / 0.006) * 87 +
      Math.exp(-((x - 0.28) ** 2) / 0.018) * 38 +
      Math.exp(-((x - 0.80) ** 2) / 0.018) * 36 +
      Math.sin(x * 16 + seed) * 6 + 12,
    ));
  });
}

const ROWS = [
  { label: "k = 40", hint: "long-range", bars: syntheticBars(0.8), color: "#3b82f6" },
  { label: "k = 20", hint: "mid-range",  bars: syntheticBars(2.1), color: "#06b6d4" },
  { label: "k = 10", hint: "local",      bars: syntheticBars(3.9), color: "#8b5cf6" },
];

const ATTN = [
  { x: 55,  w: 30, alpha: 0.55 },
  { x: 220, w: 30, alpha: 0.50 },
  { x: 93,  w: 18, alpha: 0.22 },
  { x: 258, w: 18, alpha: 0.20 },
];

const CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"];
const CLR: Record<string, string> = {
  NORM: "#3b82f6", MI: "#ef4444", STTC: "#f97316", CD: "#22c55e", HYP: "#a855f7",
};
const PROBS: Record<string, number> = {
  NORM: 0.004, MI: 0.996, STTC: 0.013, CD: 0.012, HYP: 0.015,
};

// ~30s cycle: ~5s animation + 20s hold + ~5s gap
async function runLoop(c: AnimationControls) {
  await c.start("reset");
  await new Promise((r) => setTimeout(r, 60));
  await c.start("ecgIn");
  await c.start("ecgDraw");      // 1.5 s
  await c.start("convolution");  // 2.2 s — kernel sweeps + feature maps reveal in sync
  await c.start("attnIn");       // 0.5 s
  await c.start("outputIn");     // 0.8 s
  await new Promise((r) => setTimeout(r, 20000));
  await c.start("fadeAll");
  await new Promise((r) => setTimeout(r, 3500));
}

export function Hero() {
  const c = useAnimation();

  useEffect(() => {
    let alive = true;
    (async () => { while (alive) await runLoop(c); })();
    return () => { alive = false; };
  }, [c]);

  return (
    <section className="relative min-h-screen flex flex-col items-center justify-center bg-navy-900 overflow-hidden px-6">
      <div className="absolute inset-0 pointer-events-none">
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_80%_50%_at_50%_-20%,rgba(59,130,246,0.12),transparent)]" />
      </div>

      <div className="relative z-10 flex flex-col items-center gap-8 w-full max-w-4xl">

        {/* Title — static, outside animation loop */}
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7 }}
          className="text-center space-y-4"
        >
          <p className="text-blue-400 text-sm font-mono tracking-widest uppercase">
            Deep Learning · ECG Analysis · Explainable AI
          </p>
          <h1 className="text-4xl md:text-6xl font-bold text-white leading-tight tracking-tight">
            Detecting Arrhythmia
            <br />
            <span className="text-blue-400">with Deep Learning</span>
          </h1>
          <p className="text-gray-400 text-lg max-w-xl mx-auto">
            Correctly classifies cardiac conditions in{" "}
            <span className="text-white font-semibold">9 out of 10 ECGs</span>
            <span className="text-gray-500 text-sm ml-1">(0.928 macro AUC-ROC)</span>
          </p>
        </motion.div>

        {/* Card — always visible */}
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, delay: 0.2 }}
          className="w-full bg-navy-950 rounded-2xl border border-gray-800 p-5 shadow-2xl"
        >
          <p className="text-xs text-gray-500 font-mono mb-3">
            InceptionTime + Temporal Attention · Lead II · Forward Pass
          </p>

          {/* Input signal */}
          <div className="mb-1">
            <p className="text-xs text-gray-600 font-mono mb-1">input signal</p>
            <div className="relative h-[72px] w-full overflow-hidden">
              <motion.div
                className="absolute inset-y-1 rounded pointer-events-none z-10"
                style={{ width: "7%", border: "1px solid rgba(59,130,246,0.65)", backgroundColor: "rgba(59,130,246,0.07)" }}
                initial={{ opacity: 0 }}
                animate={c}
                variants={{
                  reset:       { left: "-8%", opacity: 0, transition: { duration: 0 } },
                  convolution: {
                    left: ["0%", "93%"],
                    opacity: [0, 1, 1, 0],
                    transition: { duration: 2.2, ease: "linear", opacity: { duration: 2.2, times: [0, 0.04, 0.88, 1] } },
                  },
                  fadeAll: { opacity: 0, transition: { duration: 0 } },
                }}
              >
                <div className="absolute left-0 inset-y-0 w-px" style={{ backgroundColor: "rgba(59,130,246,0.9)" }} />
              </motion.div>

              <motion.svg
                viewBox={`0 0 ${VW} ${VH}`}
                className="w-full h-full absolute inset-0"
                initial={{ opacity: 0 }}
                animate={c}
                variants={{
                  reset:   { opacity: 0, transition: { duration: 0 } },
                  ecgIn:   { opacity: 1, transition: { duration: 0 } },
                  fadeAll: { opacity: 0, transition: { duration: 0.4 } },
                }}
              >
                <motion.path
                  d={ECG_PATH} stroke="rgba(96,165,250,0.12)" strokeWidth={7} fill="none" strokeLinecap="round"
                  animate={c}
                  variants={{
                    reset:   { pathLength: 0, transition: { duration: 0 } },
                    ecgDraw: { pathLength: 1, transition: { duration: 1.5, ease: "easeInOut" } },
                  }}
                />
                <motion.path
                  d={ECG_PATH} stroke="#60a5fa" strokeWidth={1.8} fill="none" strokeLinecap="round"
                  animate={c}
                  variants={{
                    reset:   { pathLength: 0, transition: { duration: 0 } },
                    ecgDraw: { pathLength: 1, transition: { duration: 1.5, ease: "easeInOut" } },
                  }}
                />
                {ATTN.map((a, i) => (
                  <motion.rect
                    key={i} x={a.x} y={0} width={a.w} height={VH}
                    fill={`rgba(139,92,246,${a.alpha})`} rx={2}
                    initial={{ opacity: 0 }}
                    animate={c}
                    variants={{
                      reset:   { opacity: 0, transition: { duration: 0 } },
                      attnIn:  { opacity: 1, transition: { duration: 0.35, delay: i * 0.07 } },
                      fadeAll: { opacity: 0, transition: { duration: 0.3 } },
                    }}
                  />
                ))}
              </motion.svg>
            </div>
          </div>

          {/* Inception feature maps */}
          <div className="flex items-center gap-2 mt-3 mb-2">
            <div className="h-px flex-1 bg-gradient-to-r from-transparent via-gray-800 to-gray-700" />
            <span className="text-gray-600 text-xs font-mono">inception · multi-scale conv → feature maps</span>
            <div className="h-px flex-1 bg-gradient-to-r from-gray-700 via-gray-800 to-transparent" />
          </div>

          <div className="space-y-2 mb-3">
            {ROWS.map((row, ri) => (
              <motion.div
                key={row.label}
                className="flex items-center gap-2"
                initial={{ opacity: 0 }}
                animate={c}
                variants={{
                  reset:       { opacity: 0, transition: { duration: 0 } },
                  convolution: { opacity: 1, transition: { duration: 0.1, delay: ri * 0.05 } },
                  fadeAll:     { opacity: 0, transition: { duration: 0.4 } },
                }}
              >
                <span className="text-xs font-mono w-11 shrink-0 tabular-nums" style={{ color: row.color }}>{row.label}</span>
                <span className="text-xs text-gray-700 font-mono w-16 shrink-0">{row.hint}</span>
                <div className="flex-1 relative h-7">
                  <div className="absolute inset-0 flex items-end gap-px opacity-[0.18]">
                    {row.bars.map((h, bi) => (
                      <div key={bi} className="flex-1" style={{ height: `${h}%`, backgroundColor: row.color }} />
                    ))}
                  </div>
                  <motion.div
                    className="absolute inset-0 flex items-end gap-px"
                    initial={{ clipPath: "inset(0% 100% 0% 0%)", opacity: 1 }}
                    animate={c}
                    variants={{
                      reset:       { clipPath: "inset(0% 100% 0% 0%)", opacity: 1, transition: { duration: 0 } },
                      convolution: { clipPath: "inset(0% 0% 0% 0%)",   opacity: 1, transition: { duration: 2.2, ease: "linear", delay: ri * 0.05 } },
                      fadeAll:     { opacity: 0, transition: { duration: 0.4 } },
                    }}
                  >
                    {row.bars.map((h, bi) => (
                      <div key={bi} className="flex-1" style={{ height: `${h}%`, backgroundColor: row.color }} />
                    ))}
                  </motion.div>
                </div>
              </motion.div>
            ))}
          </div>

          {/* Output */}
          <div className="flex items-center gap-2 mb-2">
            <div className="h-px flex-1 bg-gradient-to-r from-transparent via-gray-800 to-gray-700" />
            <span className="text-gray-600 text-xs font-mono">temporal attention → GAP → classifier</span>
            <div className="h-px flex-1 bg-gradient-to-r from-gray-700 via-gray-800 to-transparent" />
          </div>

          <motion.div
            className="rounded-xl border border-red-900/50 bg-red-950/20 p-3"
            initial={{ opacity: 0 }}
            animate={c}
            variants={{
              reset:    { opacity: 0, y: 6, transition: { duration: 0 } },
              outputIn: { opacity: 1, y: 0, transition: { duration: 0.4 } },
              fadeAll:  { opacity: 0, transition: { duration: 0.4 } },
            }}
          >
            <div className="flex items-center justify-between mb-2">
              <div>
                <p className="text-xs text-gray-500 font-mono">Prediction</p>
                <p className="text-white font-semibold text-sm">Signs of Myocardial Infarction</p>
              </div>
              <div className="text-right">
                <p className="text-xl font-bold text-red-400">99.6%</p>
                <p className="text-xs text-gray-500">confidence</p>
              </div>
            </div>
            <div className="flex gap-2">
              {CLASSES.map((cls) => (
                <div key={cls} className="flex-1">
                  <div className="h-1.5 rounded-full bg-gray-800 overflow-hidden">
                    <motion.div
                      className="h-full rounded-full"
                      style={{ backgroundColor: CLR[cls] }}
                      animate={c}
                      variants={{
                        reset:    { width: "0%", transition: { duration: 0 } },
                        outputIn: { width: `${(PROBS[cls] ?? 0) * 100}%`, transition: { duration: 0.6, ease: "easeOut" } },
                      }}
                    />
                  </div>
                  <p className="text-xs text-gray-500 mt-1 text-center font-mono">{cls}</p>
                </div>
              ))}
            </div>
          </motion.div>
        </motion.div>

        <motion.a
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.6, duration: 0.5 }}
          href="#demo"
          className="group flex items-center gap-2 text-blue-400 hover:text-blue-300 transition-colors font-medium"
        >
          Try it yourself
          <span className="group-hover:translate-y-0.5 transition-transform">↓</span>
        </motion.a>
      </div>
    </section>
  );
}
