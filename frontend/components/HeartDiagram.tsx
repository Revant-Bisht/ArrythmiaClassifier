"use client";

import { motion } from "framer-motion";

const C = 1.2; // heartbeat cycle in seconds

// Conduction timing (fractions of C)
const T_SA  = 0.00;
const T_AV  = 0.08;
const T_BOH = 0.13;
const T_BR  = 0.17;
const PW    = 0.07;

// ECG wave timing
const P_S  = 0.01, P_E  = 0.09;
const QR_S = 0.17, QR_E = 0.26;
const TW_S = 0.31, TW_E = 0.48;

// Phase colours
const COL_P   = "rgb(167,139,250)"; // purple  — atrial depolarisation
const COL_QRS = "rgb(96,165,250)";  // blue    — ventricular depolarisation
const COL_T   = "rgb(52,211,153)";  // emerald — repolarisation

function Ripple({ cx, cy, f, color = COL_QRS }: {
  cx: number; cy: number; f: number; color?: string;
}) {
  return (
    <motion.circle
      cx={cx} cy={cy} r={3}
      fill="none"
      stroke={color}
      strokeWidth={1.5}
      animate={{ r: [3, 3, 16, 16], opacity: [0, 0.85, 0, 0] }}
      transition={{
        duration: C, repeat: Infinity,
        times: [0, f, Math.min(f + PW, 0.99), 1],
        ease: "easeOut",
      }}
    />
  );
}

function Node({ cx, cy, f, color = COL_QRS, label, lx, ly }: {
  cx: number; cy: number; f: number; color?: string;
  label?: string; lx?: number; ly?: number;
}) {
  return (
    <g>
      <Ripple cx={cx} cy={cy} f={f} color={color} />
      <circle cx={cx} cy={cy} r={3} fill={color} opacity={0.7} />
      <circle cx={cx} cy={cy} r={1.5} fill="white" />
      {label && (
        <text x={lx ?? cx + 5} y={ly ?? cy + 1} fontSize={7.5}
          fill={color} fontFamily="ui-monospace,monospace"
          dominantBaseline="middle">{label}</text>
      )}
    </g>
  );
}

function TravelDot({ x1, y1, x2, y2, tStart, tEnd, color = COL_QRS }: {
  x1: number; y1: number; x2: number; y2: number;
  tStart: number; tEnd: number; color?: string;
}) {
  return (
    <motion.circle r={2.5} fill={color}
      animate={{
        cx: [x1, x1, x2, x2],
        cy: [y1, y1, y2, y2],
        opacity: [0, 0.9, 0.9, 0],
      }}
      transition={{
        duration: C, repeat: Infinity,
        times: [0, tStart, tEnd, Math.min(tEnd + 0.03, 1)],
        ease: "easeInOut",
      }}
    />
  );
}

function Wave({ d, tStart, tEnd, color }: {
  d: string; tStart: number; tEnd: number; color: string;
}) {
  return (
    <motion.path d={d} stroke={color} strokeWidth={1.8} fill="none"
      strokeLinecap="round"
      initial={{ pathLength: 0 }}
      animate={{ pathLength: [0, 0, 1, 1] }}
      transition={{
        duration: C, repeat: Infinity,
        times: [0, tStart, tEnd, 1],
        ease: "easeInOut",
      }}
    />
  );
}

function WaveLabel({ x, y, label, showAt, color }: {
  x: number; y: number; label: string; showAt: number; color: string;
}) {
  return (
    <motion.text x={x} y={y} fontSize={8} fill={color} textAnchor="middle"
      fontFamily="ui-monospace,monospace"
      initial={{ opacity: 0 }}
      animate={{ opacity: [0, 0, 1, 1, 0] }}
      transition={{
        duration: C, repeat: Infinity,
        times: [0, showAt, Math.min(showAt + 0.04, 0.93), 0.91, 1],
      }}
    >
      {label}
    </motion.text>
  );
}

export function HeartDiagram() {
  // Heart  viewBox: "0 0 166 142"
  //   SA(122,22)  AV(77,57)  BoH(77,90)  LBB(32,124)  RBB(132,124)
  //   Atria:     LA(5,5,65,48)   RA(90,5,65,48)
  //   Ventricles: LV(5,63,65,64) RV(90,63,65,64)
  //
  // PQRST viewBox: "0 0 200 82"  baseline y=60
  //   P: bump  14→38  peak y=43
  //   QRS: complex 52→66  R-peak y=15  S y=68
  //   T:  82→118  peak y=34

  return (
    <div className="not-prose flex flex-col sm:flex-row items-center gap-5 mt-4">

      {/* ── Heart cross-section ── */}
      <svg viewBox="0 0 166 142" className="w-48 shrink-0 h-auto"
        aria-label="Cardiac conduction system">

        {/* Atria — static */}
        <rect x="5"  y="5" width="65" height="48" rx="7"
          fill="rgb(17,24,39)" stroke="rgb(55,65,81)" strokeWidth="1" />
        <rect x="90" y="5" width="65" height="48" rx="7"
          fill="rgb(17,24,39)" stroke="rgb(55,65,81)" strokeWidth="1" />

        {/* Atrial depolarisation flash — purple (P-wave phase) */}
        <motion.rect x="90" y="5" width="65" height="48" rx="7"
          fill={COL_P}
          animate={{ opacity: [0, 0, 0.14, 0] }}
          transition={{ duration: C, repeat: Infinity,
            times: [0, T_SA + 0.01, T_SA + 0.09, T_SA + 0.20] }} />
        <motion.rect x="5" y="5" width="65" height="48" rx="7"
          fill={COL_P}
          animate={{ opacity: [0, 0, 0.10, 0] }}
          transition={{ duration: C, repeat: Infinity,
            times: [0, T_SA + 0.03, T_AV + 0.04, T_AV + 0.15] }} />

        {/* Ventricles — static border */}
        <rect x="5"  y="63" width="65" height="64" rx="7"
          fill="rgb(17,24,39)" stroke="rgb(55,65,81)" strokeWidth="1" />
        <rect x="90" y="63" width="65" height="64" rx="7"
          fill="rgb(17,24,39)" stroke="rgb(55,65,81)" strokeWidth="1" />

        {/* Ventricular depolarisation flash — blue (QRS phase) */}
        <motion.rect x="5" y="63" width="65" height="64" rx="7"
          fill={COL_QRS}
          animate={{ opacity: [0, 0, 0.16, 0] }}
          transition={{ duration: C, repeat: Infinity,
            times: [0, T_BR + 0.04, T_BR + 0.11, T_BR + 0.24] }} />
        <motion.rect x="90" y="63" width="65" height="64" rx="7"
          fill={COL_QRS}
          animate={{ opacity: [0, 0, 0.13, 0] }}
          transition={{ duration: C, repeat: Infinity,
            times: [0, T_BR + 0.04, T_BR + 0.11, T_BR + 0.24] }} />

        {/* Septum */}
        <line x1="77.5" y1="5"   x2="77.5" y2="129" stroke="rgb(55,65,81)" strokeWidth="0.8" />
        <line x1="5"    y1="57"  x2="155"  y2="57"  stroke="rgb(55,65,81)" strokeWidth="0.8" />

        {/* Chamber labels */}
        {[
          { x: 37,  y: 31, t: "LA" },
          { x: 122, y: 31, t: "RA" },
          { x: 37,  y: 95, t: "LV" },
          { x: 122, y: 95, t: "RV" },
        ].map(({ x, y, t }) => (
          <text key={t} x={x} y={y} textAnchor="middle" fontSize={9}
            fill="rgb(75,85,99)" fontFamily="ui-monospace,monospace">{t}</text>
        ))}

        {/* Conduction pathways */}
        <line x1="122" y1="22" x2="77"  y2="57"  stroke="rgb(55,65,81)" strokeWidth="0.8" strokeDasharray="3,2" />
        <line x1="77"  y1="57" x2="77"  y2="90"  stroke="rgb(55,65,81)" strokeWidth="0.8" strokeDasharray="3,2" />
        <line x1="77"  y1="90" x2="32"  y2="124" stroke="rgb(55,65,81)" strokeWidth="0.8" strokeDasharray="3,2" />
        <line x1="77"  y1="90" x2="132" y2="124" stroke="rgb(55,65,81)" strokeWidth="0.8" strokeDasharray="3,2" />

        {/* Traveling pulse dots */}
        <TravelDot x1={122} y1={22} x2={77}  y2={57}  tStart={T_SA}  tEnd={T_AV}  color={COL_P}   />
        <TravelDot x1={77}  y1={57} x2={77}  y2={90}  tStart={T_AV}  tEnd={T_BOH} color={COL_QRS} />
        <TravelDot x1={77}  y1={90} x2={32}  y2={124} tStart={T_BOH} tEnd={T_BR}  color={COL_QRS} />
        <TravelDot x1={77}  y1={90} x2={132} y2={124} tStart={T_BOH} tEnd={T_BR}  color={COL_QRS} />

        {/* Conduction nodes */}
        <Node cx={122} cy={22}  f={T_SA}  color={COL_P}   label="SA"  lx={130} ly={16} />
        <Node cx={77}  cy={57}  f={T_AV}  color={COL_P}   label="AV"  lx={82}  ly={51} />
        <Node cx={77}  cy={90}  f={T_BOH} color={COL_QRS} />
        <Node cx={32}  cy={124} f={T_BR}  color={COL_QRS} label="LBB" lx={3}   ly={136} />
        <Node cx={132} cy={124} f={T_BR}  color={COL_QRS} label="RBB" lx={138} ly={136} />
      </svg>

      {/* ── PQRST waveform ── */}
      <div className="flex-1 min-w-0 w-full">
        <p className="text-[9px] text-gray-600 font-mono mb-1 tracking-widest uppercase">
          Lead II · 1 beat @ 100 Hz
        </p>
        <svg viewBox="0 0 200 82" className="w-full h-auto">
          {/* Grid */}
          <line x1="0" y1="60" x2="200" y2="60" stroke="rgb(37,47,63)" strokeWidth="0.6" />

          {/* Static flat baseline segments */}
          <path d="M0,60 L14,60 M38,60 L52,60 M66,60 L82,60 M118,60 L200,60"
            stroke="rgb(75,85,99)" strokeWidth="1.2" fill="none" />

          {/* P-wave — purple */}
          <Wave d="M14,60 C16,60 18,43 25,43 C32,43 34,60 38,60"
            tStart={P_S} tEnd={P_E} color={COL_P} />

          {/* QRS complex — blue */}
          <Wave d="M52,60 L55,68 L59,15 L63,70 L66,60"
            tStart={QR_S} tEnd={QR_E} color={COL_QRS} />

          {/* T-wave — emerald */}
          <Wave d="M82,60 C87,60 91,34 100,34 C109,34 113,60 118,60"
            tStart={TW_S} tEnd={TW_E} color={COL_T} />

          {/* Wave labels */}
          <WaveLabel x={25}  y={37} label="P"   showAt={P_E}  color={COL_P}   />
          <WaveLabel x={59}  y={11} label="QRS" showAt={QR_E} color={COL_QRS} />
          <WaveLabel x={100} y={28} label="T"   showAt={TW_E} color={COL_T}   />
        </svg>

        {/* Colour legend */}
        <div className="flex flex-wrap gap-x-4 gap-y-1 mt-1">
          {[
            { color: COL_P,   label: "Atrial depol." },
            { color: COL_QRS, label: "Ventricular depol." },
            { color: COL_T,   label: "Repolarisation" },
          ].map(({ color, label }) => (
            <span key={label} className="flex items-center gap-1">
              <span className="inline-block w-2 h-2 rounded-full" style={{ backgroundColor: color }} />
              <span className="text-[9px] text-gray-500 font-mono">{label}</span>
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}
