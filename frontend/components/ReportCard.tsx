"use client";

import { motion } from "framer-motion";
import type { Report } from "@/lib/types";
import { CLASS_COLORS } from "@/lib/types";

interface Props {
  report: Report;
  predictedClass: string;
}

export function ReportCard({ report, predictedClass }: Props) {
  const color = CLASS_COLORS[predictedClass] ?? "#3b82f6";

  return (
    <motion.div
      key={predictedClass}
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35 }}
      className="rounded-xl border bg-navy-950 p-5 space-y-4"
      style={{ borderColor: `${color}40` }}
    >
      <div className="flex items-start justify-between gap-4">
        <div>
          <p className="text-xs text-gray-500 font-mono uppercase tracking-wider mb-1">
            Clinical Report
          </p>
          <h3 className="text-white font-semibold text-base leading-snug">
            {report.headline}
          </h3>
        </div>
        <div
          className="shrink-0 text-2xl font-bold tabular-nums"
          style={{ color }}
        >
          {report.confidence_pct.toFixed(1)}%
        </div>
      </div>

      <p className="text-gray-300 text-sm leading-relaxed">{report.summary}</p>

      {report.flagged_regions.length > 0 && (
        <div className="space-y-1.5">
          <p className="text-xs text-gray-500 font-mono">Flagged regions</p>
          {report.flagged_regions.map((r) => (
            <div
              key={r.label}
              className="flex items-center gap-3 text-sm"
            >
              <span className="text-red-400">⚠</span>
              <span className="text-gray-200 font-medium">{r.label}</span>
              <span className="text-gray-500 font-mono text-xs">
                {r.start_s.toFixed(2)}s – {r.end_s.toFixed(2)}s
              </span>
            </div>
          ))}
        </div>
      )}

      <p className="text-xs text-gray-600 leading-relaxed border-t border-gray-800 pt-3">
        {report.disclaimer}
      </p>
    </motion.div>
  );
}
