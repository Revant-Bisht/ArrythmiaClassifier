"use client";

import { motion } from "framer-motion";
import { CLASS_COLORS, CLASS_FULL } from "@/lib/types";

interface Props {
  probs: Record<string, number>;
  predictedClass: string;
}

const CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"];

export function ProbabilityBars({ probs, predictedClass }: Props) {
  return (
    <div className="space-y-2.5">
      <p className="text-xs text-gray-500 font-mono uppercase tracking-wider mb-3">
        Class Probabilities
      </p>
      {CLASSES.map((cls) => {
        const p = probs[cls] ?? 0;
        const isTop = cls === predictedClass;
        return (
          <div key={cls} className="flex items-center gap-3">
            <span
              className="text-xs font-mono w-10 text-right shrink-0"
              style={{ color: isTop ? CLASS_COLORS[cls] : "#6b7280" }}
            >
              {cls}
            </span>
            <div className="flex-1 h-2 bg-gray-800 rounded-full overflow-hidden">
              <motion.div
                className="h-full rounded-full"
                initial={{ width: 0 }}
                animate={{ width: `${p * 100}%` }}
                transition={{ duration: 0.5, ease: "easeOut" }}
                style={{
                  backgroundColor: isTop ? CLASS_COLORS[cls] : "#374151",
                }}
              />
            </div>
            <span
              className="text-xs font-mono w-10 shrink-0"
              style={{ color: isTop ? CLASS_COLORS[cls] : "#6b7280" }}
            >
              {(p * 100).toFixed(1)}%
            </span>
          </div>
        );
      })}
    </div>
  );
}
