"use client";

import { CLASS_COLORS } from "@/lib/types";

interface Props {
  active: string;
  onChange: (cls: string) => void;
}

const CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"];

export function ClassTabs({ active, onChange }: Props) {
  return (
    <div className="flex items-center gap-2 flex-wrap">
      <span className="text-xs text-gray-500 mr-1">Grad-CAM for:</span>
      {CLASSES.map((cls) => {
        const isActive = cls === active;
        return (
          <button
            key={cls}
            onClick={() => onChange(cls)}
            className={[
              "px-3 py-1 rounded-md text-xs font-mono transition-all duration-150",
              isActive
                ? "text-white"
                : "text-gray-500 hover:text-gray-300 bg-transparent",
            ].join(" ")}
            style={
              isActive
                ? { backgroundColor: `${CLASS_COLORS[cls]}25`, color: CLASS_COLORS[cls] }
                : {}
            }
          >
            {cls}
          </button>
        );
      })}
    </div>
  );
}
