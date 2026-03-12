"use client";

import { CLASS_COLORS, CLASS_FULL } from "@/lib/types";

interface Props {
  selected: string;
  onSelect: (cls: string) => void;
  loading: boolean;
}

const CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"];

export function SampleSelector({ selected, onSelect, loading }: Props) {
  return (
    <div className="flex flex-wrap gap-2">
      {CLASSES.map((cls) => {
        const active = selected === cls;
        return (
          <button
            key={cls}
            onClick={() => onSelect(cls)}
            disabled={loading}
            className={[
              "relative px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200",
              "border focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-navy-900",
              active
                ? "text-white shadow-lg scale-105"
                : "text-gray-400 border-gray-700 bg-gray-900 hover:border-gray-500 hover:text-gray-200",
              loading && "opacity-60 cursor-not-allowed",
            ].join(" ")}
            style={
              active
                ? {
                    backgroundColor: `${CLASS_COLORS[cls]}20`,
                    borderColor: CLASS_COLORS[cls],
                    boxShadow: `0 0 12px ${CLASS_COLORS[cls]}30`,
                  }
                : {}
            }
          >
            <span
              className="mr-1.5 inline-block w-2 h-2 rounded-full"
              style={{ backgroundColor: active ? CLASS_COLORS[cls] : "#6b7280" }}
            />
            <span className="font-mono text-xs">{cls}</span>
            <span className="ml-1.5 text-xs opacity-70">{CLASS_FULL[cls]}</span>
          </button>
        );
      })}
    </div>
  );
}
