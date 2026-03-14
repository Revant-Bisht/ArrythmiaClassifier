"use client";

import { useState, useEffect, useCallback } from "react";
import { getPreloaded } from "@/lib/api";
import type { PredictResponse } from "@/lib/types";
import { CLASS_COLORS } from "@/lib/types";
import { SampleSelector } from "./SampleSelector";
import { ECGChart } from "./ECGChart";
import { ClassTabs } from "./ClassTabs";
import { ProbabilityBars } from "./ProbabilityBars";
import { ReportCard } from "./ReportCard";

const LEGEND = [
  { color: "rgba(220,38,38,0.5)", label: "Grad-CAM activation" },
  { color: "rgba(139,92,246,0.5)", label: "Temporal attention α" },
  { color: "#ef4444", label: "Flagged diagnostic region" },
];

export function DemoSection() {
  const [selectedClass, setSelectedClass] = useState("MI");
  const [activeCamClass, setActiveCamClass] = useState("MI");
  const [data, setData] = useState<PredictResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async (cls: string) => {
    setLoading(true);
    setError(null);
    try {
      const result = await getPreloaded(cls);
      setData(result);
      setActiveCamClass(cls);
    } catch {
      setError("Could not reach the API. Make sure the backend is running.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load("MI");
  }, [load]);

  const handleSelectClass = (cls: string) => {
    setSelectedClass(cls);
    load(cls);
  };

  const activeCam = data?.gradcam_per_class[activeCamClass] ?? data?.gradcam ?? [];
  const activeColor = CLASS_COLORS[data?.predicted_class ?? "MI"];

  return (
    <section id="demo" className="bg-navy-900 py-24 px-6">
      <div className="max-w-5xl mx-auto space-y-10">
        <div className="space-y-2">
          <p className="text-blue-400 text-sm font-mono tracking-widest uppercase">
            Interactive Demo
          </p>
          <h2 className="text-3xl font-bold text-white">Try the Model</h2>
          <p className="text-gray-400 max-w-2xl">
            Select one of the conditions below to see how the model reads a real
            ECG from the PTB-XL dataset. All responses are pre-cached — results
            appear instantly.
          </p>
        </div>

        <SampleSelector
          selected={selectedClass}
          onSelect={handleSelectClass}
          loading={loading}
        />

        {/* Chart legend explainer */}
        <div className="rounded-xl border border-gray-800 bg-gray-900/40 p-4 text-sm space-y-2">
          <p className="text-gray-400 font-medium mb-1">Reading the chart</p>
          <p className="text-gray-500">
            <span className="inline-flex items-center gap-1.5 mr-1">
              <span className="inline-block w-2.5 h-2.5 rounded-sm bg-red-500/50 shrink-0" />
              <span className="text-red-400 font-medium">Grad-CAM (red/orange)</span>
            </span>
            — gradient-weighted class activation map. It shows which moments in
            the ECG most strongly pushed the model toward its prediction. Think of
            it as the model&apos;s &quot;evidence&quot;: bright spots are the suspicious
            segments it noticed.
          </p>
          <p className="text-gray-500">
            <span className="inline-flex items-center gap-1.5 mr-1">
              <span className="inline-block w-2.5 h-2.5 rounded-sm bg-purple-500/50 shrink-0" />
              <span className="text-purple-400 font-medium">Temporal attention α (purple)</span>
            </span>
            — the model&apos;s learned focus weights across time. High attention at a
            given time step means the network is weighing that moment heavily when
            forming its diagnosis, independent of how &quot;loud&quot; the signal is there.
          </p>
        </div>

        {error && (
          <div className="rounded-lg border border-red-900 bg-red-950/30 px-4 py-3 text-sm text-red-400">
            {error}
          </div>
        )}

        {data && (
          <div className="space-y-6">
            <div className="rounded-2xl border border-gray-800 bg-navy-950 overflow-hidden">
              <div className="flex items-center justify-between px-5 pt-4 pb-2 border-b border-gray-800 flex-wrap gap-3">
                <div className="flex items-center gap-4 text-xs text-gray-500 font-mono">
                  <span>Lead II · 10 s @ 100 Hz</span>
                  {LEGEND.map((l) => (
                    <span key={l.label} className="flex items-center gap-1.5">
                      <span
                        className="inline-block w-3 h-3 rounded-sm"
                        style={{ backgroundColor: l.color }}
                      />
                      {l.label}
                    </span>
                  ))}
                </div>
                <ClassTabs
                  active={activeCamClass}
                  onChange={setActiveCamClass}
                />
              </div>

              <div
                className={[
                  "px-4 pb-2 transition-opacity duration-200",
                  loading ? "opacity-40" : "opacity-100",
                ].join(" ")}
              >
                <ECGChart
                  signal={data.signal_lead2}
                  gradcam={activeCam}
                  attention={data.attention}
                  flaggedRegions={
                    activeCamClass === data.predicted_class
                      ? data.report.flagged_regions
                      : []
                  }
                  color={activeColor}
                />
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="rounded-xl border border-gray-800 bg-navy-950 p-5">
                <ProbabilityBars
                  probs={data.probs}
                  predictedClass={data.predicted_class}
                />
              </div>
              <ReportCard
                report={data.report}
                predictedClass={data.predicted_class}
              />
            </div>
          </div>
        )}

        {loading && !data && (
          <div className="flex items-center justify-center h-48 text-gray-600">
            <div className="w-6 h-6 rounded-full border-2 border-blue-500 border-t-transparent animate-spin mr-3" />
            Loading sample…
          </div>
        )}
      </div>
    </section>
  );
}
