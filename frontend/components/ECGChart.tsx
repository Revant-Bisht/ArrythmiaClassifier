"use client";

import { useEffect, useRef } from "react";
import type { FlaggedRegion } from "@/lib/types";

interface Props {
  signal: number[];
  gradcam: number[];
  attention: number[];
  flaggedRegions: FlaggedRegion[];
  color: string;
  samplingRate?: number;
}

export function ECGChart({
  signal,
  gradcam,
  attention,
  flaggedRegions,
  color,
  samplingRate = 100,
}: Props) {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current || !signal.length) return;

    const loadD3 = async () => {
      const d3 = await import("d3");

      const margin = { top: 28, right: 64, bottom: 36, left: 52 };
      const totalH = svgRef.current!.clientHeight || 220;
      const totalW = svgRef.current!.clientWidth || 800;
      const W = totalW - margin.left - margin.right;
      const H = totalH - margin.top - margin.bottom;

      d3.select(svgRef.current).selectAll("*").remove();

      const root = d3
        .select(svgRef.current)
        .append("g")
        .attr("transform", `translate(${margin.left},${margin.top})`);

      const T = signal.length;
      const xScale = d3.scaleLinear().domain([0, T / samplingRate]).range([0, W]);
      const sigMin = d3.min(signal)! - 0.06;
      const sigMax = d3.max(signal)! + 0.06;
      const yScale = d3.scaleLinear().domain([sigMin, sigMax]).range([H, 0]);
      const attMax = d3.max(attention)! || 1;
      const attScale = d3.scaleLinear().domain([0, attMax]).range([H, H * 0.55]);

      const gridX = d3.axisBottom(xScale).ticks(10).tickSize(-H).tickFormat(() => "");
      const gridY = d3.axisLeft(yScale).ticks(6).tickSize(-W).tickFormat(() => "");

      root.append("g").attr("class", "ecg-grid").attr("transform", `translate(0,${H})`).call(gridX);
      root.append("g").attr("class", "ecg-grid").call(gridY);

      const BIN = 4;
      const bins: number[] = [];
      for (let i = 0; i < T; i += BIN) {
        bins.push(d3.max(gradcam.slice(i, i + BIN)) ?? 0);
      }

      root
        .selectAll(".cam")
        .data(bins)
        .join("rect")
        .attr("class", "cam")
        .attr("x", (_, i) => xScale((i * BIN) / samplingRate))
        .attr("y", 0)
        .attr("width", (W / bins.length) + 0.5)
        .attr("height", H)
        .attr("fill", (d) => `rgba(220,38,38,${d * 0.52})`);

      const attArea = d3
        .area<number>()
        .x((_, i) => xScale(i / samplingRate))
        .y0(H)
        .y1((d) => attScale(d))
        .curve(d3.curveBasis);

      root
        .append("path")
        .datum(attention)
        .attr("fill", "rgba(139,92,246,0.20)")
        .attr("stroke", "rgba(139,92,246,0.60)")
        .attr("stroke-width", 1)
        .attr("d", attArea);

      const line = d3
        .line<number>()
        .x((_, i) => xScale(i / samplingRate))
        .y((d) => yScale(d))
        .curve(d3.curveLinear);

      root
        .append("path")
        .datum(signal)
        .attr("fill", "none")
        .attr("stroke", color)
        .attr("stroke-width", 1.6)
        .attr("d", line);

      flaggedRegions.forEach((r) => {
        const x1 = xScale(r.start_s);
        const x2 = xScale(r.end_s);
        const bY = -10;
        const tick = 5;
        const g = root.append("g");
        g.append("line").attr("x1", x1).attr("y1", bY).attr("x2", x2).attr("y2", bY)
          .attr("stroke", "#ef4444").attr("stroke-width", 1.5);
        g.append("line").attr("x1", x1).attr("y1", bY).attr("x2", x1).attr("y2", bY + tick)
          .attr("stroke", "#ef4444").attr("stroke-width", 1.5);
        g.append("line").attr("x1", x2).attr("y1", bY).attr("x2", x2).attr("y2", bY + tick)
          .attr("stroke", "#ef4444").attr("stroke-width", 1.5);
        g.append("text")
          .attr("x", (x1 + x2) / 2).attr("y", bY - 3)
          .attr("text-anchor", "middle")
          .attr("fill", "#ef4444")
          .attr("font-size", "9px")
          .attr("font-family", "JetBrains Mono, monospace")
          .text(r.label);
      });

      root
        .append("g")
        .attr("class", "axis-label")
        .attr("transform", `translate(0,${H})`)
        .call(d3.axisBottom(xScale).ticks(10).tickFormat((d) => `${d}s`));

      root
        .append("g")
        .attr("class", "axis-label")
        .call(d3.axisLeft(yScale).ticks(5).tickFormat((d) => `${(+d).toFixed(1)}`));

      root
        .append("text")
        .attr("transform", "rotate(-90)")
        .attr("y", -40).attr("x", -(H / 2))
        .attr("text-anchor", "middle")
        .attr("fill", "#6b7280").attr("font-size", "10px")
        .text("mV");

      const attAxisRight = d3
        .axisRight(d3.scaleLinear().domain([0, attMax]).range([H, H * 0.55]))
        .ticks(3)
        .tickFormat((d) => `${(+d).toFixed(3)}`);

      root
        .append("g")
        .attr("class", "axis-label")
        .attr("transform", `translate(${W},0)`)
        .call(attAxisRight)
        .selectAll("text")
        .style("fill", "#8b5cf6")
        .style("font-size", "9px");

      root
        .append("text")
        .attr("x", W + 50).attr("y", H * 0.77)
        .attr("transform", `rotate(-90,${W + 50},${H * 0.77})`)
        .attr("text-anchor", "middle")
        .attr("fill", "#8b5cf6").attr("font-size", "9px")
        .text("Attn α");
    };

    loadD3();
  }, [signal, gradcam, attention, flaggedRegions, color, samplingRate]);

  return (
    <svg
      ref={svgRef}
      className="w-full"
      style={{ height: 220 }}
    />
  );
}
