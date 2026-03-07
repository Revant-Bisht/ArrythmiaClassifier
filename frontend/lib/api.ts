import type { PredictResponse, SampleMeta } from "./types";

const API_BASE =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8081";

export async function getSamples(): Promise<SampleMeta[]> {
  const res = await fetch(`${API_BASE}/samples`, { cache: "no-store" });
  if (!res.ok) throw new Error("Failed to fetch samples");
  return res.json();
}

export async function getPreloaded(className: string): Promise<PredictResponse> {
  const res = await fetch(`${API_BASE}/predict/preloaded/${className}`, {
    cache: "no-store",
  });
  if (!res.ok) throw new Error(`Failed to fetch sample for ${className}`);
  return res.json();
}
