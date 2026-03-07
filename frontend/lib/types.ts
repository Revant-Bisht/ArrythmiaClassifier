export interface FlaggedRegion {
  label: string;
  start_s: number;
  end_s: number;
  peak_s: number;
}

export interface Report {
  headline: string;
  summary: string;
  confidence_pct: number;
  flagged_regions: FlaggedRegion[];
  disclaimer: string;
}

export interface PredictResponse {
  sample_id: string;
  predicted_class: string;
  confidence: number;
  probs: Record<string, number>;
  attention: number[];
  gradcam: number[];
  gradcam_per_class: Record<string, number[]>;
  signal_lead2: number[];
  report: Report;
}

export interface SampleMeta {
  id: string;
  class_name: string;
  class_full: string;
  confidence: number;
}

export const CLASS_NAMES = ["NORM", "MI", "STTC", "CD", "HYP"] as const;

export const CLASS_FULL: Record<string, string> = {
  NORM: "Normal",
  MI: "Myocardial Infarction",
  STTC: "ST/T-wave Change",
  CD: "Conduction Disturbance",
  HYP: "Hypertrophy",
};

export const CLASS_COLORS: Record<string, string> = {
  NORM: "#3b82f6",
  MI: "#ef4444",
  STTC: "#f97316",
  CD: "#22c55e",
  HYP: "#a855f7",
};
