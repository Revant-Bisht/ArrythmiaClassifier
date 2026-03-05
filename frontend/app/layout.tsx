import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Arrhythmia Classifier — Revant Bisht",
  description:
    "Detecting cardiac arrhythmia with InceptionTime + Temporal Attention on PTB-XL. 0.928 macro AUC-ROC on 5-class 12-lead ECG classification.",
  openGraph: {
    title: "Arrhythmia Classifier — Revant Bisht",
    description:
      "Deep learning ECG classifier with interactive Grad-CAM explainability demo.",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
