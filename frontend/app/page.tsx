import { Hero } from "@/components/Hero";
import { DemoSection } from "@/components/DemoSection";

export default function Home() {
  return (
    <main className="bg-navy-900 min-h-screen">
      <nav className="fixed top-0 left-0 right-0 z-50 flex items-center justify-between px-6 py-4 bg-navy-900/80 backdrop-blur-md border-b border-gray-800">
        <div className="flex items-center gap-3">
          <span className="text-white font-semibold">Revant Bisht</span>
          <span className="text-gray-600">·</span>
          <span className="text-gray-400 text-sm">Arrhythmia Classifier</span>
        </div>
        <div className="flex items-center gap-5 text-sm">
          <a
            href="#demo"
            className="text-gray-400 hover:text-white transition-colors"
          >
            Demo
          </a>
          <a
            href="/blog"
            className="text-gray-400 hover:text-white transition-colors"
          >
            How I Built It
          </a>
          <a
            href="https://github.com/Revant-Bisht/ArrythmiaClassifier"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-1.5 text-gray-400 hover:text-white transition-colors"
          >
            <svg className="w-4 h-4" viewBox="0 0 24 24" fill="currentColor">
              <path d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z" />
            </svg>
            GitHub
          </a>
        </div>
      </nav>

      <div className="pt-16">
        <Hero />
        <DemoSection />
      </div>

      <section className="bg-navy-950 border-t border-gray-800 py-20 px-6 text-center">
        <div className="max-w-2xl mx-auto space-y-6">
          <h2 className="text-2xl font-bold text-white">How I Built This</h2>
          <p className="text-gray-400">
            A deep-dive into the dataset, architecture decisions backed by EDA,
            training strategy, explainability, and deployment — written for
            engineers and researchers.
          </p>
          <a
            href="/blog"
            className="inline-flex items-center gap-2 px-6 py-3 rounded-lg bg-blue-600 hover:bg-blue-500 text-white font-medium transition-colors"
          >
            Read the Technical Writeup →
          </a>
        </div>
      </section>

      <footer className="bg-navy-950 border-t border-gray-800 py-8 px-6">
        <div className="max-w-5xl mx-auto flex flex-wrap items-center justify-between gap-4 text-sm text-gray-600">
          <span>© 2026 Revant Bisht</span>
          <span>
            PTB-XL dataset · Wagner et al. 2020 · Strodthoff et al. 2020
          </span>
          <span className="text-xs">
            For research purposes only — not a clinical tool
          </span>
        </div>
      </footer>
    </main>
  );
}
