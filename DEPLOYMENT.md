# Deployment Notes

> How the backend and frontend are hosted. Wrote this down so I don't have to remember it.

---

## Architecture

```
Browser → Vercel (Next.js frontend)
              ↓
        Fly.io (FastAPI + ONNX backend)
```

The frontend is a static Next.js site deployed on Vercel's CDN — zero cold starts, serves instantly from edge nodes.
The backend is a FastAPI app running ONNX Runtime on a single always-on Fly.io machine in San Jose (`sjc`). No PyTorch in the Docker image — just ONNX Runtime + the 1MB model file + 5 pre-cached JSON responses loaded at startup.

Cost: **$0/month** — both sit within free tiers (Vercel Hobby, Fly.io 256MB shared CPU allowance).

---

## Backend — Fly.io

**App:** `arrhythmia-api`
**URL:** `https://arrhythmia-api.fly.dev`
**Region:** `sjc` (San Jose)
**Machine:** 1× shared-cpu-1x, 256MB RAM, always-on (`min_machines_running = 1`, `auto_stop_machines = false`)

### First deploy

```bash
brew install flyctl
flyctl auth login
flyctl apps create arrhythmia-api
flyctl deploy
flyctl scale count 1   # Fly creates 2 by default — scale back to 1
```

### Redeploy after changes

```bash
flyctl deploy
```

### Useful commands

```bash
flyctl status                    # machine health
flyctl logs                      # live logs
flyctl scale count 1             # ensure only 1 machine running
curl https://arrhythmia-api.fly.dev/health   # quick sanity check
```

### Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | `{"status":"ok","cached_samples":5}` |
| `GET` | `/samples` | List of 5 pre-cached samples with class + confidence |
| `GET` | `/predict/preloaded/{class}` | Full cached response — signal, Grad-CAM, attention, report |
| `POST` | `/predict/upload` | Raw 12×1000 array → probs + attention + report |
| `GET` | `/docs` | FastAPI auto-generated interactive docs |

The five cached samples are loaded into memory at startup from `backend/cache/sample_{CLASS}.json`.
Pre-computing them locally (via `scripts/cache_samples.py`) means the production image needs no PyTorch dependency.

### Docker image

```
python:3.11-slim
├── onnxruntime
├── fastapi + uvicorn
├── src/           (arrhythmia package — data loaders, schemas)
├── backend/       (app, inference, reports, cache JSONs)
└── checkpoints/model.onnx   (1.01 MB, opset 17)
```

Image size: ~138 MB.

---

## Frontend — Vercel

**Project:** `arrhythmia-ecg`
**URL:** `https://arrhythmia-ecg.vercel.app`
**Framework:** Next.js 14 (App Router)

### Environment variables

| Variable | Value |
|---|---|
| `NEXT_PUBLIC_API_URL` | `https://arrhythmia-api.fly.dev` |

Set in Vercel dashboard → Project → Settings → Environment Variables. Also in `frontend/.env.production` for local production builds.

### First deploy

```bash
cd frontend
npx vercel                              # follow prompts, set project name to arrhythmia-ecg
npx vercel env add NEXT_PUBLIC_API_URL production
# paste: https://arrhythmia-api.fly.dev
npx vercel --prod                       # promote to production
```

### Redeploy after changes

Vercel auto-deploys on every push to `main` if you connect the GitHub repo (recommended).
Or manually:

```bash
cd frontend
npx vercel --prod
```

### Connecting GitHub for auto-deploy (recommended)

1. Vercel dashboard → project → Settings → Git
2. Connect to `Revant-Bisht/ArrythmiaClassifier`
3. Set **Root Directory** to `frontend/`
4. Every push to `main` triggers a redeploy automatically

---

## Routes

| URL | Page |
|---|---|
| `/` | Landing — hero animation, model explainer, CTAs |
| `/demo` | Interactive ECG demo |
| `/blog` | Technical writeup |
| `/eda` | EDA deep-dive (mirrors `01_eda.ipynb`) |
| `/architecture` | Architecture deep-dive (mirrors `02_model.ipynb`) |
| `/results` | Evaluation deep-dive (mirrors `03_evaluation.ipynb`) |

---

## Things I'd change if this went to production

- Add a CDN layer in front of the Fly.io backend (responses are static per class — could be cached at edge)
- The `/predict/upload` endpoint currently returns no Grad-CAM because Grad-CAM needs PyTorch. A proper prod setup would have a separate job queue for that.
- Swap the 256MB machine for something slightly bigger if traffic picks up — latency on the preloaded responses is ~7ms but a cold wake (if the machine ever stopped) would be ~2s.
- Add proper rate limiting on the upload endpoint.
