# 🛰 OpenEnv · Orbital Command

> **Autonomous Constellation Orchestrator** — An OpenEnv hackathon RL environment where an LLM must
> schedule satellite operations across a mathematically-simulated Low‑Earth Orbit swarm.

---

## What Problem Does This Solve?

Companies like **Planet Labs**, **SpaceX Starlink**, and **ESA** operate swarms of LEO satellites
under relentless pressure:

| Real-World Constraint | How It's Simulated |
|---|---|
| Solar charging windows | Battery +4 %/step in sunlight, 0 in eclipse |
| Eclipse zone (half the orbit) | Positions 170°–350° have zero solar input |
| Ground station passes | Line-of-sight window of ±15° at each station |
| Storage overflow | Data is **permanently destroyed** if drives fill |
| Fuel budgets | ISL manoeuvres and station-keeping consume ΔV |
| Thermal limits | Active components heat up; must vent or sleep |
| Stochastic disruptions | Solar flares, blizzards, congestion, atmospheric drag |

The LLM receives a **structured JSON observation** every step and must issue one **typed action**.
Poor look-ahead reasoning (greedy choices) kills satellites and destroys data.

---

## ⚡ Quick Start

```bash
# 1. Clone / navigate to project
cd orbital-command

# 2. Install
pip install -r requirements.txt

# 3. Launch dashboard (no API key needed)
python -m ui.app
# → http://127.0.0.1:7860

# 4. (Optional) run a full headless episode
python scripts/run_episode.py --task 1 --steps 120 --render

# 5. (Optional) benchmark all tasks
python scripts/benchmark.py
```

---

## 🔐 API Keys — Three Ways

> **None required** to run the app — the built-in `RuleBasedAgent` works without any key.

### Option A — Type in the UI (recommended for quick testing)

Open the **🔐 API Keys & LLM Configuration** accordion at the top of the dashboard.  
Enter your key. Press **💾 Save Keys to Session**.  
Keys are stored only in the current Python process — **never written to disk from the UI**.

### Option B — `.env` file (recommended for repeated use)

```bash
cp .env.example .env
# Edit .env with your favourite text editor
```

The file is auto-loaded by both the UI and the CLI scripts on startup.

```bash
```bash
# .env — fill in what you need:

GOOGLE_API_KEY=AIza...         # Google Gemini keys start with AIza
HF_TOKEN=hf_...                # HuggingFace tokens start with hf_
```

### Option C — OS environment variables

```bash
# Linux / macOS
export GOOGLE_API_KEY="AIza..."
python -m ui.app

# Windows PowerShell
$env:GOOGLE_API_KEY = "AIza..."
python -m ui.app

# Windows CMD
set GOOGLE_API_KEY=AIza...
python -m ui.app
```

### Where to Get Keys

| Provider | URL | Free Tier? |
|----------|-----|-----------|
| **Google Gemini** | [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey) | **Free** — recommended for hackathon |
| **HuggingFace** | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) | Free with rate limits |

---

## 🎮 Five Tasks (Escalating Difficulty)

```
Task 1 ──────────────────────────────── Easy
  ☀️  Eclipse Survival
  • 1 satellite: Sat-Alpha
  • Target at 200° (inside eclipse zone, no solar charging)
  • Must sleep → charge → wake → capture → survive
  • Grade: binary 1.0 / 0.0 (image captured & sat alive)

Task 2 ──────────────────────────────── Medium
  💾  Storage Bottleneck
  • 3 satellites with 80%+ full storage
  • Norway station OFFLINE (blizzard) — primary lost
  • Must hit a 4-step Antarctica window at 280°
  • Grade: continuous GB-downlinked ratio (overwrite = heavy penalty)

Task 3 ──────────────────────────────── Hard
  🔗  Laser Cross-Link Relay
  • 4 satellites; critical image trapped on Sat-1 at 215°
  • Pentagon (DC) station at 45° won't have direct LoS for 45 sim-minutes
  • Must route: Sat-1 → Sat-2 → Sat-3 → Sat-4 → DC
  • Sat-3 is at 35% battery — manage it or it dies mid-relay
  • Grade: speed × routing_quality − death_penalty

Task 4 ──────────────────────────────── Very Hard
  🌐  Global Swarm Harvest
  • 6 satellites, 8 imaging targets, 3 ground stations
  • Must assign, capture, and downlink all targets in one pass
  • ISL routing needed for satellites that miss their direct window
  • Grade: coverage_score + collaborative_bonus (all 8 = +500 pts)

Task 5 ──────────────────────────────── Expert
  🚨  Emergency Response Protocol
  • Earthquake at 155° — 3 EMERGENCY requests, 60-min deadline, 5× reward
  • Constellation state: one overheating, one near-dead battery, one sleeping
  • Second disaster (flooding at 310°) spawns at minute 30 — must reprioritise
  • Grade: emergency_rate × survival_rate × latency_score
```

---

## 🏆 Reward Function

$$R_t = \underbrace{D_t \cdot \alpha}_{\text{data revenue}} - \underbrace{O_t \cdot \beta}_{\text{overwrite penalty}} - \underbrace{B(b_t)}_{\text{battery}} - \underbrace{T(\tau_t)}_{\text{thermal}} - \underbrace{L(d_t)}_{\text{deadline latency}}$$

| Symbol | Value | Meaning |
|--------|-------|---------|
| α | **10** | $/GB for downlinked data |
| β | **20** | Penalty per GB overwritten |
| B(b) | 0 if b ≥ 50 %, linear → 2 at 20 %, **exponential** below 20 % | Battery damage regime |
| T(τ) | 0 if τ < 70°, linear → 3× critical | Thermal stress |
| L(d) | 2 pts/min overdue, capped at 50 | Deadline penalty |
| Death | **−500** + episode terminates | Battery = 0 % |

---

## 🛰 Pydantic Interface

### Observation (what the LLM receives)

```python
class Observation(BaseModel):
    current_orbit_minute: int
    step_number: int
    task_id: int
    satellites: List[SatelliteTelemetry]
    imaging_requests: List[ImagingRequest]     # target, reward, priority, deadline
    ground_stations: List[GroundStation]        # position, status, bandwidth
    active_events: List[StochasticEvent]        # solar flares, outages, drag…
    isl_topology: Dict[str, List[str]]          # adjacency map for relay routing
    episode_score: float
    reward_breakdown: Dict[str, float]          # last-step breakdown
```

### Action (what the LLM must return as JSON)

```json
{
  "action_type": "sleep_mode",
  "target_sat_id": "Sat-Alpha",
  "request_id": "REQ-1",
  "target_station": "Station_Norway",
  "relay_chain": ["Sat-1", "Sat-2", "Sat-4"],
  "reasoning": "Battery at 38% — entering eclipse in 4 steps. Sleep now to charge before reaching target at 200°."
}
```

### Action Resource Costs

| Action | Battery Δ | Fuel Δ | Storage Δ |
|--------|----------|--------|-----------|
| `capture_image` | **−8 %** | −0.2 % | **+15 %** |
| `downlink_data` | −5 % | — | clears pending |
| `sleep_mode` | +4 % (if sun) | — | — |
| `inter_satellite_link` | −10 % sender, −6 % relay | −0.5 % | ISL transfer |
| `station_keeping` | −12 % | **−8 %** | — |
| `emergency_transmit` | **−15 %** | — | partial burst |
| `thermal_vent` | — | — | −15° thermal |

---

## 🤖 LLM Backends

| Backend | `--backend` flag | Key env var | Notes |
|---------|-----------------|-------------|-------|
| Rule-Based | `rule_based` | — | Built-in heuristic, zero cost |
| Gemini | `gemini` | `GOOGLE_API_KEY` | Gemini 2.0 Flash / 1.5 Pro |
| HuggingFace | `huggingface` | `HF_TOKEN` | Llama 3, Mistral, Zephyr, Falcon |

---

## 📁 Project Structure

```
orbital-command/
│
├── env/                      Core simulation engine
│   ├── models.py             Pydantic v2 — Observation, Action, EpisodeResult
│   ├── physics.py            1-D orbital mechanics (eclipse, LoS, ISL BFS)
│   ├── events.py             Stochastic event engine (solar flares, outages…)
│   └── orbital_env.py        gymnasium.Env — reset / step / render
│
├── tasks/                    Task configurations
│   ├── task1_eclipse.py      ☀️  Easy
│   ├── task2_storage.py      💾  Medium
│   ├── task3_crosslink.py    🔗  Hard
│   ├── task4_swarm.py        🌐  Very Hard
│   └── task5_emergency.py    🚨  Expert
│
├── agent/
│   ├── llm_agent.py          Gemini / HF / RuleBased agents
│   └── prompt_builder.py     ReAct-style prompt with timing hints & LoS tables
│
├── scoring/
│   └── leaderboard.py        SQLite episode leaderboard (data/leaderboard.db)
│
├── ui/
│   ├── app.py                Gradio 6.x mission control dashboard
│   └── orbit_svg.py          SVG orbit renderer (satellites, ISL, eclipse, stars)
│
├── scripts/
│   ├── run_episode.py        CLI headless runner
│   └── benchmark.py          Runs all 5 tasks, prints comparison table
│
├── .env.example              ← Copy to .env and fill your keys
├── requirements.txt
├── Dockerfile                HuggingFace Spaces compatible (port 7860)
└── README.md
```

---

## 🐳 Docker / HuggingFace Space

```bash
# Build and run locally
docker build -t orbital-command .
docker run -p 7860:7860 \
  -e GOOGLE_API_KEY=AIza... \
  orbital-command

# Or pass a .env file
docker run -p 7860:7860 --env-file .env orbital-command
```

Deploy to **HuggingFace Spaces** by pushing this repo and adding your API keys as
[Spaces Secrets](https://huggingface.co/docs/hub/spaces-sdks-docker#secret-management)
(equivalent to environment variables — never exposed in the UI).

---

## 🧪 CLI Usage

```bash
# Run Task 1 with 120 steps (enough to complete eclipse cycle)
python scripts/run_episode.py --task 1 --steps 120 --render

# Run Task 3 with Gemini 2.0 Flash
python scripts/run_episode.py --task 3 --backend gemini --model gemini-2.0-flash --steps 120

# Benchmark all 5 tasks back-to-back
python scripts/benchmark.py

# Benchmark with HF Inference
python scripts/benchmark.py --backend huggingface --model meta-llama/Meta-Llama-3-8B-Instruct
```

---

## 🏅 Grading

| Grade | Normalised Score | Meaning |
|-------|-----------------|---------|
| **S** | ≥ 0.95 | Perfect / near-perfect execution |
| **A** | ≥ 0.80 | Strong planning, minor losses |
| **B** | ≥ 0.60 | Functional but suboptimal |
| **C** | ≥ 0.40 | Major missed opportunities |
| **F** | < 0.40 | Satellite death or critical failures |

Episode results are stored in `data/leaderboard.db` (SQLite) and shown in the
**🏆 Leaderboard** accordion in the UI.

---

## 💡 Why This Tests True LLM Intelligence

> Most LLMs are greedy — they optimise for the next step, often at the cost of the next 10.

This environment forces **multi-horizon reasoning** across four interdependent resources:

1. **Battery** — only charges during sunlight (≈ half the orbit). Dying kills the episode.
2. **Storage** — fills as images are captured. Must be downlinked before 100% overflow.
3. **Fuel** — consumed by station-keeping and ISL relay attitude burns.
4. **Thermal** — rises during active sunlit operations. Overheating damages hardware.

An optimal agent must:
- **Predict eclipse entry/exit** and pre-charge (temporal reasoning)
- **Resolve the charge–capture trade-off** non-greedily (delayed gratification)
- **Route ISL chains** without depleting relay satellites (multi-agent coordination)
- **Re-triage mid-episode** when new disaster requests arrive (adaptive planning)

---

## ⚙️ Environment Specs

| Property | Value |
|----------|-------|
| Observation type | Pydantic `Observation` (serialisable to JSON) |
| Action type | Pydantic `Action` (JSON-validated) |
| Orbit model | 1-D ring, 0–359°, 2°/step |
| Eclipse zone | 170°–350° |
| Satellites | 1–6 per task |
| Ground stations | 1–3 per task |
| Max steps / task | 100–200 |
| Stochastic events | 5 types, per-step probabilistic spawn |
| Compute requirement | < 2 vCPU, < 512 MB RAM |
| External dependencies | `gymnasium`, `pydantic`, `numpy` only for core env |

---

*Built for the OpenEnv Hackathon — deterministic, lightweight, reproducible.*
