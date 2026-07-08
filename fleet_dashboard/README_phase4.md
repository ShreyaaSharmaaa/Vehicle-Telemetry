# Phase 4: React Dashboard

## What's tested vs. not (same honesty as Phase 3)

My sandbox has no npm registry access either (403 Forbidden, same as pip
in Phase 3), so `npm install` has never actually run for this project.
BUT I found a workaround this time: this environment has `tsx` (an
esbuild-based TS/JSX runner) and React itself pre-installed globally for
unrelated purposes, which let me actually verify more than usual:

**Verified, with real rendering, using real data from your actual API responses:**
- All 4 presentational components (`RiskTierChart`, `FleetSummaryCards`,
  `VehicleList`, `AlertsList`) — server-side rendered with
  `react-dom/server`, using the *exact* numbers from your real
  `/fleet/summary` response (50 vehicles, 40,265 trips, avg score 16.26,
  etc.) — not made-up placeholder data.
- `VehicleDetail`'s initial render path (before its `fetch` calls resolve)
  — confirmed clean with a mocked `fetch`.
- `api.js`'s pure data-transformation logic (`normalizeTierCounts`) —
  tested directly with Node against your real tier-count numbers and edge
  cases (missing tiers, undefined input).
- All JSX syntax across every file — esbuild-transformed successfully.

**Not tested:** the actual `npm install` of the declared dependencies, the
Vite dev server actually starting, and the full app running in a real
browser with live data flowing through `useEffect` + `fetch` + the Vite
proxy. This is less risk than Phase 3 (way fewer dependencies, and the
core logic is verified), but it's still a first real run for you, not me.

## Two backend changes bundled with this phase

Building the frontend surfaced two gaps in Phase 3's API that needed
fixing first:
1. **CORS middleware** — was missing entirely. Added to `app/main.py`.
2. **`GET /vehicles`** (list all) and **`GET /alerts`** — neither existed
   yet. Added both. `/alerts` is a derived view over existing
   high/critical risk scores and maintenance predictions, not a new table
   — no need to re-run `load_data.py`.

You'll need to copy the updated `app/main.py` and `app/schemas.py` into
your existing `fleet_api/app/` folder, overwriting the Phase 3 versions.

## Setup

### 1. Install Node.js if you don't have it

Check first: `node --version`. If missing, install from nodejs.org (LTS
version). Node comes bundled with npm.

### 2. Install dependencies

From the `fleet_dashboard` folder:

```powershell
npm install
```

Only 4 packages total (react, react-dom, vite, @vitejs/plugin-react) —
deliberately minimal to reduce the chance of a version conflict like
Phase 3 hit.

### 3. Make sure the backend is running

In a separate terminal, your Phase 3 stack needs to already be up:

```powershell
cd ..\fleet_api
docker compose up
```

### 4. Start the dashboard

```powershell
npm run dev
```

This starts Vite's dev server on `http://localhost:5173`. Open that in
your browser.

## What you should see

- **Dashboard tab**: 4 stat cards (should show 50 vehicles, ~40,265 trips,
  ~16.3 avg risk score, 30 flagged for maintenance — matching what you
  already saw from `curl`), plus a bar chart of risk tier distribution.
- **Vehicles tab**: a clickable vehicle list on the left, trip history +
  live maintenance prediction on the right.
- **Alerts tab**: a merged feed of every High/Critical risk trip and
  maintenance prediction, newest first.

## Design decisions (and why)

**No react-router-dom.** Tab switching is plain `useState` in `App.jsx`.
For a 3-view dashboard, a router is unnecessary complexity — and one
fewer dependency that could fail to install.

**No axios.** Native `fetch` (built into every modern browser) covers
everything this dashboard needs.

**No charting library.** `RiskTierChart.jsx` is hand-rolled SVG. This
dashboard only ever needs one chart shape — pulling in a full charting
library for that isn't worth the dependency weight or install risk.

**Vite's dev proxy, not hardcoded URLs.** `vite.config.js` proxies
`/api/*` to `http://localhost:8000`. The frontend code never hardcodes
the backend's location, and this same proxy pattern is what you'd
configure in nginx for a real production deployment — dev and prod
architecture match instead of diverging.

## What Phase 5 could cover

This is genuinely a complete, demoable product now. If you want to keep
going: Docker Compose for the frontend too (so `docker compose up`
starts everything, not just backend + DB), authentication/roles, or
polishing the visual design pass now that the functional skeleton works.
