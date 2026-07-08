/**
 * api.js
 * Thin fetch wrapper for the FastAPI backend. Deliberately no axios
 * dependency -- native fetch covers everything this dashboard needs, and
 * every extra dependency is one more thing that can fail to install.
 *
 * All calls go through /api/... (see vite.config.js's proxy), which
 * forwards to http://localhost:8000 server-side. This avoids the browser
 * ever making a direct cross-origin request during development.
 */

const BASE = '/api'

async function request(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  })
  if (!res.ok) {
    const body = await res.text()
    throw new Error(`${res.status} ${res.statusText}: ${body}`)
  }
  return res.json()
}

export const api = {
  health: () => request('/health'),
  fleetSummary: () => request('/fleet/summary'),
  alerts: (limit = 50) => request(`/alerts?limit=${limit}`),
  vehicles: () => request('/vehicles'),
  vehicle: (vehicleId) => request(`/vehicles/${vehicleId}`),
  vehicleTrips: (vehicleId, limit = 20) =>
    request(`/vehicles/${vehicleId}/trips?limit=${limit}`),
  maintenancePrediction: (vehicleId) =>
    request(`/vehicles/${vehicleId}/maintenance-prediction`),
}

/**
 * Small pure helper, exported separately so it's testable without a
 * running server or browser -- just plain data transformation.
 * Turns { Low: 3, High: 1 } into a sorted array for chart rendering,
 * always including all four tiers even if a tier has zero trips (so the
 * chart doesn't silently drop a bar and look like a rendering bug).
 */
const TIER_ORDER = ['Low', 'Medium', 'High', 'Critical']

export function normalizeTierCounts(riskTierCounts) {
  return TIER_ORDER.map((tier) => ({
    tier,
    count: riskTierCounts?.[tier] ?? 0,
  }))
}
