import React, { useEffect, useState } from 'react'
import { api, normalizeTierCounts } from './api.js'
import FleetSummaryCards from './components/FleetSummaryCards.jsx'
import RiskTierChart from './components/RiskTierChart.jsx'
import VehicleList from './components/VehicleList.jsx'
import VehicleDetail from './components/VehicleDetail.jsx'
import AlertsList from './components/AlertsList.jsx'

const TABS = ['Dashboard', 'Vehicles', 'Alerts']

export default function App() {
  const [tab, setTab] = useState('Dashboard')
  const [summary, setSummary] = useState(null)
  const [vehicles, setVehicles] = useState([])
  const [alerts, setAlerts] = useState([])
  const [selectedVehicleId, setSelectedVehicleId] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    api.fleetSummary().then(setSummary).catch((e) => setError(e.message))
    api.vehicles().then((data) => {
      setVehicles(data)
      if (data.length > 0) setSelectedVehicleId(data[0].vehicle_id)
    }).catch((e) => setError(e.message))
    api.alerts(50).then(setAlerts).catch((e) => setError(e.message))
  }, [])

  if (error) {
    return (
      <div className="app-error">
        <h2>Couldn't reach the API</h2>
        <p>{error}</p>
        <p className="muted">
          Check that <code>docker compose up</code> is running and you loaded
          data with <code>load_data.py</code>.
        </p>
      </div>
    )
  }

  return (
    <div className="app">
      <header>
        <h1>Fleet Telemetry &amp; Analytics Platform</h1>
        <nav>
          {TABS.map((t) => (
            <button
              key={t}
              className={tab === t ? 'active' : ''}
              onClick={() => setTab(t)}
            >
              {t}
            </button>
          ))}
        </nav>
      </header>

      <main>
        {tab === 'Dashboard' && summary && (
          <>
            <FleetSummaryCards summary={summary} />
            <RiskTierChart data={normalizeTierCounts(summary.risk_tier_counts)} />
          </>
        )}

        {tab === 'Vehicles' && (
          <div className="vehicles-layout">
            <VehicleList
              vehicles={vehicles}
              selectedId={selectedVehicleId}
              onSelect={setSelectedVehicleId}
            />
            {selectedVehicleId && <VehicleDetail vehicleId={selectedVehicleId} />}
          </div>
        )}

        {tab === 'Alerts' && (
          <div>
            <h2>Active Alerts</h2>
            <AlertsList alerts={alerts} />
          </div>
        )}
      </main>
    </div>
  )
}
