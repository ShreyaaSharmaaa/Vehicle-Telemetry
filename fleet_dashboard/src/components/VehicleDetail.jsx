import React, { useEffect, useState } from 'react'
import { api } from '../api.js'

const TIER_CLASS = {
  Low: 'tier-low',
  Medium: 'tier-medium',
  High: 'tier-high',
  Critical: 'tier-critical',
}

export default function VehicleDetail({ vehicleId }) {
  const [trips, setTrips] = useState([])
  const [maintenance, setMaintenance] = useState(null)
  const [maintenanceError, setMaintenanceError] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    setMaintenanceError(null)
    setMaintenance(null)

    api.vehicleTrips(vehicleId, 15).then((data) => {
      if (!cancelled) setTrips(data)
    })

    api.maintenancePrediction(vehicleId)
      .then((data) => { if (!cancelled) setMaintenance(data) })
      .catch((err) => { if (!cancelled) setMaintenanceError(err.message) })
      .finally(() => { if (!cancelled) setLoading(false) })

    return () => { cancelled = true }
  }, [vehicleId])

  return (
    <div className="vehicle-detail">
      <h2>{vehicleId}</h2>

      <div className="maintenance-card">
        <h3>Maintenance Prediction</h3>
        {loading && <p>Loading...</p>}
        {!loading && maintenanceError && (
          <p className="muted">Not enough trip history yet ({maintenanceError})</p>
        )}
        {!loading && maintenance && (
          <div className={`maintenance-tier ${TIER_CLASS[maintenance.risk_tier]}`}>
            <div className="big-number">{(maintenance.failure_probability * 100).toFixed(1)}%</div>
            <div>failure probability within 30 days</div>
            <div className="tier-badge">{maintenance.risk_tier}</div>
            <div className="muted small">
              based on last {maintenance.trips_used_for_prediction} trips
            </div>
          </div>
        )}
      </div>

      <h3>Recent Trips</h3>
      <table className="trips-table">
        <thead>
          <tr>
            <th>Start Time</th>
            <th>Distance (km)</th>
            <th>Avg Speed</th>
            <th>Harsh Braking</th>
            <th>Risk Score</th>
            <th>Tier</th>
          </tr>
        </thead>
        <tbody>
          {trips.map((t) => (
            <tr key={t.trip_id}>
              <td>{new Date(t.trip_start_time).toLocaleString()}</td>
              <td>{t.distance_km.toFixed(1)}</td>
              <td>{t.avg_speed_kmh.toFixed(1)} km/h</td>
              <td>{t.harsh_braking_count}</td>
              <td>{t.risk_score?.risk_score.toFixed(1) ?? '-'}</td>
              <td>
                {t.risk_score && (
                  <span className={`tier-badge ${TIER_CLASS[t.risk_score.risk_tier]}`}>
                    {t.risk_score.risk_tier}
                  </span>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
