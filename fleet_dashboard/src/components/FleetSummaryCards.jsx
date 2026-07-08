import React from 'react'

export default function FleetSummaryCards({ summary }) {
  const cards = [
    { label: 'Total Vehicles', value: summary.total_vehicles },
    { label: 'Trips Scored', value: summary.total_trips_scored.toLocaleString() },
    { label: 'Avg Fleet Risk Score', value: summary.avg_fleet_risk_score.toFixed(1) },
    { label: 'Flagged for Maintenance', value: summary.vehicles_flagged_for_maintenance },
  ]

  return (
    <div className="cards-row">
      {cards.map((c) => (
        <div className="stat-card" key={c.label}>
          <div className="stat-value">{c.value}</div>
          <div className="stat-label">{c.label}</div>
        </div>
      ))}
    </div>
  )
}
