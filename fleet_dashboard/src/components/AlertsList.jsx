import React from 'react'

const TIER_CLASS = {
  low: 'tier-low',
  medium: 'tier-medium',
  high: 'tier-high',
  critical: 'tier-critical',
}

export default function AlertsList({ alerts }) {
  if (alerts.length === 0) {
    return <p className="muted">No High/Critical alerts right now.</p>
  }

  return (
    <ul className="alerts-list">
      {alerts.map((a, i) => (
        <li key={`${a.reference_id}-${i}`} className={`alert-item ${TIER_CLASS[a.severity]}`}>
          <span className={`tier-badge ${TIER_CLASS[a.severity]}`}>{a.severity}</span>
          <span className="alert-type">{a.alert_type}</span>
          <span className="alert-vehicle">{a.vehicle_id}</span>
          <span className="alert-message">{a.message}</span>
          <span className="alert-time muted">{new Date(a.occurred_at).toLocaleString()}</span>
        </li>
      ))}
    </ul>
  )
}
