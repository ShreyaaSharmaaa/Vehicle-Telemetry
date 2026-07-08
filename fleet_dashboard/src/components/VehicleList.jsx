import React from 'react'

export default function VehicleList({ vehicles, selectedId, onSelect }) {
  return (
    <div className="vehicle-list">
      <h3>Vehicles ({vehicles.length})</h3>
      <ul>
        {vehicles.map((v) => (
          <li
            key={v.vehicle_id}
            className={v.vehicle_id === selectedId ? 'selected' : ''}
            onClick={() => onSelect(v.vehicle_id)}
          >
            {v.vehicle_id}
            <span className="odometer">{Number(v.odometer_km).toLocaleString()} km</span>
          </li>
        ))}
      </ul>
    </div>
  )
}
