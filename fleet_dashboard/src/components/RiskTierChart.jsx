import React from 'react'

const TIER_COLORS = {
  Low: '#4caf7d',
  Medium: '#e8a838',
  High: '#e8703a',
  Critical: '#d64545',
}

/**
 * data: [{ tier: 'Low', count: 30766 }, ...]
 * Deliberately hand-rolled SVG rather than a charting library -- one less
 * dependency to install, and this dashboard only ever needs this one
 * chart shape.
 */
export default function RiskTierChart({ data }) {
  const maxCount = Math.max(...data.map((d) => d.count), 1)
  const barWidth = 90
  const gap = 30
  const chartHeight = 220
  const width = data.length * (barWidth + gap)

  return (
    <div className="chart-card">
      <h3>Risk Tier Distribution</h3>
      <svg width={width} height={chartHeight + 40} role="img" aria-label="Risk tier distribution bar chart">
        {data.map((d, i) => {
          const barHeight = (d.count / maxCount) * chartHeight
          const x = i * (barWidth + gap)
          const y = chartHeight - barHeight
          return (
            <g key={d.tier}>
              <rect
                x={x}
                y={y}
                width={barWidth}
                height={barHeight}
                fill={TIER_COLORS[d.tier] || '#999'}
                rx={4}
              />
              <text x={x + barWidth / 2} y={y - 8} textAnchor="middle" fontSize="14" fontWeight="600">
                {d.count.toLocaleString()}
              </text>
              <text x={x + barWidth / 2} y={chartHeight + 22} textAnchor="middle" fontSize="13">
                {d.tier}
              </text>
            </g>
          )
        })}
      </svg>
    </div>
  )
}
