import { titleCase, formatScore } from "../lib/format";

/* Semicircular gauge for one assessment axis (trust / potential). */
export default function LevelGauge({ score, level, label }) {
  const value = typeof score === "number" ? score : null;
  const pct = value === null ? 0 : Math.max(0, Math.min(1, value));
  // Semicircle arc: 180 degrees.
  const angle = pct * 180;
  const large = 0;
  const r = 62;
  const cx = 80;
  const cy = 76;
  const end = polar(cx, cy, r, angle);
  const levelText = level ? titleCase(level) : "No data";

  return (
    <div className="gauge">
      <svg
        viewBox="0 0 160 92"
        width="100%"
        role="img"
        aria-label={`${label}: ${levelText}${
          value === null ? "" : `, score ${Math.round(value * 100)} of 100`
        }`}
      >
        <path
          d={arcPath(cx, cy, r, 180)}
          fill="none"
          stroke="var(--light-2)"
          strokeWidth="12"
          strokeLinecap="round"
        />
        {value !== null && (
          <path
            d={`M ${cx - r} ${cy} A ${r} ${r} 0 ${large} 1 ${end.x} ${end.y}`}
            fill="none"
            stroke={levelColor(level)}
            strokeWidth="12"
            strokeLinecap="round"
            className="gauge-arc"
          />
        )}
        <text
          x={cx}
          y={cy - 8}
          textAnchor="middle"
          className={`gauge-score level-${level || "insufficient_evidence"}`}
        >
          {value === null ? "—" : formatScore(value)}
        </text>
        <text x={cx} y={cy + 14} textAnchor="middle" className="gauge-unit">
          / 100
        </text>
      </svg>
      <div className={`gauge-level level-${level || "insufficient_evidence"}`}>
        {levelText}
      </div>
    </div>
  );
}

function polar(cx, cy, r, angleDeg) {
  const rad = ((180 - angleDeg) * Math.PI) / 180;
  return { x: cx + r * Math.cos(rad), y: cy - r * Math.sin(rad) };
}

function arcPath(cx, cy, r, angleDeg) {
  const start = { x: cx - r, y: cy };
  const end = polar(cx, cy, r, angleDeg);
  return `M ${start.x} ${start.y} A ${r} ${r} 0 0 1 ${end.x} ${end.y}`;
}

function levelColor(level) {
  switch (level) {
    case "high":
      return "var(--high)";
    case "moderate":
      return "var(--moderate)";
    case "low":
      return "var(--low)";
    default:
      return "var(--insufficient)";
  }
}
