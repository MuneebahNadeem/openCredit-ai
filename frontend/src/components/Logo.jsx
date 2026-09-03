export default function Logo({
  size = 34,
  inverted = false,
  withWordmark = false,
  className = "",
}) {
  const bg = inverted ? "var(--accent)" : "var(--ink)";
  const fg = inverted ? "var(--ink)" : "var(--accent)";
  const column = inverted ? "var(--ink)" : "var(--light)";

  const mark = (
    <span className={`logo ${className}`} aria-hidden="true">
      <svg width={size} height={size} viewBox="0 0 40 40" role="img">
        <rect width="40" height="40" rx="10" fill={bg} />
        {/* Roof / pediment */}
        <path d="M8 14 L20 6 L32 14 Z" fill={fg} />
        {/* Columns */}
        <rect x="10" y="16" width="3.5" height="15" rx="1" fill={column} />
        <rect x="15.5" y="16" width="3.5" height="15" rx="1" fill={column} />
        <rect x="21" y="16" width="3.5" height="15" rx="1" fill={column} />
        <rect x="26.5" y="16" width="3.5" height="15" rx="1" fill={column} />
        {/* Dollar sign in pediment */}
        <text
          x="20"
          y="14"
          textAnchor="middle"
          dominantBaseline="central"
          fontFamily="var(--font-mono)"
          fontSize="9"
          fontWeight="700"
          fill={bg}
        >
          $
        </text>
      </svg>
    </span>
  );

  if (!withWordmark) return mark;

  return (
    <span className={`logo-wordmark ${className}`} style={{ display: "inline-flex", alignItems: "center", gap: "0.55em" }}>
      {mark}
      <span className="logo-text" style={{ fontFamily: "var(--font-display)", fontWeight: 700, fontSize: size * 0.75, letterSpacing: "-0.02em", color: inverted ? "var(--light)" : "var(--ink)" }}>
        Open<em style={{ fontStyle: "normal", color: "var(--accent)" }}>Credit</em>
      </span>
    </span>
  );
}
