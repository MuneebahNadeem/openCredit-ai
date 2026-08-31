export default function Logo({ size = 34, inverted = false }) {
  const bg = inverted ? "var(--accent)" : "var(--ink)";
  const fg = inverted ? "var(--ink)" : "var(--accent)";
  return (
    <span className="logo" aria-hidden="true">
      <svg width={size} height={size} viewBox="0 0 40 40" role="img">
        <rect width="40" height="40" rx="11" fill={bg} />
        <circle
          cx="20"
          cy="20"
          r="11.5"
          fill="none"
          stroke={fg}
          strokeWidth="2.6"
        />
        <path
          d="M17.2 20c0-2.2 1.2-3.6 2.8-3.6c1.1 0 2 .6 2.3 1.6l-1.3.5c-.2-.5-.6-.8-1-.8c-.8 0-1.4.9-1.4 2.3c0 1.4.6 2.3 1.4 2.3c.5 0 .9-.3 1-.9l1.3.5c-.4 1.1-1.2 1.7-2.4 1.7c-1.6 0-2.7-1.4-2.7-3.6z"
          fill={fg}
        />
      </svg>
    </span>
  );
}
