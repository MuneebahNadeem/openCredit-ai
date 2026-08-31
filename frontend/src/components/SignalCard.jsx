export default function SignalCard({ signal, kind }) {
  const positive = kind === "positive";
  return (
    <li className={`signal-card ${positive ? "signal-positive" : "signal-risk"}`}>
      <span className="signal-icon" aria-hidden="true">
        {positive ? "▲" : "▼"}
      </span>
      <div>
        <div className="signal-label">{signal.label}</div>
        <p className="signal-detail">{signal.detail}</p>
        {signal.evidence_refs && signal.evidence_refs.length > 0 && (
          <div className="signal-refs">
            Evidence:{" "}
            {signal.evidence_refs.map((ref, i) => (
              <span key={ref} className="signal-ref mono">
                {ref}
                {i < signal.evidence_refs.length - 1 ? "," : ""}
              </span>
            ))}
          </div>
        )}
      </div>
    </li>
  );
}
