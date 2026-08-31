import { prettyName } from "../lib/format";
import { TypeBadge, ReliabilityBadge } from "./Badges";

export default function EvidenceTable({ evidence }) {
  if (!evidence || evidence.length === 0) {
    return (
      <p className="empty-note">No evidence items were collected.</p>
    );
  }
  return (
    <div className="evidence-wrap" role="region" aria-label="Evidence items">
      <table className="evidence-table">
        <thead>
          <tr>
            <th scope="col">What we found</th>
            <th scope="col">Value</th>
            <th scope="col">Type</th>
            <th scope="col">Confidence</th>
            <th scope="col">Source</th>
          </tr>
        </thead>
        <tbody>
          {evidence.map((item, i) => (
            <tr key={`${item.field_name}-${i}`}>
              <td>
                <span className="evidence-field">{prettyName(item.field_name)}</span>
                {item.raw_snippet && (
                  <span className="evidence-snippet">“{item.raw_snippet}”</span>
                )}
              </td>
              <td className="evidence-value">
                <span className="mono">{item.value}</span>
                {item.unit && <span className="evidence-unit"> {item.unit}</span>}
              </td>
              <td>
                <TypeBadge value={item.evidence_type} />
              </td>
              <td>
                <div
                  className="confidence-bar"
                  role="img"
                  aria-label={`Confidence ${Math.round((item.confidence ?? 0) * 100)}%`}
                >
                  <span
                    style={{ width: `${Math.round((item.confidence ?? 0) * 100)}%` }}
                  />
                </div>
                <span className="confidence-num">
                  {Math.round((item.confidence ?? 0) * 100)}%
                </span>
              </td>
              <td>
                {item.source_url ? (
                  <a
                    href={item.source_url}
                    target="_blank"
                    rel="noreferrer noopener"
                    className="evidence-source"
                  >
                    {item.source_name || hostname(item.source_url)}
                  </a>
                ) : (
                  <span className="evidence-source muted">
                    {item.source_name || "No source"}
                  </span>
                )}
                <ReliabilityBadge value={item.source_reliability} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function hostname(url) {
  try {
    return new URL(url).hostname;
  } catch {
    return url;
  }
}
