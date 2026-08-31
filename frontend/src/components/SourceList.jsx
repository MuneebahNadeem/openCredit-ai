import { SOURCE_TYPE_META } from "../lib/format";
import { ReliabilityBadge } from "./Badges";

export default function SourceList({ sources }) {
  if (!sources || sources.length === 0) {
    return <p className="empty-note">No sources were examined.</p>;
  }
  return (
    <ul className="source-list">
      {sources.map((source) => {
        const meta = SOURCE_TYPE_META[source.type] || SOURCE_TYPE_META.web;
        return (
          <li key={source.url} className="source-item">
            <span className="source-icon" aria-hidden="true">
              {meta.icon}
            </span>
            <div className="source-body">
              <div className="source-head">
                <a
                  href={source.url}
                  target="_blank"
                  rel="noreferrer noopener"
                  className="source-name"
                >
                  {source.name}
                </a>
                <span className="chip">{meta.label}</span>
                <ReliabilityBadge value={source.reliability} />
              </div>
              <div className="source-meta">
                {source.evidence_count > 0 ? (
                  <>
                    {source.evidence_count} evidence item
                    {source.evidence_count === 1 ? "" : "s"} ·{" "}
                    {source.evidence_fields.join(", ")}
                  </>
                ) : (
                  "Visited — no extractable evidence"
                )}
              </div>
            </div>
          </li>
        );
      })}
    </ul>
  );
}
