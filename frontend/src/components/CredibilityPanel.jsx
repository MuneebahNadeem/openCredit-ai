import { titleCase } from "../lib/format";

const SUB_SCORES = [
  ["source_reliability_score", "Source reliability"],
  ["evidence_quality_score", "Evidence quality"],
  ["confidence_score", "Agent confidence"],
  ["reliable_ratio", "Reliable ratio"],
  ["source_diversity_score", "Source diversity"],
  ["corroboration_score", "Corroboration"],
  ["evidence_depth_score", "Evidence depth"],
];

export default function CredibilityPanel({ credibility }) {
  if (!credibility) return null;
  return (
    <div className="credibility">
      <div className="credibility-head">
        <span className="credibility-label">Evidence credibility</span>
        <span className={`credibility-level level-${credibility.level}`}>
          {titleCase(credibility.level)}
        </span>
        <span className="credibility-overall">
          {Math.round((credibility.overall_score ?? 0) * 100)} / 100
        </span>
      </div>
      <ul className="score-rows">
        {SUB_SCORES.map(([key, label]) => (
          <li key={key}>
            <span className="score-row-label">{label}</span>
            <div
              className="score-row-bar"
              role="img"
              aria-label={`${label}: ${Math.round(
                (credibility[key] ?? 0) * 100
              )}%`}
            >
              <span
                style={{ width: `${Math.round((credibility[key] ?? 0) * 100)}%` }}
              />
            </div>
            <span className="score-row-num">
              {Math.round((credibility[key] ?? 0) * 100)}
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}
