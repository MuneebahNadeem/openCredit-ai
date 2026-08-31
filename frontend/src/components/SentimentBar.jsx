import { titleCase } from "../lib/format";

export default function SentimentBar({ sentiment }) {
  if (!sentiment) return null;
  const { positive, negative, neutral, compound, label } = sentiment;
  const pct = (v) => `${Math.round((v ?? 0) * 100)}%`;
  const compoundPct = (((compound ?? 0) + 1) / 2) * 100;
  return (
    <div className="sentiment">
      <div className="sentiment-head">
        <span className="credibility-label">Public sentiment</span>
        <span
          className={`sentiment-label sentiment-${label || "neutral"}`}
        >
          {titleCase(label || "neutral")}
        </span>
      </div>
      <div
        className="sentiment-stacked"
        role="img"
        aria-label={`Positive ${pct(positive)}, neutral ${pct(neutral)}, negative ${pct(
          negative
        )}`}
      >
        <span className="seg seg-positive" style={{ width: pct(positive) }} />
        <span className="seg seg-neutral" style={{ width: pct(neutral) }} />
        <span className="seg seg-negative" style={{ width: pct(negative) }} />
      </div>
      <div className="sentiment-legend">
        <span>
          <i className="dot dot-positive" /> Positive {pct(positive)}
        </span>
        <span>
          <i className="dot dot-neutral" /> Neutral {pct(neutral)}
        </span>
        <span>
          <i className="dot dot-negative" /> Negative {pct(negative)}
        </span>
      </div>
      <div className="sentiment-compound">
        <span className="score-row-label">Overall tone</span>
        <div
          className="compound-track"
          role="img"
          aria-label={`Compound sentiment ${((compound ?? 0)).toFixed(2)} of 1`}
        >
          <span className="compound-zero" />
          <span
            className={`compound-fill ${compound >= 0 ? "up" : "down"}`}
            style={
              compound >= 0
                ? { left: "50%", width: `${compoundPct - 50}%` }
                : { left: `${compoundPct}%`, width: `${50 - compoundPct}%` }
            }
          />
        </div>
        <span className="score-row-num">{(compound ?? 0).toFixed(2)}</span>
      </div>
    </div>
  );
}
