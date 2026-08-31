import { useEffect, useRef, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import Nav from "../components/Nav";
import Logo from "../components/Logo";
import { getInvestigation, getStatus } from "../lib/api";
import { TERMINAL_STATUSES, POLL_MS } from "../lib/format";

const PHASES = [
  { key: "queued", label: "Queued", blurb: "Investigation request received." },
  {
    key: "investigating",
    label: "Investigating",
    blurb: "Searching and reading public sources for evidence.",
  },
  {
    key: "analyzing",
    label: "Analyzing",
    blurb: "Scoring evidence quality, sentiment, and risk.",
  },
  { key: "done", label: "Report ready", blurb: "Evidence-backed report complete." },
];

/* The investigation room — polls real status from the backend and shows
   only the phase the orchestration is actually in. No fake progress. */
export default function InvestigationRoom() {
  const { id } = useParams();
  const navigate = useNavigate();
  const [record, setRecord] = useState(null);
  const [status, setStatus] = useState(null);
  const [error, setError] = useState(null);
  const navigatedRef = useRef(false);

  useEffect(() => {
    let active = true;
    let timer = null;

    async function poll() {
      try {
        const s = await getStatus(id);
        if (!active) return;
        setStatus(s);
        if (TERMINAL_STATUSES.includes(s.status)) {
          if (s.status === "failed") {
            const full = await getInvestigation(id);
            if (active) setRecord(full);
          } else if (!navigatedRef.current) {
            navigatedRef.current = true;
            navigate(`/report/${id}`, { replace: true });
          }
          return; // stop polling
        }
      } catch (err) {
        if (active) setError(err);
        return; // stop polling on hard errors (404 etc.)
      }
      if (active) timer = setTimeout(poll, POLL_MS);
    }

    poll();
    return () => {
      active = false;
      if (timer) clearTimeout(timer);
    };
  }, [id, navigate]);

  useEffect(() => {
    // Fetch once for the business header.
    getInvestigation(id)
      .then((r) => setRecord(r))
      .catch(() => {});
  }, [id]);

  if (error) {
    return (
      <div className="page-room">
        <Nav dark />
        <main className="container room-container">
          <ErrorState
            title="We couldn’t find that investigation."
            body={error.message}
          >
            <Link to="/new" className="btn btn-primary">
              Start a new investigation
            </Link>
          </ErrorState>
        </main>
      </div>
    );
  }

  if (status && status.status === "failed") {
    return (
      <div className="page-room">
        <Nav dark />
        <main className="container room-container">
          <div className="room-card card card-pad">
            <div className="room-failed-icon" aria-hidden="true">
              ⚠
            </div>
            <h1>Investigation failed</h1>
            <p className="muted">
              {status.error ||
                "The investigation service could not complete this request."}
            </p>
            <div className="room-actions">
              <Link to="/new" className="btn btn-primary">
                Try again
              </Link>
              <Link to="/" className="btn btn-secondary">
                Back to home
              </Link>
            </div>
          </div>
        </main>
      </div>
    );
  }

  const currentPhase = phaseIndex(status?.status);
  const business = record?.business || {};

  return (
    <div className="page-room page-room-dark">
      <Nav dark />
      <main className="container room-container">
        <div className="room-card room-live">
          <div className="room-brand">
            <Logo size={28} inverted />
            <span className="room-brand-name">OpenCredit</span>
          </div>
          <p className="room-eyebrow">Investigation in progress</p>
          <h1 className="room-title">{business.name || "Your business"}</h1>
          {(business.location || business.category) && (
            <p className="room-sub muted">
              {[business.location, business.category].filter(Boolean).join(" · ")}
            </p>
          )}

          <ol className="phase-list" aria-label="Investigation progress">
            {PHASES.map((phase, i) => {
              const state =
                i < currentPhase ? "done" : i === currentPhase ? "active" : "todo";
              return (
                <li key={phase.key} className={`phase phase-${state}`}>
                  <span className="phase-dot" aria-hidden="true">
                    {state === "done" ? "✓" : i + 1}
                  </span>
                  <span className="phase-body">
                    <span className="phase-label">{phase.label}</span>
                    <span className="phase-blurb">{phase.blurb}</span>
                  </span>
                  {state === "active" && (
                    <span className="phase-spinner" aria-hidden="true" />
                  )}
                </li>
              );
            })}
          </ol>

          <div className="room-status" aria-live="polite">
            <span className="room-status-label">Current phase</span>
            <span className="room-status-value">
              {status?.phase_label || "Connecting…"}
            </span>
          </div>

          <p className="room-honest muted">
            We only show real progress — nothing here is simulated. A typical
            investigation takes 30–90 seconds.
          </p>
        </div>
      </main>
    </div>
  );
}

function phaseIndex(status) {
  switch (status) {
    case "queued":
      return 0;
    case "investigating":
      return 1;
    case "analyzing":
      return 2;
    case "completed":
    case "partial":
      return 3;
    default:
      return 0;
  }
}

function ErrorState({ title, body, children }) {
  return (
    <div className="room-card card card-pad">
      <h1>{title}</h1>
      <p className="muted">{body}</p>
      <div className="room-actions">{children}</div>
    </div>
  );
}
