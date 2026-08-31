import { useState } from "react";
import { askQuestion } from "../lib/api";

/* "Ask OpenCredit" — Q&A over a completed report.
   Disabled in demo mode and when no LLM key is configured. */
export default function AskPanel({ investigationId, demo = false }) {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [unavailable, setUnavailable] = useState(false);

  async function submit(event) {
    event.preventDefault();
    if (!question.trim() || busy) return;
    setBusy(true);
    setError(null);
    setAnswer(null);
    try {
      const response = await askQuestion(investigationId, question.trim());
      setAnswer(response.answer);
    } catch (err) {
      if (err.status === 503) {
        setUnavailable(true);
        setError(err.message);
      } else if (err.status === 409) {
        setError(err.message);
      } else {
        setError(err.message || "Ask OpenCredit is unavailable right now.");
      }
    } finally {
      setBusy(false);
    }
  }

  if (demo) {
    return (
      <div className="ask-panel ask-demo">
        <h3>Ask OpenCredit</h3>
        <p className="muted">
          Ask a follow-up question about any report — “Why is the
          trustworthiness score moderate?” — and get an answer grounded only in
          the evidence above. Available on live reports when the demo server has
          an LLM key configured.
        </p>
        <div className="ask-disabled">
          <input
            type="text"
            placeholder="Ask about this report…"
            disabled
            aria-label="Ask about this report (disabled in demo)"
          />
          <button className="btn btn-secondary" disabled>
            Ask
          </button>
        </div>
      </div>
    );
  }

  if (unavailable) {
    return (
      <div className="ask-panel">
        <h3>Ask OpenCredit</h3>
        <div className="ask-note" role="note">
          {error}
        </div>
      </div>
    );
  }

  return (
    <div className="ask-panel">
      <h3>Ask OpenCredit</h3>
      <p className="muted">
        Answers come only from this report’s evidence — never speculation.
      </p>
      <form onSubmit={submit} className="ask-form">
        <label htmlFor="ask-input" className="visually-hidden">
          Question about this report
        </label>
        <input
          id="ask-input"
          type="text"
          value={question}
          placeholder="e.g. What is the biggest risk signal?"
          onChange={(e) => setQuestion(e.target.value)}
          disabled={busy}
        />
        <button className="btn btn-primary" type="submit" disabled={busy || !question.trim()}>
          {busy ? "Asking…" : "Ask"}
        </button>
      </form>
      {error && (
        <p className="ask-error" role="alert">
          {error}
        </p>
      )}
      {answer && (
        <div className="ask-answer">
          <div className="ask-answer-label">OpenCredit</div>
          <p>{answer}</p>
        </div>
      )}
    </div>
  );
}
