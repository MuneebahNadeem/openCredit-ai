import { Link } from "react-router-dom";
import Nav from "../components/Nav";
import Logo from "../components/Logo";
import ReportView from "../components/ReportView";
import { exampleBusiness, exampleReport } from "../data/exampleReport";

const STEPS = [
  {
    n: "01",
    title: "Share what you know",
    body: "A business name is enough. Add a website, Instagram, Daraz store, or a short description — every extra link sharpens the investigation.",
  },
  {
    n: "02",
    title: "The agent investigates",
    body: "OpenCredit’s investigation agent searches public sources, reads pages, and extracts structured evidence — followers, reviews, sales history, activity.",
  },
  {
    n: "03",
    title: "Get an evidence-backed report",
    body: "The ML layer scores trustworthiness and business potential separately. Every number links back to the exact source it came from.",
  },
];

const EVIDENCE_TYPES = [
  {
    icon: "◎",
    title: "Social presence",
    body: "Follower counts, engagement, and how recently the business was active.",
  },
  {
    icon: "★",
    title: "Reviews & reputation",
    body: "Star ratings, review volume, and complaint patterns across platforms.",
  },
  {
    icon: "▣",
    title: "Marketplace activity",
    body: "Units sold, active listings, and price activity on Daraz and similar marketplaces.",
  },
  {
    icon: "✎",
    title: "Self-reported info",
    body: "What the business owner shares — always labeled as inference, never treated as verified fact.",
  },
];

const PRINCIPLES = [
  {
    title: "Never fabricates",
    body: "If something can’t be found, the report says so. No score is invented to fill a gap.",
  },
  {
    title: "Evidence or nothing",
    body: "Every claim traces to a source URL and a confidence level. Inferences stay labeled as inferences.",
  },
  {
    title: "Missing ≠ fraud",
    body: "An informal business without a website isn’t hiding — most simply never needed one. Gaps are gaps.",
  },
  {
    title: "Decision support, not decisions",
    body: "OpenCredit informs human lenders. It never approves or declines anyone on its own.",
  },
];

export default function Landing() {
  return (
    <div className="landing">
      <Nav dark />

      {/* ── Hero ─────────────────────────────────────────────── */}
      <section className="hero">
        <div className="hero-glow" aria-hidden="true" />
        <div className="container hero-inner">
          <div className="hero-copy">
            <span className="chip hero-chip">Evidence-based business intelligence</span>
            <h1>
              Lend to businesses
              <br />
              the system <em>can’t see</em>.
            </h1>
            <p className="hero-sub">
              Millions of creditworthy micro-businesses — home boutiques,
              tailors, home kitchens — are invisible to traditional underwriting.
              OpenCredit investigates their public footprint and turns it into an
              evidence-backed trust report.
            </p>
            <div className="hero-actions">
              <Link to="/new" className="btn btn-primary btn-lg">
                Investigate a business
              </Link>
              <a href="#example" className="btn btn-ghost btn-lg">
                See an example report
              </a>
            </div>
            <div className="hero-stats" aria-label="Report contents">
              <div>
                <strong>2</strong>
                <span>independent assessments</span>
              </div>
              <div>
                <strong>40+</strong>
                <span>ML features per report</span>
              </div>
              <div>
                <strong>100%</strong>
                <span>claims traceable to sources</span>
              </div>
            </div>
          </div>
          <div className="hero-visual" aria-hidden="true">
            <div className="hero-card card">
              <div className="hero-card-head">
                <span className="hero-card-dot" />
                <span className="hero-card-dot" />
                <span className="hero-card-dot" />
                <span className="hero-card-title">
                  report · karachi-threads
                </span>
              </div>
              <div className="hero-card-body">
                <div className="hc-row">
                  <span className="hc-label">Trustworthiness</span>
                  <div className="hc-bar">
                    <span style={{ width: "62%" }} className="hc-fill hc-mod" />
                  </div>
                  <span className="hc-val">62</span>
                </div>
                <div className="hc-row">
                  <span className="hc-label">Business potential</span>
                  <div className="hc-bar">
                    <span style={{ width: "74%" }} className="hc-fill hc-high" />
                  </div>
                  <span className="hc-val">74</span>
                </div>
                <div className="hc-evidence">
                  <div className="hc-eq">
                    <span className="hc-icon">◎</span>
                    <span>instagram · 12.5k followers</span>
                  </div>
                  <div className="hc-eq">
                    <span className="hc-icon">★</span>
                    <span>google · 4.6★ · 212 reviews</span>
                  </div>
                  <div className="hc-eq">
                    <span className="hc-icon">▣</span>
                    <span>daraz · 340 sold</span>
                  </div>
                  <div className="hc-eq hc-eq-dim">
                    <span className="hc-icon">✎</span>
                    <span>self-reported revenue (inference)</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ── Problem ──────────────────────────────────────────── */}
      <section className="section">
        <div className="container">
          <p className="section-eyebrow">The problem</p>
          <h2 className="section-title">
            Great businesses, invisible to credit.
          </h2>
          <p className="section-sub">
            A tailor with a decade of orders, 212 five-star reviews, and a
            waiting list has no credit score — no financial statements, no
            collateral file, no formal history. Lenders see risk. The evidence
            of trustworthiness is public; nobody was reading it. Until now.
          </p>
        </div>
      </section>

      {/* ── How it works ─────────────────────────────────────── */}
      <section className="section section-dark" id="how-it-works">
        <div className="container">
          <p className="section-eyebrow">How it works</p>
          <h2 className="section-title">From a name to a decision-grade report.</h2>
          <div className="steps-grid">
            {STEPS.map((step) => (
              <div key={step.n} className="step-card">
                <span className="step-n">{step.n}</span>
                <h3>{step.title}</h3>
                <p>{step.body}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── What we look at ──────────────────────────────────── */}
      <section className="section">
        <div className="container">
          <p className="section-eyebrow">What we look at</p>
          <h2 className="section-title">Public evidence, structured.</h2>
          <p className="section-sub">
            The agent reads the same pages a diligent loan officer would — it
            just does it in seconds and never forgets a source.
          </p>
          <div className="evidence-types-grid">
            {EVIDENCE_TYPES.map((t) => (
              <div key={t.title} className="card card-pad et-card">
                <span className="et-icon" aria-hidden="true">
                  {t.icon}
                </span>
                <h3>{t.title}</h3>
                <p className="muted">{t.body}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── Two assessments ──────────────────────────────────── */}
      <section className="section section-dark">
        <div className="container two-col">
          <div>
            <p className="section-eyebrow">Two questions, not one</p>
            <h2 className="section-title">
              Can you trust them? Can they grow?
            </h2>
            <p className="section-sub">
              Most scoring systems blend everything into one number. OpenCredit
              keeps two assessments separate, because they answer different
              questions — and conflating them hides exactly the nuance a lender
              needs.
            </p>
            <ul className="separate-list">
              <li>
                <strong>Trustworthiness</strong> — does the evidence support that
                this business is real, active, and honest?
              </li>
              <li>
                <strong>Business potential</strong> — does the evidence support
                demand, audience, and room to grow?
              </li>
            </ul>
          </div>
          <div className="two-gauges card card-pad">
            <div>
              <h3>Trustworthiness</h3>
              <div className="mini-bars">
                <div className="hc-row">
                  <span className="hc-label">Example A</span>
                  <div className="hc-bar">
                    <span style={{ width: "82%" }} className="hc-fill hc-high" />
                  </div>
                  <span className="hc-val">82</span>
                </div>
                <div className="hc-row">
                  <span className="hc-label">Example B</span>
                  <div className="hc-bar">
                    <span style={{ width: "38%" }} className="hc-fill hc-low" />
                  </div>
                  <span className="hc-val">38</span>
                </div>
              </div>
            </div>
            <div>
              <h3>Business potential</h3>
              <div className="mini-bars">
                <div className="hc-row">
                  <span className="hc-label">Example A</span>
                  <div className="hc-bar">
                    <span style={{ width: "44%" }} className="hc-fill hc-mod" />
                  </div>
                  <span className="hc-val">44</span>
                </div>
                <div className="hc-row">
                  <span className="hc-label">Example B</span>
                  <div className="hc-bar">
                    <span style={{ width: "77%" }} className="hc-fill hc-high" />
                  </div>
                  <span className="hc-val">77</span>
                </div>
              </div>
            </div>
            <p className="muted two-gauges-note">
              The same two businesses, scored on both axes — high trust with low
              potential, low trust with high potential. One blended score would
              call them equal.
            </p>
          </div>
        </div>
      </section>

      {/* ── Example report ───────────────────────────────────── */}
      <section className="section" id="example">
        <div className="container">
          <p className="section-eyebrow">Example report</p>
          <h2 className="section-title">What a report looks like.</h2>
          <p className="section-sub">
            This is the real report interface with sample data for a fictional
            business — everything below is what a live investigation produces.
          </p>
          <div className="example-frame">
            <ReportView report={exampleReport} business={exampleBusiness} demo />
          </div>
        </div>
      </section>

      {/* ── Principles ───────────────────────────────────────── */}
      <section className="section section-dark">
        <div className="container">
          <p className="section-eyebrow">Our rules</p>
          <h2 className="section-title">Built on evidence discipline.</h2>
          <div className="principles-grid">
            {PRINCIPLES.map((p) => (
              <div key={p.title} className="principle">
                <h3>{p.title}</h3>
                <p>{p.body}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── CTA ──────────────────────────────────────────────── */}
      <section className="section cta-final">
        <div className="container cta-final-inner">
          <h2>Run your first investigation.</h2>
          <p className="section-sub">
            All you need is a business name. The report takes about a minute.
          </p>
          <Link to="/new" className="btn btn-primary btn-lg">
            Start investigation
          </Link>
        </div>
      </section>

      {/* ── Footer ───────────────────────────────────────────── */}
      <footer className="footer">
        <div className="container footer-inner">
          <div className="footer-brand">
            <Logo size={26} inverted withWordmark />
          </div>
          <p className="footer-note">
            Hackathon project · OpenCredit AI · Decision support for
            informal-market lending
          </p>
        </div>
      </footer>
    </div>
  );
}
