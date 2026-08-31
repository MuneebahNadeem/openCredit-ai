import { Link, NavLink } from "react-router-dom";
import Logo from "./Logo";

export default function Nav({ dark = true }) {
  return (
    <header className={`nav ${dark ? "nav-dark" : ""}`}>
      <div className="container nav-inner">
        <Link to="/" className="nav-brand">
          <Logo size={32} inverted={dark} />
          <span className="nav-name">
            Open<em>Credit</em>
          </span>
        </Link>
        <nav className="nav-links" aria-label="Primary">
          <NavLink to="/" end>
            Home
          </NavLink>
          <a href="/#how-it-works">How it works</a>
          <a href="/#example">Example report</a>
          <NavLink to="/new" className="nav-cta">
            Start investigation
          </NavLink>
        </nav>
      </div>
    </header>
  );
}
