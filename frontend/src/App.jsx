import { BrowserRouter, Route, Routes, useLocation } from "react-router-dom";
import { useEffect } from "react";
import Landing from "./pages/Landing";
import NewInvestigation from "./pages/NewInvestigation";
import InvestigationRoom from "./pages/InvestigationRoom";
import ReportPage from "./pages/ReportPage";
import SavedReports from "./pages/SavedReports";

function ScrollToTop() {
  const { pathname } = useLocation();
  useEffect(() => {
    window.scrollTo(0, 0);
  }, [pathname]);
  return null;
}

export default function App() {
  return (
    <BrowserRouter>
      <ScrollToTop />
      <Routes>
        <Route path="/" element={<Landing />} />
        <Route path="/new" element={<NewInvestigation />} />
        <Route path="/investigation/:id" element={<InvestigationRoom />} />
        <Route path="/report/:id" element={<ReportPage />} />
        <Route path="/saved" element={<SavedReports />} />
        <Route path="*" element={<Landing />} />
      </Routes>
    </BrowserRouter>
  );
}
