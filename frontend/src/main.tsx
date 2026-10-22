import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import { upgradeLanHttp } from "./upgradeHttps";
import "./styles.css";

upgradeLanHttp();

if (typeof window !== "undefined") {
  const p = window.location.pathname.replace(/\/$/, "") || "/";
  if (p === "/") {
    window.history.replaceState(null, "", `/patient${window.location.search}${window.location.hash}`);
  }
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
