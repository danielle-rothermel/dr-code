import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import "@dr-code/viewer/styles.css";
import { PreprocessingViewer } from "./app";
import "./styles.css";

const root = document.getElementById("root");

if (root === null) {
  throw new Error("Preprocessing analysis root element was not found");
}

createRoot(root).render(
  <StrictMode>
    <PreprocessingViewer />
  </StrictMode>,
);
