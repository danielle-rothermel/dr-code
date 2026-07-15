import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import "@dr-code/viewer/styles.css";
import { Gallery } from "./gallery";
import "./styles.css";

const root = document.getElementById("root");

if (root === null) {
  throw new Error("Gallery root element was not found");
}

createRoot(root).render(
  <StrictMode>
    <Gallery />
  </StrictMode>,
);
