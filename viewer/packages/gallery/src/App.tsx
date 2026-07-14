import { useEffect, useState } from "react";
import { CodeBlock, CodeDiff, StatusBadge } from "@dr-code/viewer";

import { GalleryCell } from "./GalleryCell";
import {
  CODE_BLOCK_FIXTURES,
  CODE_DIFF_FIXTURES,
  STATUS_BADGE_FIXTURES,
} from "./fixtures";

type Theme = "light" | "dark";

/**
 * Renders every `@dr-code/viewer` primitive against hand-written
 * fixtures, in a grid, so changes can be eyeballed in light and dark.
 * This is a verification surface, not a product: no router, no state
 * library, no CSS framework.
 */
export function App() {
  const [theme, setTheme] = useState<Theme>("light");

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
  }, [theme]);

  return (
    <div className="gallery">
      <header className="gallery-header">
        <h1>@dr-code/viewer gallery</h1>
        <button
          type="button"
          className="gallery-theme-toggle"
          onClick={() => setTheme((current) => (current === "light" ? "dark" : "light"))}
        >
          Switch to {theme === "light" ? "dark" : "light"} mode
        </button>
      </header>

      <section>
        <h2>CodeBlock</h2>
        <div className="gallery-grid">
          {CODE_BLOCK_FIXTURES.map((fixture) => (
            <GalleryCell
              key={fixture.description}
              title={fixture.title}
              description={fixture.description}
            >
              <CodeBlock code={fixture.code} lang={fixture.lang} />
            </GalleryCell>
          ))}
        </div>
      </section>

      <section>
        <h2>CodeDiff</h2>
        <div className="gallery-grid">
          {CODE_DIFF_FIXTURES.map((fixture) => (
            <GalleryCell
              key={fixture.description}
              title={fixture.title}
              description={fixture.description}
            >
              <CodeDiff
                oldContent={fixture.oldContent}
                newContent={fixture.newContent}
                oldName={fixture.oldName}
                newName={fixture.newName}
                lang={fixture.lang}
                mode={fixture.mode}
                theme={theme}
              />
            </GalleryCell>
          ))}
        </div>
      </section>

      <section>
        <h2>StatusBadge</h2>
        <div className="gallery-grid">
          {STATUS_BADGE_FIXTURES.map((fixture) => (
            <GalleryCell
              key={fixture.description}
              title={fixture.title}
              description={fixture.description}
            >
              <StatusBadge status={fixture.status}>{fixture.label}</StatusBadge>
            </GalleryCell>
          ))}
        </div>
      </section>
    </div>
  );
}
