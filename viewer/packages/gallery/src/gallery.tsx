import {
  CodeBlock,
  CodeDiff,
  StatusBadge,
  type CodeDiffTheme,
} from "@dr-code/viewer";

import {
  BADGE_FIXTURES,
  CHANGED_DIFF,
  CODE_FIXTURES,
  DIFF_MODES,
  UNCHANGED_DIFF,
} from "./fixtures";

interface ThemeBoardProps {
  theme: CodeDiffTheme;
}

function ThemeBoard({ theme }: ThemeBoardProps) {
  return (
    <article className={`gallery-board gallery-board--${theme}`}>
      <header className="gallery-board__header">
        <p className="gallery-eyebrow">{theme} presentation</p>
        <h2>{theme === "light" ? "Paper" : "Midnight"}</h2>
        <p>
          Every primitive is rendered against a fixed {theme} color scheme.
        </p>
      </header>

      <section className="gallery-section" aria-labelledby={`${theme}-badges`}>
        <div className="gallery-section__heading">
          <h3 id={`${theme}-badges`}>StatusBadge</h3>
          <code>status + children</code>
        </div>
        <div className="gallery-badge-row">
          {BADGE_FIXTURES.map(({ status, label }) => (
            <StatusBadge
              key={status}
              status={status}
              theme={theme}
              className={
                status === "neutral" ? "gallery-badge--outlined" : undefined
              }
            >
              {label}
            </StatusBadge>
          ))}
        </div>
        <p className="gallery-note">
          The neutral badge has a caller-provided class name.
        </p>
      </section>

      <section className="gallery-section" aria-labelledby={`${theme}-code`}>
        <div className="gallery-section__heading">
          <h3 id={`${theme}-code`}>CodeBlock</h3>
          <code>five bundled languages</code>
        </div>
        <div className="gallery-code-grid">
          {CODE_FIXTURES.map((fixture, index) => (
            <div className="gallery-language" key={fixture.lang}>
              <h4>{fixture.name}</h4>
              <div className="gallery-example">
                <span>Short</span>
                <CodeBlock
                  code={fixture.short}
                  lang={fixture.lang}
                  theme={theme}
                  className={
                    index === 0 ? "gallery-code-block--accent" : undefined
                  }
                />
              </div>
              <div className="gallery-example">
                <span>Long</span>
                <CodeBlock
                  code={fixture.long}
                  lang={fixture.lang}
                  theme={theme}
                />
              </div>
            </div>
          ))}
        </div>
        <p className="gallery-note">
          The short Python block has a caller-provided class name.
        </p>
      </section>

      <section className="gallery-section" aria-labelledby={`${theme}-diffs`}>
        <div className="gallery-section__heading">
          <h3 id={`${theme}-diffs`}>CodeDiff</h3>
          <code>changed + unchanged × unified + split</code>
        </div>
        <div className="gallery-diff-grid">
          {DIFF_MODES.map((mode) => (
            <div className="gallery-diff-example" key={`changed-${mode}`}>
              <h4>Changed · {mode}</h4>
              <CodeDiff {...CHANGED_DIFF} mode={mode} theme={theme} />
            </div>
          ))}
          {DIFF_MODES.map((mode) => (
            <div className="gallery-diff-example" key={`empty-${mode}`}>
              <h4>Empty diff · {mode}</h4>
              <CodeDiff {...UNCHANGED_DIFF} mode={mode} theme={theme} />
            </div>
          ))}
        </div>
        <p className="gallery-note">
          Empty diff panels are expected: identical before and after strings
          produce no changed rows.
        </p>
      </section>
    </article>
  );
}

export function Gallery() {
  return (
    <main>
      <header className="gallery-hero">
        <p className="gallery-eyebrow">@dr-code/viewer</p>
        <h1>Primitive gallery</h1>
        <p>
          A visual contract for code blocks, diffs, and status badges. Fixtures
          are deliberately static so component changes are easy to compare.
        </p>
      </header>
      <div className="gallery-boards">
        <ThemeBoard theme="light" />
        <ThemeBoard theme="dark" />
      </div>
    </main>
  );
}
