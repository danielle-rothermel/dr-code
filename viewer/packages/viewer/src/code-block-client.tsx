"use client";

import { useEffect, useState } from "react";
import ShikiHighlighter from "react-shiki/core";

import { getClientHighlighter, type ClientHighlighter } from "./highlighter.js";
import { DEFAULT_LANGUAGE, SHIKI_THEMES } from "./themes.js";

export interface CodeBlockClientProps {
  code: string;
  lang?: string;
  className?: string;
}

/**
 * Client-tier code panel for live-fetched text (e.g. playground POST
 * responses arriving in the browser). Static code should use the
 * zero-JS server `CodeBlock` instead.
 */
export function CodeBlockClient({
  code,
  lang = DEFAULT_LANGUAGE,
  className,
}: CodeBlockClientProps) {
  const [highlighter, setHighlighter] = useState<ClientHighlighter | null>(
    null,
  );
  useEffect(() => {
    let active = true;
    void getClientHighlighter().then((loaded) => {
      if (active) setHighlighter(loaded);
    });
    return () => {
      active = false;
    };
  }, []);

  const classes = className ? `drv-code-block ${className}` : "drv-code-block";
  if (highlighter === null) {
    return (
      <div className={classes}>
        <pre>
          <code>{code}</code>
        </pre>
      </div>
    );
  }
  return (
    <div className={classes}>
      <ShikiHighlighter
        highlighter={highlighter}
        language={lang}
        theme={SHIKI_THEMES}
        defaultColor="light-dark()"
        showLanguage={false}
        addDefaultStyles={false}
      >
        {code}
      </ShikiHighlighter>
    </div>
  );
}
