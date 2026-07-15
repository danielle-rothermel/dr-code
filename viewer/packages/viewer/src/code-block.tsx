"use client";

import { useEffect, useState } from "react";

import { getHighlighter } from "./highlighter.js";
import { DEFAULT_LANGUAGE, SHIKI_THEMES } from "./themes.js";

export interface CodeBlockProps {
  code: string;
  lang?: string;
  theme?: "light" | "dark";
  className?: string;
}

interface HighlightedCode {
  code: string;
  lang: string;
  theme: "light" | "dark";
  html: string;
}

/** A code panel that shows plain text until syntax highlighting is ready. */
export function CodeBlock({
  code,
  lang = DEFAULT_LANGUAGE,
  theme = "light",
  className,
}: CodeBlockProps) {
  const [highlighted, setHighlighted] = useState<HighlightedCode | null>(null);

  useEffect(() => {
    let active = true;
    void getHighlighter().then((loaded) => {
      const html = loaded.codeToHtml(code, {
        lang,
        theme: SHIKI_THEMES[theme],
      });
      if (active) setHighlighted({ code, lang, theme, html });
    });
    return () => {
      active = false;
    };
  }, [code, lang, theme]);

  const classes = className ? `drv-code-block ${className}` : "drv-code-block";
  if (
    highlighted === null ||
    highlighted.code !== code ||
    highlighted.lang !== lang ||
    highlighted.theme !== theme
  ) {
    return (
      <div className={classes} data-theme={theme}>
        <pre>
          <code>{code}</code>
        </pre>
      </div>
    );
  }

  return (
    <div
      className={classes}
      data-theme={theme}
      dangerouslySetInnerHTML={{ __html: highlighted.html }}
    />
  );
}
