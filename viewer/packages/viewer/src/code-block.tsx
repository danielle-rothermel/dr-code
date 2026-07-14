"use client";

import { useEffect, useState } from "react";
import ShikiHighlighter from "react-shiki/core";

import {
  getClientHighlighter,
  resolveSupportedLang,
  type ClientHighlighter,
} from "./highlighter.js";
import { DEFAULT_LANGUAGE, SHIKI_THEMES } from "./themes.js";

export interface CodeBlockProps {
  code: string;
  lang?: string;
  className?: string;
}

/**
 * Highlighted code panel. Renders the plain code in a `<pre>`
 * immediately, then swaps in shiki-highlighted markup once the
 * highlighter has loaded in the browser.
 *
 * `lang` is matched against the grammars registered in
 * `highlighter.ts` (`SUPPORTED_LANGUAGES` there is the source of
 * truth). An unregistered `lang` renders as plaintext instead of
 * throwing or warning.
 */
export function CodeBlock({
  code,
  lang = DEFAULT_LANGUAGE,
  className,
}: CodeBlockProps) {
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
        language={resolveSupportedLang(lang, highlighter)}
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
