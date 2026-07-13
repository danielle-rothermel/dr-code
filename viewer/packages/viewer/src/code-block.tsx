import { codeToHtml } from "shiki";
import type { JSX } from "react";

import { DEFAULT_LANGUAGE, SHIKI_THEMES } from "./themes.js";

export interface CodeBlockProps {
  code: string;
  lang?: string;
  className?: string;
}

/**
 * Server-tier code panel (RSC, zero client JS). Use for static code:
 * task cards, persisted results. For live-fetched text use
 * `CodeBlockClient`.
 */
export async function CodeBlock({
  code,
  lang = DEFAULT_LANGUAGE,
  className,
}: CodeBlockProps): Promise<JSX.Element> {
  const html = await codeToHtml(code, {
    lang,
    themes: SHIKI_THEMES,
    defaultColor: "light-dark()",
  });
  const classes = className ? `drv-code-block ${className}` : "drv-code-block";
  return (
    <div className={classes} dangerouslySetInnerHTML={{ __html: html }} />
  );
}
