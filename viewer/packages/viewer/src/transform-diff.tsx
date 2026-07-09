"use client";

import { generateDiffFile } from "@git-diff-view/file";
import { DiffModeEnum, DiffView } from "@git-diff-view/react";
import { getDiffViewHighlighter } from "@git-diff-view/shiki";
import { useEffect, useState } from "react";

import { DEFAULT_LANGUAGE } from "./themes.js";

import "@git-diff-view/react/styles/diff-view-pure.css";

export type TransformDiffMode = "split" | "unified";
export type TransformDiffTheme = "light" | "dark";

export interface TransformDiffProps {
  oldContent: string;
  newContent: string;
  oldName?: string;
  newName?: string;
  lang?: string;
  mode?: TransformDiffMode;
  theme?: TransformDiffTheme;
}

const DIFF_MODES: Record<TransformDiffMode, DiffModeEnum> = {
  split: DiffModeEnum.Split,
  unified: DiffModeEnum.Unified,
};

type DiffFileInstance = ReturnType<typeof generateDiffFile>;

/**
 * Client-tier diff of two plain strings, computed in-browser.
 * DiffFile instances cannot cross the RSC boundary, so this component
 * only ever accepts strings (ADR 0006).
 */
export function TransformDiff({
  oldContent,
  newContent,
  oldName = "before",
  newName = "after",
  lang = DEFAULT_LANGUAGE,
  mode = "unified",
  theme = "light",
}: TransformDiffProps) {
  const [diffFile, setDiffFile] = useState<DiffFileInstance | null>(null);

  useEffect(() => {
    let active = true;
    const file = generateDiffFile(
      oldName,
      oldContent,
      newName,
      newContent,
      lang,
      lang,
    );
    void getDiffViewHighlighter().then((highlighter) => {
      if (!active) return;
      file.initTheme(theme);
      file.initRaw();
      file.initSyntax({ registerHighlighter: highlighter });
      file.buildSplitDiffLines();
      file.buildUnifiedDiffLines();
      setDiffFile(file);
    });
    return () => {
      active = false;
    };
  }, [oldContent, newContent, oldName, newName, lang, theme]);

  if (diffFile === null) {
    return (
      <div className="drv-transform-diff drv-transform-diff-pending">
        <pre>
          <code>{newContent}</code>
        </pre>
      </div>
    );
  }
  return (
    <div className="drv-transform-diff">
      <DiffView
        diffFile={diffFile}
        diffViewMode={DIFF_MODES[mode]}
        diffViewTheme={theme}
        diffViewHighlight
        diffViewWrap
      />
    </div>
  );
}
