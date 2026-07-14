"use client";

import { generateDiffFile } from "@git-diff-view/file";
import { DiffModeEnum, DiffView } from "@git-diff-view/react";
import { getDiffViewHighlighter } from "@git-diff-view/shiki";
import { useEffect, useState } from "react";

import { DEFAULT_LANGUAGE } from "./themes.js";

import "@git-diff-view/react/styles/diff-view-pure.css";

export type CodeDiffMode = "split" | "unified";
export type CodeDiffTheme = "light" | "dark";

export interface CodeDiffProps {
  oldContent: string;
  newContent: string;
  oldName?: string;
  newName?: string;
  lang?: string;
  mode?: CodeDiffMode;
  theme?: CodeDiffTheme;
}

const DIFF_MODES: Record<CodeDiffMode, DiffModeEnum> = {
  split: DiffModeEnum.Split,
  unified: DiffModeEnum.Unified,
};

type DiffFileInstance = ReturnType<typeof generateDiffFile>;

/**
 * Diff of two plain strings, computed in-browser. Takes strings only —
 * never a `DiffFile` instance — so it composes with any data source,
 * not just this package's other components.
 */
export function CodeDiff({
  oldContent,
  newContent,
  oldName = "before",
  newName = "after",
  lang = DEFAULT_LANGUAGE,
  mode = "unified",
  theme = "light",
}: CodeDiffProps) {
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
