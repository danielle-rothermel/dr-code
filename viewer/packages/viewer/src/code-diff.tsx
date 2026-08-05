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

const DIFF_FONT_SIZES: Record<CodeDiffMode, number> = {
  split: 10.5,
  unified: 11,
};

type DiffFileInstance = ReturnType<typeof generateDiffFile>;

interface RenderedDiff {
  diffFile: DiffFileInstance;
  oldContent: string;
  newContent: string;
  oldName: string;
  newName: string;
  lang: string;
  theme: CodeDiffTheme;
}

export function CodeDiff({
  oldContent,
  newContent,
  oldName = "before",
  newName = "after",
  lang = DEFAULT_LANGUAGE,
  mode = "unified",
  theme = "light",
}: CodeDiffProps) {
  const [rendered, setRendered] = useState<RenderedDiff | null>(null);

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
      setRendered({
        diffFile: file,
        oldContent,
        newContent,
        oldName,
        newName,
        lang,
        theme,
      });
    });
    return () => {
      active = false;
    };
  }, [oldContent, newContent, oldName, newName, lang, theme]);

  if (
    rendered === null ||
    rendered.oldContent !== oldContent ||
    rendered.newContent !== newContent ||
    rendered.oldName !== oldName ||
    rendered.newName !== newName ||
    rendered.lang !== lang ||
    rendered.theme !== theme
  ) {
    return (
      <div className="drv-code-diff drv-code-diff-pending">
        <pre>
          <code>{newContent}</code>
        </pre>
      </div>
    );
  }

  return (
    <div className="drv-code-diff">
      <DiffView
        diffFile={rendered.diffFile}
        diffViewMode={DIFF_MODES[mode]}
        diffViewTheme={theme}
        diffViewFontSize={DIFF_FONT_SIZES[mode]}
        diffViewHighlight
        diffViewWrap
      />
    </div>
  );
}
