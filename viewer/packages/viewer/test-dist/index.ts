import {
  CodeBlock,
  CodeDiff,
  StatusBadge,
  type CodeBlockProps,
  type CodeDiffProps,
  type StatusBadgeProps,
} from "@dr-code/viewer";

const codeBlockProps: CodeBlockProps = { code: "plain text", lang: "unknown" };
const codeDiffProps: CodeDiffProps = {
  oldContent: "before",
  newContent: "after",
};
const statusBadgeProps: StatusBadgeProps = {
  status: "neutral",
  children: "not run",
};

void [
  CodeBlock,
  CodeDiff,
  StatusBadge,
  codeBlockProps,
  codeDiffProps,
  statusBadgeProps,
];
