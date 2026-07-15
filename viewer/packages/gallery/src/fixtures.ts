import type {
  CodeDiffMode,
  StatusBadgeStatus,
} from "@dr-code/viewer";

export interface CodeFixture {
  name: string;
  lang: string;
  short: string;
  long: string;
}

export const CODE_FIXTURES: readonly CodeFixture[] = [
  {
    name: "Python",
    lang: "python",
    short: `answer = sum(range(10))`,
    long: `from dataclasses import dataclass

@dataclass(frozen=True)
class Result:
    name: str
    score: float

def passing(results: list[Result], threshold: float = 0.8) -> list[str]:
    return [
        result.name
        for result in results
        if result.score >= threshold
    ]`,
  },
  {
    name: "JavaScript",
    lang: "javascript",
    short: `const total = values.reduce((sum, value) => sum + value, 0);`,
    long: `const groupByStatus = (checks) => {
  const groups = new Map();

  for (const check of checks) {
    const matches = groups.get(check.status) ?? [];
    matches.push(check.name);
    groups.set(check.status, matches);
  }

  return Object.fromEntries(groups);
};`,
  },
  {
    name: "TypeScript",
    lang: "typescript",
    short: `const greeting: string = "hello, gallery";`,
    long: `type Job = {
  id: string;
  state: "queued" | "running" | "complete";
};

export function nextJob(jobs: readonly Job[]): Job | undefined {
  return jobs.find((job) => job.state === "queued");
}

export const describeJob = (job: Job): string =>
  [job.id, job.state, "ready for visual inspection"].join(" · ");`,
  },
  {
    name: "JSON",
    lang: "json",
    short: `{"status":"success","duration_ms":18}`,
    long: `{
  "run_id": "gallery-001",
  "status": "success",
  "checks": [
    { "name": "imports", "passed": true },
    { "name": "types", "passed": true },
    { "name": "visual", "passed": true }
  ],
  "metadata": {
    "attempt": 1,
    "source": "hand-written fixture"
  }
}`,
  },
  {
    name: "Bash",
    lang: "bash",
    short: `pnpm --filter @dr-code/gallery dev`,
    long: `set -euo pipefail

workspace="viewer"
package="@dr-code/gallery"

cd "$workspace"
pnpm --filter "$package" typecheck
pnpm --filter "$package" build

echo "Gallery checks completed"`,
  },
];

export const BADGE_FIXTURES: readonly {
  status: StatusBadgeStatus;
  label: string;
}[] = [
  { status: "success", label: "Passed" },
  { status: "failure", label: "Failed" },
  { status: "warning", label: "Needs review" },
  { status: "neutral", label: "Not run" },
];

export const DIFF_MODES: readonly CodeDiffMode[] = ["unified", "split"];

export const CHANGED_DIFF = {
  oldContent: `export function formatName(name: string): string {
  return name.trim();
}`,
  newContent: `export function formatName(name: string): string {
  const normalized = name.trim();
  return normalized || "Anonymous";
}`,
  oldName: "format-name.before.ts",
  newName: "format-name.after.ts",
  lang: "typescript",
} as const;

export const UNCHANGED_DIFF = {
  oldContent: `echo "No changes"`,
  newContent: `echo "No changes"`,
  oldName: "unchanged.before.sh",
  newName: "unchanged.after.sh",
  lang: "bash",
} as const;
