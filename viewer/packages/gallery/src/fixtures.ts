/**
 * Hand-written fixture data for the gallery. Lives here, not in
 * `@dr-code/viewer` — the viewer package stays fixture-free (see
 * REDESIGN.md step 3).
 */
import type { BadgeStatus, CodeDiffMode } from "@dr-code/viewer";

export interface CodeBlockFixture {
  title: string;
  description: string;
  code: string;
  lang: string;
}

const PYTHON_SHORT = `def double(x):
    return x * 2
`;

const PYTHON_LONG = `class LruCache:
    """A tiny least-recently-used cache backed by an OrderedDict."""

    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._capacity = capacity
        self._store: dict[str, object] = {}
        self._order: list[str] = []

    def get(self, key: str) -> object | None:
        if key not in self._store:
            return None
        self._touch(key)
        return self._store[key]

    def put(self, key: str, value: object) -> None:
        if key in self._store:
            self._store[key] = value
            self._touch(key)
            return
        if len(self._store) >= self._capacity:
            oldest = self._order.pop(0)
            del self._store[oldest]
        self._store[key] = value
        self._order.append(key)

    def _touch(self, key: str) -> None:
        self._order.remove(key)
        self._order.append(key)
`;

const TYPESCRIPT_SHORT = `export function double(x: number): number {
  return x * 2;
}
`;

const TYPESCRIPT_LONG = `export interface RetryOptions {
  attempts: number;
  delayMs: number;
  shouldRetry?: (error: unknown) => boolean;
}

export async function withRetry<T>(
  fn: () => Promise<T>,
  { attempts, delayMs, shouldRetry = () => true }: RetryOptions,
): Promise<T> {
  let lastError: unknown;
  for (let attempt = 1; attempt <= attempts; attempt++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error;
      if (attempt === attempts || !shouldRetry(error)) {
        throw error;
      }
      await new Promise((resolve) => setTimeout(resolve, delayMs * attempt));
    }
  }
  throw lastError;
}
`;

const JSON_SHORT = `{
  "name": "double",
  "arity": 1
}
`;

const JSON_LONG = `{
  "name": "@dr-code/gallery",
  "primitives": ["CodeBlock", "CodeDiff", "StatusBadge"],
  "fixtures": {
    "codeBlock": {
      "languages": ["python", "typescript", "json"],
      "sizes": ["short", "long"]
    },
    "codeDiff": {
      "modes": ["split", "unified"],
      "cases": ["real-change", "identical-content"]
    }
  },
  "themes": ["light", "dark"],
  "generated": false
}
`;

export const CODE_BLOCK_FIXTURES: CodeBlockFixture[] = [
  {
    title: "CodeBlock",
    description: "python, short",
    code: PYTHON_SHORT,
    lang: "python",
  },
  {
    title: "CodeBlock",
    description: "python, long",
    code: PYTHON_LONG,
    lang: "python",
  },
  {
    title: "CodeBlock",
    description: "typescript, short",
    code: TYPESCRIPT_SHORT,
    lang: "typescript",
  },
  {
    title: "CodeBlock",
    description: "typescript, long",
    code: TYPESCRIPT_LONG,
    lang: "typescript",
  },
  {
    title: "CodeBlock",
    description: "json, short",
    code: JSON_SHORT,
    lang: "json",
  },
  {
    title: "CodeBlock",
    description: "json, long",
    code: JSON_LONG,
    lang: "json",
  },
];

export interface CodeDiffFixture {
  title: string;
  description: string;
  oldContent: string;
  newContent: string;
  oldName: string;
  newName: string;
  lang: string;
  mode: CodeDiffMode;
}

const PYTHON_DIFF_OLD = `def greet(name):
    return "Hello, " + name
`;

const PYTHON_DIFF_NEW = `def greet(name, excited=False):
    greeting = "Hello, " + name
    if excited:
        greeting += "!"
    return greeting
`;

const TYPESCRIPT_DIFF_OLD = `export function add(a: number, b: number): number {
  return a + b;
}
`;

const TYPESCRIPT_DIFF_NEW = `export function add(a: number, b: number, c = 0): number {
  return a + b + c;
}
`;

const JSON_DIFF_OLD = `{
  "retries": 1,
  "timeoutMs": 500
}
`;

const JSON_DIFF_NEW = `{
  "retries": 3,
  "timeoutMs": 1500,
  "backoff": "exponential"
}
`;

const IDENTICAL_CONTENT = `export const VERSION = "1.0.0";
`;

export const CODE_DIFF_FIXTURES: CodeDiffFixture[] = [
  {
    title: "CodeDiff",
    description: "python, real change, unified",
    oldContent: PYTHON_DIFF_OLD,
    newContent: PYTHON_DIFF_NEW,
    oldName: "greet.py",
    newName: "greet.py",
    lang: "python",
    mode: "unified",
  },
  {
    title: "CodeDiff",
    description: "python, real change, split",
    oldContent: PYTHON_DIFF_OLD,
    newContent: PYTHON_DIFF_NEW,
    oldName: "greet.py",
    newName: "greet.py",
    lang: "python",
    mode: "split",
  },
  {
    title: "CodeDiff",
    description: "typescript, real change, split",
    oldContent: TYPESCRIPT_DIFF_OLD,
    newContent: TYPESCRIPT_DIFF_NEW,
    oldName: "add.ts",
    newName: "add.ts",
    lang: "typescript",
    mode: "split",
  },
  {
    title: "CodeDiff",
    description: "json, real change, unified",
    oldContent: JSON_DIFF_OLD,
    newContent: JSON_DIFF_NEW,
    oldName: "config.json",
    newName: "config.json",
    lang: "json",
    mode: "unified",
  },
  {
    title: "CodeDiff",
    description: "identical content (empty diff), unified",
    oldContent: IDENTICAL_CONTENT,
    newContent: IDENTICAL_CONTENT,
    oldName: "version.ts",
    newName: "version.ts",
    lang: "typescript",
    mode: "unified",
  },
  {
    title: "CodeDiff",
    description: "identical content (empty diff), split",
    oldContent: IDENTICAL_CONTENT,
    newContent: IDENTICAL_CONTENT,
    oldName: "version.ts",
    newName: "version.ts",
    lang: "typescript",
    mode: "split",
  },
];

export interface StatusBadgeFixture {
  title: string;
  description: string;
  status: BadgeStatus;
  label: string;
}

export const STATUS_BADGE_FIXTURES: StatusBadgeFixture[] = [
  {
    title: "StatusBadge",
    description: "status: positive",
    status: "positive",
    label: "passed",
  },
  {
    title: "StatusBadge",
    description: "status: negative",
    status: "negative",
    label: "failed",
  },
  {
    title: "StatusBadge",
    description: "status: warning",
    status: "warning",
    label: "flaky",
  },
  {
    title: "StatusBadge",
    description: "status: neutral (default)",
    status: "neutral",
    label: "pending",
  },
];
