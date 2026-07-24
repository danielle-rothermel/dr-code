import { spawn } from "node:child_process";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { createServer, type Server } from "node:http";
import { createRequire } from "node:module";
import { tmpdir } from "node:os";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";

import { expect, test } from "@playwright/test";

const HOST = "127.0.0.1";
const LEGACY_BACKEND_PORT = 8011;
const LEGACY_FRONTEND_PORT = 4173;
const PROBE_TAG = "@isolated-endpoint-probe";
const INJECTED_FAILURE = "injected first-attempt failure";
const packageRoot = fileURLToPath(new URL("..", import.meta.url));
const playwrightCli = createRequire(import.meta.url).resolve(
  "@playwright/test/cli",
);

interface EndpointRecord {
  backendURL: string;
  frontendURL: string;
}

interface RunResult {
  code: number | null;
  output: string;
}

function endpointRecord(raw: string): EndpointRecord {
  const parsed: unknown = JSON.parse(raw);
  if (
    typeof parsed !== "object"
    || parsed === null
    || !("backendURL" in parsed)
    || typeof parsed.backendURL !== "string"
    || !("frontendURL" in parsed)
    || typeof parsed.frontendURL !== "string"
  ) {
    throw new Error(`Invalid endpoint record: ${raw}`);
  }
  return parsed as EndpointRecord;
}

async function runProbe(
  temporaryRoot: string,
  name: string,
  extraEnvironment: NodeJS.ProcessEnv = {},
): Promise<RunResult> {
  const child = spawn(
    process.execPath,
    [
      playwrightCli,
      "test",
      "--config",
      path.join(packageRoot, "playwright.config.ts"),
      "--grep",
      PROBE_TAG,
      "--reporter=line",
      "--output",
      path.join(temporaryRoot, `${name}-results`),
    ],
    {
      cwd: packageRoot,
      env: {
        ...process.env,
        ...extraEnvironment,
        DR_CODE_ENDPOINT_RECORD: path.join(temporaryRoot, `${name}.json`),
      },
      stdio: ["ignore", "pipe", "pipe"],
    },
  );

  let output = "";
  child.stdout.setEncoding("utf8");
  child.stderr.setEncoding("utf8");
  child.stdout.on("data", (chunk: string) => {
    output += chunk;
  });
  child.stderr.on("data", (chunk: string) => {
    output += chunk;
  });

  return await new Promise<RunResult>((resolve, reject) => {
    child.once("error", reject);
    child.once("close", (code) => resolve({ code, output }));
  });
}

async function startDecoy(port: number): Promise<Server | null> {
  const server = createServer((_request, response) => {
    response.statusCode = 418;
    response.end("stale fixed-port decoy");
  });
  return await new Promise<Server | null>((resolve, reject) => {
    server.once("error", (error: NodeJS.ErrnoException) => {
      if (error.code === "EADDRINUSE") {
        resolve(null);
        return;
      }
      reject(error);
    });
    server.listen(port, HOST, () => resolve(server));
  });
}

async function closeServer(server: Server | null): Promise<void> {
  if (server === null) return;
  await new Promise<void>((resolve, reject) => {
    server.close((error) => {
      if (error) reject(error);
      else resolve();
    });
  });
}

test.setTimeout(180_000);

test(`frontend proxies to its captured backend ${PROBE_TAG}`, async ({
  request,
}, testInfo) => {
  const backendURL = process.env.DR_CODE_VIEWER_API_URL;
  const frontendURL = process.env.PLAYWRIGHT_TEST_BASE_URL;
  if (!backendURL || !frontendURL) {
    throw new Error("Playwright did not capture both isolated server URLs");
  }
  expect(new URL(backendURL).hostname).toBe(HOST);
  expect(new URL(frontendURL).hostname).toBe(HOST);

  const response = await request.get("/api/annotations/export");
  expect(response.ok()).toBe(true);
  expect(await response.json()).toEqual([]);

  const recordPath = process.env.DR_CODE_ENDPOINT_RECORD;
  if (recordPath) {
    await writeFile(
      recordPath,
      JSON.stringify({ backendURL, frontendURL } satisfies EndpointRecord),
      "utf8",
    );
  }

  if (process.env.DR_CODE_INJECT_FIRST_ATTEMPT_FAILURE && testInfo.retry === 0) {
    throw new Error(INJECTED_FAILURE);
  }
});

test("fixed-port decoys cannot intercept concurrent invocations", async () => {
  const temporaryRoot = await mkdtemp(
    path.join(tmpdir(), "dr-code-playwright-isolation-"),
  );
  const decoys: Array<Server | null> = [];

  try {
    decoys.push(await startDecoy(LEGACY_BACKEND_PORT));
    decoys.push(await startDecoy(LEGACY_FRONTEND_PORT));
    const results = await Promise.all([
      runProbe(temporaryRoot, "first"),
      runProbe(temporaryRoot, "second"),
    ]);
    for (const result of results) {
      expect(result.code, result.output).toBe(0);
    }

    const records = await Promise.all(
      ["first", "second"].map(async (name) => endpointRecord(
        await readFile(path.join(temporaryRoot, `${name}.json`), "utf8"),
      )),
    );
    expect(new Set(records.map(({ backendURL }) => backendURL)).size).toBe(2);
    expect(new Set(records.map(({ frontendURL }) => frontendURL)).size).toBe(2);
    for (const record of records) {
      expect(new URL(record.backendURL).port).not.toBe(
        String(LEGACY_BACKEND_PORT),
      );
      expect(new URL(record.frontendURL).port).not.toBe(
        String(LEGACY_FRONTEND_PORT),
      );
    }
  } finally {
    await Promise.all(decoys.map(closeServer));
    await rm(temporaryRoot, { force: true, recursive: true });
  }
});

test("an injected first-attempt failure fails the invocation", async () => {
  const temporaryRoot = await mkdtemp(
    path.join(tmpdir(), "dr-code-playwright-no-retry-"),
  );
  try {
    const result = await runProbe(temporaryRoot, "no-retry", {
      DR_CODE_INJECT_FIRST_ATTEMPT_FAILURE: "1",
    });
    expect(result.output).toContain(INJECTED_FAILURE);
    expect(result.code, result.output).not.toBe(0);
  } finally {
    await rm(temporaryRoot, { force: true, recursive: true });
  }
});
