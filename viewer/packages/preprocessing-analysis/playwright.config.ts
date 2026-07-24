import { fileURLToPath } from "node:url";

import { defineConfig, devices } from "@playwright/test";

const repositoryRoot = fileURLToPath(new URL("../../..", import.meta.url));
const packageRoot = fileURLToPath(new URL(".", import.meta.url));

export default defineConfig({
  expect: { timeout: 10_000 },
  fullyParallel: false,
  retries: 0,
  testDir: "./e2e",
  workers: 1,
  use: {
    trace: "retain-on-failure",
    ...devices["Desktop Chrome"],
  },
  webServer: [
    {
      command: [
        "uv run python",
        "viewer/packages/preprocessing-analysis/e2e/start-viewer-fixture.py",
      ].join(" "),
      cwd: repositoryRoot,
      gracefulShutdown: { signal: "SIGTERM", timeout: 5_000 },
      name: "viewer fixture",
      reuseExistingServer: false,
      timeout: 120_000,
      wait: {
        stdout: /^DR_CODE_VIEWER_API_URL=(?<DR_CODE_VIEWER_API_URL>http:\/\/127\.0\.0\.1:\d+)$/m,
      },
    },
    {
      command: "node e2e/start-vite.mjs",
      cwd: packageRoot,
      gracefulShutdown: { signal: "SIGTERM", timeout: 5_000 },
      name: "viewer frontend",
      reuseExistingServer: false,
      timeout: 120_000,
      wait: {
        stdout: /^PLAYWRIGHT_TEST_BASE_URL=(?<PLAYWRIGHT_TEST_BASE_URL>http:\/\/127\.0\.0\.1:\d+)$/m,
      },
    },
  ],
});
