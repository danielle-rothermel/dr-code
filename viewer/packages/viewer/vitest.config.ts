import { defineConfig } from "vitest/config";

export default defineConfig({
  esbuild: { jsx: "automatic" },
  test: {
    environment: "jsdom",
    include: ["tests/**/*.test.tsx"],
    setupFiles: ["tests/setup.ts"],
    server: {
      deps: {
        // react-shiki's dist imports its own CSS; inline it so vite
        // transforms the import instead of node choking on .css.
        inline: ["react-shiki"],
      },
    },
  },
});
