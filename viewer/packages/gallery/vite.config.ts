import { fileURLToPath, URL } from "node:url";

import { defineConfig } from "vite";

export default defineConfig({
  resolve: {
    alias: [
      {
        find: "@dr-code/viewer/styles.css",
        replacement: fileURLToPath(
          new URL("../viewer/src/styles.css", import.meta.url),
        ),
      },
      {
        find: /^@dr-code\/viewer$/,
        replacement: fileURLToPath(
          new URL("../viewer/src/index.ts", import.meta.url),
        ),
      },
    ],
  },
});
