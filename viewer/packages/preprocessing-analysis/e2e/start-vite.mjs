import process from "node:process";

import { createServer } from "vite";

const HOST = "127.0.0.1";

if (!process.env.DR_CODE_VIEWER_API_URL) {
  throw new Error("DR_CODE_VIEWER_API_URL was not captured from the fixture server");
}

const server = await createServer({
  server: {
    host: HOST,
    strictPort: true,
  },
});

const httpServer = server.httpServer;
if (!httpServer) {
  await server.close();
  throw new Error("Vite did not create an HTTP server");
}
await new Promise((resolve, reject) => {
  httpServer.once("error", reject);
  httpServer.once("listening", resolve);
  // Vite normalizes port 0 to its default in server.listen(). Binding its
  // owned HTTP server directly preserves the kernel's atomic port assignment.
  httpServer.listen(0, HOST);
});

const address = httpServer.address();
if (!address || typeof address === "string") {
  await server.close();
  throw new Error("Vite did not expose an IPv4 TCP listener");
}

console.log(`PLAYWRIGHT_TEST_BASE_URL=http://${HOST}:${address.port}`);

let closing = false;
const close = async () => {
  if (closing) return;
  closing = true;
  await server.close();
};
process.once("SIGINT", close);
process.once("SIGTERM", close);
await new Promise((resolve) => httpServer.once("close", resolve));
