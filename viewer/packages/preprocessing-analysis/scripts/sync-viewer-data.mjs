import { cp, mkdir, rm } from "node:fs/promises";
import { dirname } from "node:path";

import {
  failureExamplesDestination,
  failureExamplesSource,
  viewerDataDestination,
  viewerDataSource,
} from "./viewer-data-files.mjs";

await mkdir(dirname(viewerDataDestination), { recursive: true });
await cp(viewerDataSource, viewerDataDestination);
await rm(failureExamplesDestination, { force: true, recursive: true });
await mkdir(dirname(failureExamplesDestination), { recursive: true });
await cp(failureExamplesSource, failureExamplesDestination, { recursive: true });
console.log(`Synced ${viewerDataSource} to ${viewerDataDestination}`);
console.log(`Synced ${failureExamplesSource} to ${failureExamplesDestination}`);
