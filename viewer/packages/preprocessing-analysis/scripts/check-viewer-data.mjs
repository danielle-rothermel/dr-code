import {
  assertDirectoryMatches,
  assertFileMatches,
  failureExamplesDestination,
  failureExamplesSource,
  viewerDataDestination,
  viewerDataSource,
} from "./viewer-data-files.mjs";

await Promise.all([
  assertFileMatches(viewerDataSource, viewerDataDestination),
  assertDirectoryMatches(failureExamplesSource, failureExamplesDestination),
]);

console.log("Packaged preprocessing viewer data matches the canonical analysis artifacts");
