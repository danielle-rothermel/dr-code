import { readdir, readFile } from "node:fs/promises";
import { dirname, join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";

export const packageRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
export const canonicalRoot = resolve(
  packageRoot,
  "../../../analysis/preprocessing/generation-corpus-functions-v1-20260719",
);
export const viewerDataSource = resolve(canonicalRoot, "viewer-data.json");
export const viewerDataDestination = resolve(packageRoot, "src/data/viewer-data.json");
export const failureExamplesSource = resolve(canonicalRoot, "failure-examples");
export const failureExamplesDestination = resolve(packageRoot, "public/data/failure-examples");

async function relativeFileList(root) {
  const files = [];

  async function visit(directory) {
    const entries = await readdir(directory, { withFileTypes: true });
    for (const entry of entries) {
      const path = join(directory, entry.name);
      if (entry.isDirectory()) {
        await visit(path);
      } else if (entry.isFile()) {
        files.push(relative(root, path));
      } else {
        throw new Error(`Unsupported entry in viewer data: ${path}`);
      }
    }
  }

  await visit(root);
  return files.sort();
}

export async function assertFileMatches(source, destination) {
  const [sourceBytes, destinationBytes] = await Promise.all([
    readFile(source),
    readFile(destination),
  ]);
  if (!sourceBytes.equals(destinationBytes)) {
    throw new Error(`Packaged viewer data differs from canonical source: ${destination}`);
  }
}

export async function assertDirectoryMatches(source, destination) {
  const [sourceFiles, destinationFiles] = await Promise.all([
    relativeFileList(source),
    relativeFileList(destination),
  ]);
  if (
    sourceFiles.length !== destinationFiles.length ||
    sourceFiles.some((path, index) => path !== destinationFiles[index])
  ) {
    const sourceSet = new Set(sourceFiles);
    const destinationSet = new Set(destinationFiles);
    const missing = sourceFiles.filter((path) => !destinationSet.has(path));
    const stale = destinationFiles.filter((path) => !sourceSet.has(path));
    throw new Error(
      `Packaged failure shard file list differs from canonical source` +
      `\nMissing: ${missing.join(", ") || "none"}` +
      `\nStale: ${stale.join(", ") || "none"}`,
    );
  }

  await Promise.all(
    sourceFiles.map((path) => assertFileMatches(join(source, path), join(destination, path))),
  );
}
