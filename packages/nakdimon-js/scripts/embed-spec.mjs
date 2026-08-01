// Reads ../../../spec/hebrew.json and writes it into src/spec.gen.ts as a
// typed const, so the package has no runtime dependency on the filesystem
// (or on the location of the monorepo) once built.
import { readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

const here = path.dirname(fileURLToPath(import.meta.url));
const pkgRoot = path.resolve(here, "..");
const specPath = path.resolve(pkgRoot, "..", "..", "spec", "hebrew.json");
const outPath = path.join(pkgRoot, "src", "spec.gen.ts");

const spec = JSON.parse(readFileSync(specPath, "utf8"));

const banner = `// GENERATED from spec/hebrew.json — do not edit. Run npm run embed-spec.\n`;
const body = `export const SPEC = ${JSON.stringify(spec, null, 2)} as const;\n`;

writeFileSync(outPath, banner + body, "utf8");
console.log(`wrote ${path.relative(pkgRoot, outPath)} from ${path.relative(pkgRoot, specPath)}`);
