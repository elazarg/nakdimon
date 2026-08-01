import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

import { SPEC } from "../dist/index.js";

const here = path.dirname(fileURLToPath(import.meta.url));
const specPath = path.resolve(here, "..", "..", "..", "spec", "hebrew.json");

test("embedded SPEC matches spec/hebrew.json exactly (run `npm run embed-spec` if this fails)", () => {
  const fromDisk = JSON.parse(readFileSync(specPath, "utf8"));
  assert.deepStrictEqual(SPEC, fromDisk);
});
