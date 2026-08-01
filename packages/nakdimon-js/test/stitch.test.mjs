// Center-stitched windowed inference against spec/golden/stitch.json.
// The fake session spec lives in spec/tools/generate_stitch_golden.py; it is
// deliberately re-implemented here (as in the Python test) so this runtime's
// test stands alone against the golden.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

import { Diacritizer } from "../dist/runtime.js";
import { removeNiqqud } from "../dist/hebrew.js";

const here = dirname(fileURLToPath(import.meta.url));
const golden = JSON.parse(readFileSync(join(here, "..", "..", "..", "spec", "golden", "stitch.json"), "utf8"));

class FakeTensor {
  constructor(type, data, dims) {
    this.type = type;
    this.data = data;
    this.dims = dims;
  }
}

function oneHot(rows, cols, target, classes) {
  const data = new Float32Array(rows * cols * classes);
  for (let b = 0; b < rows; b++) {
    for (let t = 0; t < cols; t++) {
      data[(b * cols + t) * classes + target(b, t)] = 1.0;
    }
  }
  return { data, dims: [rows, cols, classes] };
}

const fakeSession = {
  inputNames: ["input"],
  async run(feeds) {
    const x = feeds.input;
    const [rows, cols] = x.dims;
    const id = (b, t) => x.data[b * cols + t];
    return {
      N: oneHot(rows, cols, (b, t) => (id(b, t) + b + t) % 15, 15),
      D: oneHot(rows, cols, (b, t) => (id(b, t) + b + 2 * t) % 3, 3),
      S: oneHot(rows, cols, (b, t) => (id(b, t) + 2 * b + t) % 4, 4),
    };
  },
};

const card = {
  inventory: golden.card.inventory,
  window: golden.card.window,
  padToWindow: golden.card.pad_to_window,
  overlap: golden.card.overlap,
};

test("stitch.json: center-stitched inference matches golden byte-for-byte", async () => {
  const diacritizer = new Diacritizer(fakeSession, FakeTensor, card);
  for (const c of golden.cases) {
    const out = await diacritizer.diacritize(c.text);
    assert.strictEqual(out, c.output, `stitched output for ${c.text.length}-char input`);
    // Alignment contract: stripping marks recovers the input letters exactly.
    assert.strictEqual(removeNiqqud(out), c.text);
    // Emission is an NFC fixpoint.
    assert.strictEqual(out.normalize("NFC"), out);
  }
});
