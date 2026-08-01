import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

import {
  normalizeText,
  removeNiqqud,
  decompose,
  itemsToText,
  Inventory,
  SPEC,
} from "../dist/index.js";

const here = path.dirname(fileURLToPath(import.meta.url));
const goldenDir = path.resolve(here, "..", "..", "..", "spec", "golden");

function readGolden(name) {
  return JSON.parse(readFileSync(path.join(goldenDir, name), "utf8"));
}

test("normalize.json: normalizeText matches golden", () => {
  const cases = readGolden("normalize.json");
  for (const { input, normalized } of cases) {
    assert.strictEqual(normalizeText(input), normalized, `normalize(${JSON.stringify(input)})`);
  }
});

test("remove_niqqud.json: removeNiqqud matches golden", () => {
  const cases = readGolden("remove_niqqud.json");
  for (const { input, input_nfc, output } of cases) {
    assert.strictEqual(removeNiqqud(input), output, `removeNiqqud(${JSON.stringify(input)})`);
    assert.strictEqual(removeNiqqud(input_nfc), output, `removeNiqqud(nfc ${JSON.stringify(input_nfc)})`);
  }
});

test("decompose.json: decompose/itemsToText/encode match golden byte-for-byte", () => {
  const cases = readGolden("decompose.json");
  const v1 = new Inventory("v1");
  for (const c of cases) {
    const items = decompose(c.dotted);
    const asTuples = items.map((it) => [it.letter, it.normalized, it.dagesh, it.sin, it.niqqud]);
    assert.deepStrictEqual(asTuples, c.items, `items for ${JSON.stringify(c.dotted)}`);

    assert.strictEqual(itemsToText(items), c.reconstructed, `itemsToText for ${JSON.stringify(c.dotted)}`);

    // Order tolerance: NFC-ordered input must decompose to the very same items.
    assert.deepStrictEqual(decompose(c.dotted_nfc), items, `nfc items for ${JSON.stringify(c.dotted_nfc)}`);
    // Emission is an NFC fixpoint: re-parsing our own output is the identity.
    assert.strictEqual(itemsToText(decompose(c.reconstructed)), c.reconstructed, `fixpoint for ${JSON.stringify(c.reconstructed)}`);

    const encoded = v1.encodeItems(items);
    assert.deepStrictEqual(encoded.niqqud, c.niqqud_ids, `niqqud_ids for ${JSON.stringify(c.dotted)}`);
    assert.deepStrictEqual(encoded.dagesh, c.dagesh_ids, `dagesh_ids for ${JSON.stringify(c.dotted)}`);
    assert.deepStrictEqual(encoded.sin, c.sin_ids, `sin_ids for ${JSON.stringify(c.dotted)}`);
  }
});

test("encode_v1.json: encodeLetters(normalized_flat) matches golden ids", () => {
  const cases = readGolden("encode_v1.json");
  const v1 = new Inventory("v1");
  for (const c of cases) {
    assert.deepStrictEqual(
      v1.encodeLetters(c.normalized_flat),
      c.ids,
      `encodeLetters for ${JSON.stringify(c.text)}`,
    );
  }
});

test("v1 niqqud encode_overrides are realized by decode-array iteration order", () => {
  const v1 = new Inventory("v1");
  const overrides = SPEC.inventories.v1.niqqud.encode_overrides;
  for (const [char, expectedId] of Object.entries(overrides)) {
    const items = [{ letter: "א", normalized: "א", dagesh: "", sin: "", niqqud: char }];
    const { niqqud } = v1.encodeItems(items);
    assert.strictEqual(niqqud[0], expectedId, `encode override for ${JSON.stringify(char)}`);
  }
});
