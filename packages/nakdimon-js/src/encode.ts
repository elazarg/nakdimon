// Numeric encode/decode between the text layer (HebrewItem / normalized
// chars) and the model's integer vocabularies, per SPEC.inventories.
import { SPEC } from "./spec.gen.js";
import type { HebrewItem } from "./hebrew.js";

export type InventoryName = keyof typeof SPEC.inventories;

interface EncodedItems {
  letters: number[];
  niqqud: number[];
  dagesh: number[];
  sin: number[];
}

/**
 * Builds an encode map by iterating a decode array in order and letting
 * later duplicates overwrite earlier ones. This mirrors v1's
 * dict-overwrite CharacterTable and, for the v1 niqqud table, reproduces
 * the historic patah->15 encode_override for free (patah decodes from
 * index 9 but always encodes to 15, since 15 is written last).
 */
function buildEncodeMap(decode: readonly string[]): Map<string, number> {
  const map = new Map<string, number>();
  decode.forEach((c, i) => map.set(c, i));
  return map;
}

function encodeOne(map: Map<string, number>, c: string, kind: string): number {
  const id = map.get(c);
  if (id === undefined) {
    throw new Error(`unknown ${kind} character: ${JSON.stringify(c)} (U+${(c.codePointAt(0) ?? 0).toString(16)})`);
  }
  return id;
}

export class Inventory {
  readonly name: InventoryName;
  readonly letters: readonly string[];
  readonly niqqud: readonly string[];
  readonly dagesh: readonly string[];
  readonly sin: readonly string[];

  private readonly lettersEncode: Map<string, number>;
  private readonly niqqudEncode: Map<string, number>;
  private readonly dageshEncode: Map<string, number>;
  private readonly sinEncode: Map<string, number>;

  constructor(name: InventoryName) {
    const inv = SPEC.inventories[name];
    this.name = name;
    this.letters = inv.letters;
    this.niqqud = inv.niqqud.decode;
    this.dagesh = inv.dagesh.decode;
    this.sin = inv.sin.decode;

    this.lettersEncode = buildEncodeMap(this.letters);
    this.niqqudEncode = buildEncodeMap(this.niqqud);
    this.dageshEncode = buildEncodeMap(this.dagesh);
    this.sinEncode = buildEncodeMap(this.sin);
  }

  encodeLetters(normalized: string): number[] {
    return [...normalized].map((c) => encodeOne(this.lettersEncode, c, "letter"));
  }

  encodeItems(items: readonly HebrewItem[]): EncodedItems {
    return {
      letters: items.map((it) => encodeOne(this.lettersEncode, it.normalized, "letter")),
      niqqud: items.map((it) => encodeOne(this.niqqudEncode, it.niqqud, "niqqud")),
      dagesh: items.map((it) => encodeOne(this.dageshEncode, it.dagesh, "dagesh")),
      sin: items.map((it) => encodeOne(this.sinEncode, it.sin, "sin")),
    };
  }

  decodeNiqqud(i: number): string {
    return this.niqqud[i];
  }

  decodeDagesh(i: number): string {
    return this.dagesh[i];
  }

  decodeSin(i: number): string {
    return this.sin[i];
  }
}
