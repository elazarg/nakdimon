// Core Hebrew text layer: normalization, cleanup, and dotted-text
// decomposition/reconstruction. Driven by spec/hebrew.json (embedded as
// SPEC in spec.gen.ts) — see that file for the semantics this ports.
import { SPEC } from "./spec.gen.js";

/** "category applies but deliberately absent" marker (U+05BF). */
export const RAFE = "ֿ";

const PASS_THROUGH = new Set<string>(SPEC.normalization.pass_through);
const NORMALIZE_MAP: Record<string, string> = SPEC.normalization.map;
const DIGIT_RE = /^\p{Nd}$/u;

const STRIP_CHARS = new Set<string>(SPEC.text_cleanup.strip);

const CAN_DAGESH = new Set<string>([...SPEC.decomposition.can_dagesh]);
const CAN_SIN = new Set<string>([...SPEC.decomposition.can_sin]);
const CAN_NIQQUD = new Set<string>([...SPEC.decomposition.can_niqqud]);

// Combining-mark codepoint ranges used while parsing dotted text.
const DAGESH_LETTER = "ּ"; // also doubles as shuruk on vav
const SHIN_YEMANIT = "ׁ";
const SHIN_SMALIT = "ׂ";
const NIQQUD_LOW = 0x05b0; // shva
const NIQQUD_HIGH = 0x05bb; // kubuts

/** True for the 27 Hebrew base letters (U+05D0-U+05EA). */
function isHebrewLetter(c: string): boolean {
  const cp = c.codePointAt(0) ?? 0;
  return cp >= 0x05d0 && cp <= 0x05ea;
}

export function canDagesh(letter: string): boolean {
  return CAN_DAGESH.has(letter);
}

export function canSin(letter: string): boolean {
  return CAN_SIN.has(letter);
}

export function canNiqqud(letter: string): boolean {
  return CAN_NIQQUD.has(letter);
}

/** Normalize a single character to the model's input vocabulary. */
export function normalizeChar(c: string): string {
  if (PASS_THROUGH.has(c)) return c;
  if (c in NORMALIZE_MAP) return NORMALIZE_MAP[c];
  if (DIGIT_RE.test(c)) return "5";
  return "O";
}

export function normalizeText(text: string): string {
  return [...text].map(normalizeChar).join("");
}

/** Delete characters the spec marks for removal before decomposition/encoding. */
export function cleanText(text: string): string {
  return [...text].filter((c) => !STRIP_CHARS.has(c)).join("");
}

// remove_niqqud.chars is the BODY of a regex character class: a range
// (shva..qamats-qatan) plus a handful of individual marks. It must be
// spliced in as-is — none of its characters are regex metacharacters other
// than the intentional range '-', so no escaping is needed or wanted.
const REMOVE_NIQQUD_RE = new RegExp(`[${SPEC.remove_niqqud.chars}]`, "gu");

export function removeNiqqud(text: string): string {
  return text.replace(REMOVE_NIQQUD_RE, "");
}

export interface HebrewItem {
  letter: string;
  normalized: string;
  dagesh: string;
  sin: string;
  niqqud: string;
}

/**
 * Order-tolerant parser for dotted text. Combining marks immediately
 * following a Hebrew base letter are consumed in ANY order, last-wins per
 * channel; itemsToText re-emits in NFC canonical order (niqqud, dagesh,
 * sin-dot). Legacy v1 corpora use dagesh-sin-niqqud order, which this
 * parser reads equally well (see spec mark_order).
 */
export function decompose(dotted: string): HebrewItem[] {
  const chars = [...dotted];
  const items: HebrewItem[] = [];
  let i = 0;
  while (i < chars.length) {
    const letter = chars[i];
    i++;

    let dagesh = canDagesh(letter) ? RAFE : "";
    let sin = canSin(letter) ? RAFE : "";
    let niqqud = canNiqqud(letter) ? RAFE : "";
    const normalized = normalizeChar(letter);

    if (isHebrewLetter(letter)) {
      while (i < chars.length) {
        const mark = chars[i];
        const cp = mark.codePointAt(0) ?? -1;
        if (mark === DAGESH_LETTER) {
          dagesh = mark;
        } else if (mark === SHIN_YEMANIT || mark === SHIN_SMALIT) {
          sin = mark;
        } else if ((cp >= NIQQUD_LOW && cp <= NIQQUD_HIGH) || mark === RAFE) {
          niqqud = mark;
        } else {
          break;
        }
        i++;
      }
      // vav + shuruk: a dagesh on vav with no explicit niqqud is really the
      // shuruk vowel, not a "real" dagesh.
      if (letter === "ו" && dagesh === DAGESH_LETTER && niqqud === RAFE) {
        dagesh = RAFE;
        niqqud = DAGESH_LETTER;
      }
    }

    items.push({ letter, normalized, dagesh, sin, niqqud });
  }
  return items;
}

export function itemsToText(items: HebrewItem[]): string {
  // NFC canonical emission (spec mark_order): niqqud (ccc 10-20), dagesh (21),
  // sin-dot (24/25) — ascending combining class, so the result is an NFC fixpoint.
  const joined = items.map((it) => it.letter + it.niqqud + it.dagesh + it.sin).join("");
  return joined.replaceAll(RAFE, "");
}
