// Thin ONNX wrapper around the text layer. Deliberately has no hard
// dependency on onnxruntime-web/onnxruntime-node: callers inject a session
// and a Tensor constructor that satisfy the structural types below, so this
// file (and `tsc`) works offline and in either a browser or Node host.
import { RAFE, canDagesh, canNiqqud, canSin, cleanText, decompose, type HebrewItem } from "./hebrew.js";
import { Inventory, type InventoryName } from "./encode.js";

export interface OrtTensorLike {
  data: ArrayLike<number>;
  dims: number[];
}

export interface OrtSessionLike {
  inputNames: string[];
  run(feeds: Record<string, unknown>): Promise<Record<string, OrtTensorLike>>;
}

export type TensorCtor = new (type: string, data: Int32Array, dims: number[]) => unknown;

export interface ModelCard {
  inventory?: InventoryName;
  window?: number;
  padToWindow?: boolean;
  overlap?: number;
}

const DEFAULT_CARD: Required<ModelCard> = {
  inventory: "v1",
  window: 10000,
  padToWindow: true,
  overlap: 0,
};

/** Start offsets of overlapping windows covering [0, n); last window right-aligned. */
export function windowStarts(n: number, window: number, overlap: number): number[] {
  if (n <= window) return [0];
  const stride = window - overlap;
  const starts: number[] = [];
  for (let s = 0; s < n - window; s += stride) starts.push(s);
  starts.push(n - window);
  return starts;
}

/**
 * Exclusive end of each window's owned region: consecutive windows meet at the
 * integer floor midpoint of their overlap, so every position is predicted by
 * the window in which it is most central. Keep identical to the Python port.
 */
export function ownershipCuts(starts: number[], window: number, n: number): number[] {
  const cuts = starts.slice(0, -1).map((s, k) => Math.floor((starts[k + 1] + s + window) / 2));
  cuts.push(n);
  return cuts;
}

/**
 * Ports nakdimon/hebrew.py's split_by_length exactly: greedily fills `out`
 * up to maxlen-1 items, splitting at the last space seen so far rather than
 * mid-word. If no space was seen before maxlen-1 is reached, the whole
 * (word-exceeding-maxlen) run is yielded as-is.
 *
 * `space` is intentionally NEVER reset to the `maxlen` sentinel after a
 * yield — only `is_space` updates it. This means it is often *stale*
 * (an index into a `out` that has since been sliced away) for one or more
 * chunks following a split that wasn't triggered by a space. That
 * staleness is load-bearing: reproducing it exactly (by porting the loop
 * structure verbatim rather than "fixing" it) is required to match v1's
 * chunk boundaries byte-for-byte.
 */
export function splitByLength<T extends { letter: string }>(items: readonly T[], maxlen: number): T[][] {
  if (maxlen <= 1) throw new Error("maxlen must be > 1");
  const chunks: T[][] = [];
  let out: T[] = [];
  let space = maxlen;
  for (const c of items) {
    if (c.letter === " ") space = out.length;
    out.push(c);
    if (out.length === maxlen - 1) {
      chunks.push(out.slice(0, space + 1));
      out = out.slice(space + 1);
    }
  }
  if (out.length > 0) chunks.push(out);
  return chunks;
}

/** argmax over the last axis of a row-major tensor; returns one winning class index per leading-axis "row". */
function argmaxLastAxis(t: OrtTensorLike): Int32Array {
  const classes = t.dims[t.dims.length - 1];
  const rows = t.data.length / classes;
  const out = new Int32Array(rows);
  for (let r = 0; r < rows; r++) {
    const base = r * classes;
    let best = 0;
    let bestVal = -Infinity;
    for (let c = 0; c < classes; c++) {
      const v = t.data[base + c];
      if (v > bestVal) {
        bestVal = v;
        best = c;
      }
    }
    out[r] = best;
  }
  return out;
}

export class Diacritizer {
  private readonly session: OrtSessionLike;
  private readonly Tensor: TensorCtor;
  private readonly inventory: Inventory;
  private readonly window: number;
  private readonly padToWindow: boolean;
  private readonly overlap: number;

  constructor(session: OrtSessionLike, Tensor: TensorCtor, card: ModelCard = {}) {
    const merged = { ...DEFAULT_CARD, ...card };
    this.session = session;
    this.Tensor = Tensor;
    this.inventory = new Inventory(merged.inventory);
    this.window = merged.window;
    this.padToWindow = merged.padToWindow;
    this.overlap = merged.overlap;
  }

  private async runArgmax(inputData: Int32Array, rows: number, width: number) {
    const tensor = new this.Tensor("int32", inputData, [rows, width]);
    const feeds: Record<string, unknown> = { [this.session.inputNames[0]]: tensor };
    const results = await this.session.run(feeds);
    // The model's three output heads are returned in insertion order,
    // matching the ONNX graph's declared output order: niqqud, dagesh, sin.
    const [nOut, dOut, sOut] = Object.values(results);
    return { niqqudIds: argmaxLastAxis(nOut), dageshIds: argmaxLastAxis(dOut), sinIds: argmaxLastAxis(sOut) };
  }

  private emit(item: HebrewItem, niqqudId: number, dageshId: number, sinId: number): string {
    // NFC canonical emission order: niqqud, dagesh, sin-dot (spec mark_order).
    let s = item.letter;
    if (canNiqqud(item.letter)) s += this.inventory.decodeNiqqud(niqqudId);
    if (canDagesh(item.letter)) s += this.inventory.decodeDagesh(dageshId);
    if (canSin(item.letter)) s += this.inventory.decodeSin(sinId);
    return s;
  }

  async diacritize(text: string): Promise<string> {
    const items = decompose(cleanText(text));
    if (items.length === 0) return "";
    if (!this.padToWindow) return this.diacritizeStitched(items);
    return this.diacritizeV1(items);
  }

  /**
   * v2 inference: overlapping char-aligned windows, each position predicted by
   * the window where it is most central (see Python runtime._diacritize_stitched).
   */
  private async diacritizeStitched(items: HebrewItem[]): Promise<string> {
    if (!(this.overlap >= 0 && this.overlap < this.window)) throw new Error("model card: need 0 <= overlap < window");
    const ids = this.inventory.encodeLetters(items.map((it) => it.normalized).join(""));
    const n = ids.length;
    const starts = windowStarts(n, this.window, this.overlap);
    // Rows are always full window width (zero-padded): keeps the input shape
    // fixed so trace-exported models with a baked sequence length work too.
    const width = this.window;
    const inputData = new Int32Array(starts.length * width);
    starts.forEach((s, row) => {
      const len = Math.min(width, n - s);
      for (let i = 0; i < len; i++) inputData[row * width + i] = ids[s + i];
    });

    const { niqqudIds, dageshIds, sinIds } = await this.runArgmax(inputData, starts.length, width);

    const cuts = ownershipCuts(starts, this.window, n);
    const out: string[] = [];
    let ownStart = 0;
    starts.forEach((s, row) => {
      for (let pos = ownStart; pos < cuts[row]; pos++) {
        const flat = row * width + (pos - s);
        out.push(this.emit(items[pos], niqqudIds[flat], dageshIds[flat], sinIds[flat]));
      }
      ownStart = cuts[row];
    });
    return out.join("").replaceAll(RAFE, "");
  }

  private async diacritizeV1(items: HebrewItem[]): Promise<string> {
    const chunks = splitByLength(items, this.window);

    // v1 pads every chunk to the full window width; the bidirectional pass reads
    // padding, so full-width padding is behaviorally load-bearing.
    const padLen = this.window;
    const nChunks = chunks.length;
    const inputData = new Int32Array(nChunks * padLen);
    chunks.forEach((chunk, row) => {
      const ids = this.inventory.encodeLetters(chunk.map((it) => it.normalized).join(""));
      const base = row * padLen;
      ids.forEach((id, col) => {
        inputData[base + col] = id;
      });
      // remaining columns stay 0 (mask/padding token) via Int32Array's zero-init
    });

    const { niqqudIds, dageshIds, sinIds } = await this.runArgmax(inputData, nChunks, padLen);

    const perChunk: string[] = chunks.map((chunk, row) => {
      const base = row * padLen;
      const parts: string[] = [];
      chunk.forEach((item, col) => {
        parts.push(this.emit(item, niqqudIds[base + col], dageshIds[base + col], sinIds[base + col]));
      });
      return parts.join("");
    });

    // v1's predict() does " ".join(chunks) then a single left-to-right pass
    // collapsing every doubled space introduced at chunk boundaries.
    // Python's str.replace already replaces all non-overlapping
    // occurrences in one pass; String.prototype.replaceAll is the exact JS
    // equivalent (not a loop to a fixed point).
    return perChunk.join(" ").replaceAll("  ", " ").replaceAll(RAFE, "");
  }
}

export interface OrtModule {
  InferenceSession: { create(pathOrUrl: string): Promise<OrtSessionLike> };
  Tensor: TensorCtor;
}

/** Convenience wiring for an injected onnxruntime-web (or -node) module. */
export async function createDiacritizer(ort: OrtModule, modelUrl: string, card?: ModelCard): Promise<Diacritizer> {
  const session = await ort.InferenceSession.create(modelUrl);
  return new Diacritizer(session, ort.Tensor, card);
}
