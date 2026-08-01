// Wires the Nakdimon web demo UI to the nakdimon-js v2 runtime.
// `ort` (onnxruntime-web) is loaded as a global via a <script> tag in
// index.html before this module runs.
import { createDiacritizer } from "./lib/index.js";

const MODEL_URL = "./Nakdimon.onnx";

const statusEl = document.getElementById("status");
const inputEl = document.getElementById("input_text");
const outputEl = document.getElementById("output_text");
const dotButton = document.getElementById("dot_button");
const clearButton = document.getElementById("clear_button");
const copyButton = document.getElementById("copy_button");

function setStatus(text, isError = false) {
  statusEl.textContent = text;
  statusEl.classList.toggle("error", isError);
}

let diacritizerPromise = null;

function loadDiacritizer() {
  if (diacritizerPromise) return diacritizerPromise;

  setStatus("טוען את המודל…");
  dotButton.disabled = true;

  diacritizerPromise = (async () => {
    if (typeof ort === "undefined") {
      throw new Error("ort (onnxruntime-web) לא נטען — בדקו את החיבור לאינטרנט.");
    }
    let response;
    try {
      response = await fetch(MODEL_URL, { method: "HEAD" });
    } catch {
      throw new Error(`לא ניתן לגשת לקובץ המודל (${MODEL_URL}).`);
    }
    if (!response.ok) {
      throw new Error(
        `קובץ המודל חסר (${MODEL_URL}). יש להריץ את build.sh כדי להעתיק אותו אל web/.`,
      );
    }
    return createDiacritizer(ort, MODEL_URL);
  })();

  diacritizerPromise
    .then(() => {
      setStatus("המודל נטען. אפשר להתחיל לנקד.");
      dotButton.disabled = false;
    })
    .catch((err) => {
      console.error(err);
      setStatus(String(err.message || err), true);
      diacritizerPromise = null; // allow retry on next click
    });

  return diacritizerPromise;
}

async function onDot() {
  const text = inputEl.value;
  if (!text.trim()) {
    outputEl.value = "";
    return;
  }

  dotButton.disabled = true;
  inputEl.disabled = true;
  setStatus("מנקד…");

  try {
    const diacritizer = await loadDiacritizer();
    const result = await diacritizer.diacritize(text);
    outputEl.value = result;
    setStatus("הניקוד הושלם.");
  } catch (err) {
    console.error(err);
    setStatus(`שגיאה בניקוד: ${err.message || err}`, true);
  } finally {
    inputEl.disabled = false;
    dotButton.disabled = false;
  }
}

function onClear() {
  inputEl.value = "";
  outputEl.value = "";
  inputEl.focus();
}

async function onCopy() {
  const text = outputEl.value;
  if (!text) return;
  try {
    await navigator.clipboard.writeText(text);
    setStatus("הטקסט הועתק ללוח.");
  } catch (err) {
    console.error(err);
    setStatus("העתקה ללוח נכשלה (נדרש חיבור מאובטח HTTPS).", true);
  }
}

dotButton.addEventListener("click", onDot);
clearButton.addEventListener("click", onClear);
copyButton.addEventListener("click", onCopy);

// Start loading the model as soon as the page is ready, so the button is
// usually enabled by the time the user finishes typing.
loadDiacritizer();
