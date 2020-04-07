const niqqud_array = ['', '', 'ְ', 'ֱ', 'ֲ', 'ֳ', 'ִ', 'ֵ', 'ֶ', 'ַ', 'ָ', 'ֹ', 'ֺ', 'ֻ', 'ּ', 'ַ'];
const dagesh_array = ['', '', 'ּ'];
const sin_array = ['', '', 'ׁ', 'ׂ'];

const HEBREW_LETTERS = ['א', 'ב', 'ג', 'ד', 'ה', 'ו', 'ז', 'ח', 'ט', 'י', 'ך', 'כ', 'ל', 'ם', 'מ', 'ן', 'נ', 'ס', 'ע', 'ף',
'פ', 'ץ', 'צ', 'ק', 'ר', 'ש', 'ת'];
const VALID_LETTERS = [' ', '!', '"', "'", '(', ')', ',', '-', '.', ':', ';', '?'].concat(HEBREW_LETTERS);
const SPECIAL_TOKENS = ['H', 'O', '5'];
const ALL_TOKENS =['', ''].concat(SPECIAL_TOKENS).concat(VALID_LETTERS);
const BATCH_SIZE = 32;
let MAXLEN = 0;

function normalize(c) {
    if (VALID_LETTERS.includes(c)) return c;
    if (c === '\n' || c === '\t') return ' ';
    if (['־', '‒', '–', '—', '―', '−'].includes(c)) return '-';
    if (c === '[') return '(';
    if (c === ']') return ')';
    if (['´', '‘', '’'].includes(c)) return "'";
    if (['“', '”', '״'].includes(c)) return '"';
    if (c.isdigit()) return '5';
    if (c === '…') return ',';
    if (['ײ', 'װ', 'ױ'].includes(c)) return 'H';
    return 'O';
}

function from_categorical(arr, len) {
    return arr.argMax(-1).reshape([1, BATCH_SIZE*MAXLEN]).arraySync()[0].slice(1, len);
}

function text_to_input(text) {
    text = text.replace(/./, normalize);
    text = Array.from(text);
    const ords = text.map(v=>ALL_TOKENS.indexOf(v));
    const input = tf.tensor1d(ords).pad([[1, BATCH_SIZE*MAXLEN - text.length - 1]]).reshape([BATCH_SIZE, MAXLEN]);
    return input;
}

function prediction_to_text(model_output, undotted_text) {
    const [niqqud, dagesh, sin] = model_output;
    const len = undotted_text.length;
    const niqqud_result = from_categorical(niqqud, len);
    const dagesh_result = from_categorical(dagesh, len);
    const sin_result = from_categorical(sin, len);
    let output = '';
    for (let i = 0; i < len; i++) {
        const c = undotted_text[i];
        output += c;
        if (HEBREW_LETTERS.includes(c)) {
            output += niqqud_array[niqqud_result[i]] || '';
            output += dagesh_array[dagesh_result[i]] || '';
            output += sin_array[sin_result[i]] || '';
        }
    }
    return output;
}

async function load_model() {
    const model = await tf.loadLayersModel('model.json');
    MAXLEN = model.input.shape[1];
    function perform_dot(undotted_text) {
        const input = text_to_input(undotted_text);
        const prediction = model.predict(input, {batchSize: BATCH_SIZE});
        return prediction_to_text(prediction, undotted_text);
    }

    document.getElementById("loader").remove();
    document.getElementById("content").style.visibility = 'visible';

    const dotButton = document.getElementById("perform_dot");
    const undotted_text = document.getElementById("undotted_text");
    const dotted_text = document.getElementById("dotted_text");
    dotButton.disabled = false;
    dotButton.textContent = "נקד";
    dotButton.addEventListener("click", (ev) => dotted_text.value = perform_dot(undotted_text.value));
}

load_model();