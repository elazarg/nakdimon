const niqqud_array = ['', '', 'ְ', 'ֱ', 'ֲ', 'ֳ', 'ִ', 'ֵ', 'ֶ', 'ַ', 'ָ', 'ֹ', 'ֺ', 'ֻ', 'ּ', 'ַ'];
const dagesh_array = ['', '', 'ּ'];
const sin_array = ['', '', 'ׁ', 'ׂ'];

const HEBREW_LETTERS = ['א', 'ב', 'ג', 'ד', 'ה', 'ו', 'ז', 'ח', 'ט', 'י', 'ך', 'כ', 'ל', 'ם', 'מ', 'ן', 'נ', 'ס', 'ע', 'ף',
'פ', 'ץ', 'צ', 'ק', 'ר', 'ש', 'ת'];
const VALID_LETTERS = [' ', '!', '"', "'", '(', ')', ',', '-', '.', ':', ';', '?'].concat(HEBREW_LETTERS);
const SPECIAL_TOKENS = ['H', 'O', '5'];
const ALL_TOKENS =['', ''].concat(SPECIAL_TOKENS).concat(VALID_LETTERS);
const BATCH_SIZE = 32;
let MAXLEN = null;

function normalize(c) {
    if (VALID_LETTERS.includes(c)) return c;
    if (c === '\n' || c === '\t') return ' ';
    if (['־', '‒', '–', '—', '―', '−'].includes(c)) return '-';
    if (c === '[') return '(';
    if (c === ']') return ')';
    if (['´', '‘', '’'].includes(c)) return "'";
    if (['“', '”', '״'].includes(c)) return '"';
    if ('0123456789'.includes(c)) return '5';
    if (c === '…') return ',';
    if (['ײ', 'װ', 'ױ'].includes(c)) return 'H';
    return 'O';
}

function split_to_rows(text) {
    const space = ALL_TOKENS.indexOf(" ");
    const arr = text.split(" ").map(s => Array.from(s).map(c => ALL_TOKENS.indexOf(c)));
    let line = [];
    const rows = [line];
    for (let i=0; i < arr.length; i++) {
        if (arr[i].length + line.length + 1> MAXLEN) {
            while (line.length < MAXLEN)
                line.push(0);
            line = [];
            rows.push(line);
        }
        line.push(...arr[i]);
        line.push(space);
    }
    while (line.length < MAXLEN)
        line.push(0);
    console.log(rows);
    return rows;
}

function text_to_input(text) {
    const ords = Array.from(text).map(c => ALL_TOKENS.indexOf(normalize(c)));
    return  tf.tensor1d(ords).pad([[0, BATCH_SIZE*MAXLEN - text.length]]).reshape([BATCH_SIZE, MAXLEN]);
}

function prediction_to_text(input, model_output, undotted_text) {
        
    function from_categorical(arr, len) {
        return arr.argMax(-1).reshape([-1]).arraySync().filter((x, i) => input[i]);
    }

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

function remove_niqqud(text) {
    return text.replace(/[\u0591-\u05C7]/g, '');
}

async function load_model() {
    const bar = new Nanobar();
    const model = await tf.loadLayersModel('model.json', {onProgress: (fraction) => { bar.go(100 * fraction); } });
    model.summary();
    MAXLEN = model.input.shape[1];


    function perform_dot(undotted_text) {
        undotted_text = remove_niqqud(undotted_text);
        const input = split_to_rows(undotted_text);
        const prediction = model.predict(tf.tensor2d(input), {batchSize: 32});
        return prediction_to_text([].concat(... input), prediction, undotted_text);
    }

    const dotButton = document.getElementById("perform_dot");
    const undotted_text = document.getElementById("undotted_text");
    const dotted_text = document.getElementById("dotted_text");
    dotButton.disabled = false;
    dotButton.textContent = "נקד";
    dotButton.addEventListener("click", function (ev) {
        dotted_text.value = perform_dot(undotted_text.value);
    });
}
