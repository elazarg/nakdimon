const letters_array = ['^', '@', '', ' ', '!', '"', "'", '(', ')', '*', ',', '-', '.', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '?', 'D', 'O', 'R', '[', ']', '´', 'ְ', 'ַ', 'ֹ', '־', 'ׁ', 'א', 'ב', 'ג', 'ד', 'ה', 'ו', 'ז', 'ח', 'ט', 'י', 'ך', 'כ', 'ל', 'ם', 'מ', 'ן', 'נ', 'ס', 'ע', 'ף', 'פ', 'ץ', 'צ', 'ק', 'ר', 'ש', 'ת', 'ײ', '״', '–', '—', '‘', '’', '“', '”', '…', '−', '\uf04a'];
const niqqud_array = ['^', '@', '', 'ְ', 'ֱ', 'ֲ', 'ֳ', 'ִ', 'ֵ', 'ֶ', 'ַ', 'ָ', 'ֹ', 'ֺ', 'ֻ', 'ּ'];
const dagesh_array = ['^', '@', '', 'ּ'];
const sin = ['^', '@', '', 'ׁ', 'ׂ'];

function from_categorical(arr, len) {
    return arr.argMax(-1).reshape([1, 32*60]).arraySync()[0].slice(1, len);
}

function text_to_input(text) {
    text = Array.from(text);
    const ords = text.map(v=>letters_array.indexOf(v));
    const input = tf.tensor1d(ords).pad([[1, 32*60 - text.length - 1]]).reshape([32, 60]);
    return input;
}

function prediction_to_text(model_output, undotted_text) {
    const [niqqud, dagesh] = model_output;
    const len = undotted_text.length;
    const niqqud_result = from_categorical(niqqud, len);
    const dagesh_result = from_categorical(dagesh, len);
    let output = '';
    for (let i = 0; i < len; i++) {
        output += undotted_text[i];
        if (undotted_text[i] !== '\n') {
            output += niqqud_array[niqqud_result[i]] || '';
            output += dagesh_array[dagesh_result[i]] || '';
        }
    }
    return output;
}

async function load_model() {
    const model = await tf.loadLayersModel('model.json');
    function perform_dot(undotted_text) {
        const input = text_to_input(undotted_text);
        const prediction = model.predict(input, {batchSize: 32});
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