const DAGESH = 'ּ';

function to_text(item) {
    return item.char + (item.dagesh || '') + (item.sin || '') + (item.niqqud || '');
}

function update_dotted(items) {
    const dotted_text = document.getElementById("dotted_text");
    dotted_text.innerHTML = '';
    console.log(items);
    for (const item of items) {
        const elem = document.createElement('span');
        elem.textContent = to_text(item);
        elem.addEventListener('dblclick', function (e) {
            item.dagesh = item.dagesh ? '' : DAGESH;
            elem.textContent = to_text(item);
        });
        dotted_text.appendChild(elem);
    }
}