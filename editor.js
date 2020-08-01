
function update_dotted(items) {
    const dotted_text = document.getElementById("dotted_text");
    dotted_text.innerHTML = '';
    console.log(items);
    for (const item of items) {
        const elem = document.createElement('span');
        elem.textContent = item.char + (item.dagesh || '') + (item.sin || '') + (item.niqqud || '');
        dotted_text.appendChild(elem);
    }
}