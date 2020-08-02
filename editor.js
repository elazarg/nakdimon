const DAGESH = 'ּ';

function to_text(item) {
    return item.char + (item.dagesh || '') + (item.sin || '') + (item.niqqud || '');
}

const itemProto = document.getElementById("itemProto").cloneNode(true);



function update_dotted(items) {
    const dotted_text = document.getElementById("dotted_text");
    dotted_text.innerHTML = '';
    console.log(items);
    for (const item of items) {
        const node = itemProto.cloneNode(true);
        node.textContent = to_text(item);
        function tipcontent(item) {
            const res = document.createElement('div');
            for (const n of niqqud_array) {
                const button = document.createElement('input');
                button.setAttribute('class', "niqqud-button");
                button.setAttribute('type', "button");
                button.setAttribute('value', item.char + item.dagesh + item.sin + n);
                button.addEventListener('onClick', function update_niqqud() {
                    item.niqqud = n;
                    node.textContent = to_text(item);
                });

                res.appendChild(button);
            }
            console.log(res);
            return res;
        }
        if (HEBREW_LETTERS.includes(item.char)) {
            let t = tippy(node, {
                content: tipcontent(item),
                interactive: true,
                trigger: 'click'
            });
            if (!'אחעםןףץ'.includes(item.char)) {
                function listener(e) {
                    item.dagesh = item.dagesh ? '' : DAGESH;
                    node.textContent = to_text(item);
                    t.setProps({'content': tipcontent(item)});
                }
                node.addEventListener('dblclick', listener);
            }
        }
        dotted_text.appendChild(node);
    }

}