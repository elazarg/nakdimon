const DAGESH = 'ּ';
const SIN = 'ׁ';
const SHIN = 'ׂ';
const SHURUK = 'ּ';
const PLAIN_NIQQUD = ['ְ',   'ֳ', 'ֲ', 'ֱ', null, '', 'ָ', 'ַ', 'ֶ', 'ֵ', 'ֹ', 'ֻ', 'ִ', null];
const GARON_NIQQUD = [null, 'ֳ', 'ֲ', 'ֱ', null, '', 'ָ', 'ַ', 'ֶ', 'ֵ', 'ֹ', 'ֻ', 'ִ', null];
const VAV_NIQQUD   = ['ְ',   'ֳ', 'ֲ', 'ֱ', null, '', 'ָ', 'ַ', 'ֶ', 'ֵ', 'ֹ', 'ֻ', 'ִ', 'ּ'];
const KHAF_SOFIT_NIQQUD = ['', 'ְ', 'ָ'];
const TAF_SOFIT_NIQQUD = ['', 'ְ'];
const GARON_SOFIT_NIQQUD = ['', 'ַ'];

function to_text(item) {
    return item.char + (item.dagesh || '') + (item.sin || '') + (item.niqqud || '');
}

const itemProto = document.getElementById("itemProto").cloneNode(true);
itemProto.removeAttribute("id");


function update_dotted(items) {
    const dotted_text = document.getElementById("dotted_text");
    dotted_text.innerHTML = '';
    console.log(items);
    for (const [i, item] of items.entries()) {
        const node = itemProto.cloneNode(true);
        node.textContent = to_text(item);
        function tipcontent(t, item) {
            let whitespace_seen = false;
            let future_hebrew = false;
            for (let j=i+1; j < items.length; j++) {
                const q = items[j].char;
                if (q === " ")
                    whitespace_seen = true;
                if (HEBREW_LETTERS.includes(q)) {
                    future_hebrew = true
                    break;
                }
            }
            const sofit = whitespace_seen || !future_hebrew;

            const res = document.createElement('div');
            const iterable = sofit && 'אהחע'.includes(item.char) ? GARON_SOFIT_NIQQUD
                           : sofit && 'ת' === item.char ? TAF_SOFIT_NIQQUD
                           : item.char === 'ך' ? KHAF_SOFIT_NIQQUD
                           : item.char === 'ו' ? VAV_NIQQUD
                           : 'אהחע'.includes(item.char) ? GARON_NIQQUD
                           : PLAIN_NIQQUD;
            for (const n of iterable) {
                const button = document.createElement('input');
                button.setAttribute('type', "button");
                if (n !== null) {
                    button.setAttribute('class', "niqqud-button");
                    button.setAttribute('value', item.char + item.dagesh + item.sin + n);
                    button.addEventListener('click', function () {
                        item.niqqud = n;
                        node.textContent = to_text(item);
                        t.hide();
                    });
                } else {
                    button.setAttribute('class', "niqqud-button empty");
                    button.setAttribute('value', " ");
                }

                res.appendChild(button);
            }
            return res;
        }
        if (HEBREW_LETTERS.includes(item.char)) {
            if (!'םןףץ'.includes(item.char)) {
                let t = tippy(node, {
                    interactive: true,
                    // interactiveBorder: 30,
                    maxWidth: 210,
                    trigger: 'click'
                });
                t.setContent(tipcontent(t, item));
                if (!'אחע'.includes(item.char)) {
                    function toggle_dagesh(e) {
                        item.dagesh = item.dagesh ? '' : DAGESH;
                        node.textContent = to_text(item);
                        t.setContent(tipcontent(t, item));
                    }
                    node.addEventListener('contextmenu', toggle_dagesh);
                }
                if (item.char === 'ש') {
                    function toggle_sin(e) {
                        item.sin = item.sin === SIN ? SHIN : SIN;
                        node.textContent = to_text(item);
                        t.setContent(tipcontent(t, item));
                    }
                    node.addEventListener('dblclick', toggle_sin);
                }
            }
        }
        dotted_text.appendChild(node);
    }

}