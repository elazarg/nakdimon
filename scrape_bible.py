import re
import os
import hebrew

REMOVE_KRI_KTIV = True


def html_to_text(filename):
    with open(filename, 'rb') as f:
        started = False
        for text in f:
            # line = re.sub(b'\xca', b'', text)  #  beresheet text[10201]
            if re.match(b'^<P><B>', text):
                started = True
            if not started:
                continue
            if re.match(b'^<HR>|^<BODY>', text):
                break
            text = re.sub(b'[\xa0]', b'', text)
            text = re.sub(b'[\xca]', bytes('\u05B9', encoding='windows-1255'), text)  # Holam
            text = text.decode('windows-1255')

            # remove HTML tokens
            text = re.sub(r'</?BIG>|</?SMALL>|</?SUP>', '', text)
            text = re.sub(r'<B>[^<]+</B>|<BR>|</?P[^>]*>|</BODY>|</HTML>|<A [^>]*>[^<]*</A>\n?', ' ', text)

            # remove open/close parashiot
            text = re.sub(r'{[^}]+}|\]', ' ', text)

            # remove pisuk
            text = re.sub(r'[,;:.]|--', ' ', text)

            if REMOVE_KRI_KTIV:
                text = re.sub(r'-\) ?', ')-', text)
                text = re.sub(r'(?:[^\u0591-\u05c7 -]{2,}[-]?)*\s+\(([^)]*)\)', r'\1', text)  # \u0591-\u05c7 is the niqqud
            text = re.sub(r'\s\s+', ' ', text)
            text = text.strip()
            if text:
                yield text + '\n'


def run_all():
    chars = set()
    with open('bible_text/bible.txt', 'w', encoding='utf-8') as bible:
        with open('bible_text/bible_undotted.txt', 'w', encoding='utf-8') as undotted:
            for fname in os.listdir('bible_html'):
                if len(fname) > 7:  # only tXX.htm, not tXXYY.htm
                    continue
                text = ''.join(html_to_text('bible_html/' + fname))
                print(fname, ':')
                undotted_text = re.sub(r'[\u0591-\u05c7]', '', text)
                with open('bible_text/' + fname[:-4] + '.txt', 'w', encoding='utf-8') as f:
                    f.write(text)
                bible.write(text)
                chars.update(set(text))
                undotted.write(undotted_text)
    print(sorted(chars))


if __name__ == '__main__':
    run_all()
