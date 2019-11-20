import re
import os

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


def is_text(c):
    return '\u05d0' <= c <= '\u05ea'


def is_niqqud(c):
    return '\u0591' <= c <= '\u05c7'


def iterate_dotted_text(line):
    n = len(line)
    line += '  '
    i = 0
    while i < n:
        dagesh = '_'
        niqqud = '_'
        sin = '_'
        c = line[i]
        i += 1
        if is_text(c):
            if line[i] == '\u05bc':
                dagesh = line[i]
                i += 1
            if line[i] in '\u05c1\u05c2':
                sin = line[i]
                i += 1
            if is_niqqud(line[i]):
                niqqud = line[i]
                i += 1
        yield (c, sin, dagesh, niqqud)


def unzip_dotted_text(line):
    return zip(*iterate_dotted_text(line))


def unzip_dotted_lines(lines):
    ws, xs, ys, zs = [], [], [], []
    for line in lines:
        w, x, y, z = zip(*iterate_dotted_text(line.strip()))
        ws.append(w)
        xs.append(x)
        ys.append(y)
        zs.append(z)
    return ws, xs, ys, zs


def run_all():
    chars = set()
    with open('bible_text/bible.txt', 'w', encoding='utf-8') as bible:
        with open('bible_text/bible_undotted.txt', 'w', encoding='utf-8') as undotted:
            for fname in os.listdir('bible_html'):
                if len(fname) > 7:  # only tXX.htm, not tXXYY.htm
                    continue
                text = ''.join(html_to_text('bible_html/' + fname))
                w, x, y, z = unzip_dotted_text(text)
                print(w)
                print(x)
                print(y)
                print(z)
                undotted_text = re.sub(r'[\u0591-\u05c7]', '', text)
                print(fname, ':', len(text.replace('-', ' ').split()))
                with open('bible_text/' + fname[:-4] + '.txt', 'w', encoding='utf-8') as f:
                    f.write(text)
                bible.write(text)
                chars.update(set(text))
                undotted.write(undotted_text)
    print(sorted(chars))


if __name__ == '__main__':
    run_all()
