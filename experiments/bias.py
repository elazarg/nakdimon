import re
import hebrew


def maktel_from_root(root, *, male):
    gender_niqqud = hebrew.Niqqud.SEGOL if male else hebrew.Niqqud.KAMATZ
    if root[0] == 'י':
        MaKTeL = '{mem}{wow}{holam}{r2}{segol_or_kamatz}{h}'
        return MaKTeL.format(
            mem='מ',
            patakh=hebrew.Niqqud.PATAKH,
            wow='ו',
            holam=hebrew.Niqqud.HOLAM,
            r2=root[1],
            segol_or_kamatz=gender_niqqud,
            h='ה'
        )
    else:
        MaKTeL = '{mem}{patakh}{r1}{shva}{r2}{maybe_dagesh}{segol_or_kamatz}{h}'
        return MaKTeL.format(
            mem='מ',
            patakh=hebrew.Niqqud.PATAKH,
            r1=root[0],
            shva=hebrew.Niqqud.SHVA if root[0] not in 'אהחע' else hebrew.Niqqud.REDUCED_PATAKH,
            r2=root[1],
            maybe_dagesh=hebrew.DAGESH_LETTER if root[1] in 'בגדכפת' else '',
            segol_or_kamatz=gender_niqqud,
            h='ה'
        )


def make_definite(word):
    w, *ord = word
    dagesh = '' if w in 'אהחער' else hebrew.DAGESH_LETTER
    niqqud = hebrew.Niqqud.PATAKH if w != 'א' else hebrew.Niqqud.KAMATZ
    return niqqud + w + dagesh + ''.join(ord)


with open('roots.txt', encoding='utf8') as f:
    roots = [line.strip().split('\t')[::-1] for line in f if line.strip()]

with open('../hebrew_diacritized/modern/wiki/5.txt', encoding='utf8') as f:
    text = f.read()

    for root in roots:
        word = maktel_from_root(root, male=True)
        nw = len(re.findall(word, text))
        ndw = len(re.findall(make_definite(word), text))
        if nw or ndw:
            print(word, nw, ndw)
