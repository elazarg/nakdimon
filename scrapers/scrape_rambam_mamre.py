import re
import os

from bs4 import BeautifulSoup

with open('texts/rambam_mamre.txt', 'w', encoding='utf-8') as out:
    for fname in os.listdir('html/rambam_mamre/'):
        with open('html/rambam_mamre/' + fname, 'rb') as f:
            text = re.sub(b'[\xa0]', b'', f.read())
            text = re.sub(b'[\xca]', bytes('\u05B9', encoding='windows-1255'), text)  # Holam
            html_doc = text.decode('windows-1255')

        soup = BeautifulSoup(html_doc, 'html.parser')

        for irrelevant in soup.find_all(['small', 'u', 'title']):
            irrelevant.decompose()

        for s in soup.find_all('b'):
            if '.' not in s.get_text():
                s.decompose()

        text = soup.get_text()

        out.write(text)
        print(file=out)
