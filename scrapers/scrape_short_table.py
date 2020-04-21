from bs4 import BeautifulSoup
import re

with open('short_table/short_table.html', encoding='windows-1255') as f:
    html_doc = f.read()
soup = BeautifulSoup(html_doc, 'html.parser')

for irrelevant in soup.find_all(['small', 'u']):
    irrelevant.decompose()

for s in soup.find_all('b'):
    if '.' not in s.get_text():
        s.decompose()

text = soup.get_text()
text = '\n'.join(text.split('\n')[620:-2])

text = re.sub(',', '', text)
text = re.sub(' [\u05d0-\u05ea)]+. ', ' ', text)
text = re.sub(r'\.', '\n', text)

text = '\n'.join(' '.join(x.strip().split()) for x in text.split('\n') if x.strip())
# text = re.sub(r' \.', '.', text)
# text = re.sub(' ?\n[ \n]*', '\n', text)
# text = re.sub('\n[^\u0591-\u05c7]+\n', '\n', text)
# text = re.sub('\([^\u0591-\u05c7)]+\)', '\n', text)
# pat = "סעיף [א-ת]+'"
# text = re.sub(pat + '\s*', '', text)
# text = re.sub('\n+', '\n', text)
# print(text)

with open('short_table/short_table.txt', 'w', encoding='utf-8') as f:
    f.write(text)
