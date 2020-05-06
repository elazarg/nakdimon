import sys

from bs4 import BeautifulSoup
import requests


def rel_link(rel):
    return 'https://www.ynet.co.il' + rel


def fetch(url):
    response = requests.get(rel_link(url))
    html_doc = response.content.decode('utf-8')
    return BeautifulSoup(html_doc, 'lxml')


MONTHS = {'ינואר', 'פברואר', 'מרץ', 'אפריל', 'מאי', 'יוני', 'יולי', 'אוגוסט', 'ספטמבר', 'אוקטובר', 'נובמבר', 'דצמבר'}

main_links = [(s.text, s['href']) for s in fetch('/home/0,7340,L-4269,00.html').find_all('a', {"class": "CSH"})]

with open('ynet_links.txt', 'a') as out:
    for h, link in main_links:
        sub_links = [(s.text, s['href']) for s in fetch(link).find_all('a', {"class": "smallheader"})]
        print(''.join(reversed(h)))
        i = 0
        for m, sub_link in sub_links:
            # print(m, end=' ')
            targets = [(s.text, s['href']) for s in fetch(sub_link).find_all('a', {"class": "smallheader"}) if s.text.strip() not in MONTHS]
            for title, target in targets:
                print(rel_link(target), file=out)
                i += 1
                print('\r', i, end='')
                sys.stdout.flush()
        print()
