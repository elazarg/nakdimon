from concurrent.futures import ThreadPoolExecutor
import logging

from bs4 import BeautifulSoup
import requests

import utils


def get(url):
    with requests.Session() as session:
        session.max_redirects = 1
        try:
            response = session.get(url, allow_redirects=False)
            response.raise_for_status()
        except requests.RequestException as rx:
            logging.error('failed to fetch {}: {}'.format(url, rx))
            return None
        return response.content.decode('utf-8')


def output(html, out_filename):
    soup = BeautifulSoup(html, 'lxml')
    with utils.smart_open(out_filename, 'w', encoding='utf-8') as f:
        for p in soup.find_all('p'):
            bold = p.find('font', attrs={'style': 'FONT-WEIGHT: bold; FONT-SIZE: 13px;'})
            if bold and bold.text.strip().endswith(':'):
                bold.decompose()
            if p.find('img'):
                continue
            text = p.text
            if text:
                print(text, file=f)


def fetch(ynet_id, out_filename=None):
    url = 'https://www.ynet.co.il/articles/0,7340,L-{},00.html'.format(ynet_id)
    print(url)
    html = get(url)
    if html is None:
        return
    out_filename = out_filename or '../undotted_texts/ynet/{}.txt'.format(ynet_id)
    output(html, out_filename)


def run_parallel(n=None):
    with open('scrapers/ynet_ids.txt') as f:
        if n is not None:
            lines = [next(f).strip() for x in range(n)]
        else:
            lines = f.read().split()

    logging.basicConfig(filename='ynet.log', filemode='w', level=logging.INFO, format='%(asctime)s %(message)s')
    logging.info('start fetching {} pages'.format(len(lines)))

    with ThreadPoolExecutor(max_workers=16) as executor:
        executor.map(fetch, lines)

    logging.info('end')


if __name__ == '__main__':
    fetch('5265624', '-')
