import requests
import json

from hebrew import Niqqud


def call_snopi(text: str) -> str:
    url = 'http://www.nakdan.com/GetResult.aspx'

    payload = {
        "txt": text,
        "ktivmale": 'false',
    }
    headers = {
        'Referer': 'http://www.nakdan.com/nakdan.aspx',
    }

    r = requests.post(url, data=payload, headers=headers)
    return r.text.split('Result')[1][1:-2]


def call_morfix(text: str) -> str:
    url = 'https://nakdan.morfix.co.il/nikud/NikudText'

    payload = {
        "text": text,
        "isLogged": 'false',
    }
    headers = {
    }

    r = requests.post(url, data=payload, headers=headers)
    return json.loads(r.json()['nikud'])['OutputText']


def call_dicta(text: str) -> str:
    def extract_word(k):
        if k['options']:
            res = k['options'][0][0]
            res = res.replace('|', '')
            res = res.replace(Niqqud.KUBUTZ + 'ו' + Niqqud.METEG, 'ו' + Niqqud.SHURUK)
            res = res.replace(Niqqud.HOLAM + 'ו' + Niqqud.METEG, 'ו' + Niqqud.HOLAM)
            res = res.replace(Niqqud.METEG, '')
            return res
        return k['word']

    # TODO: split by 10,000

    url = 'https://nakdan-2-0.loadbalancer.dicta.org.il/api'

    payload = {
        "task": "nakdan",
        "genre": "modern",
        "data": text,
        "addmorph": True,
        "keepqq": False,
        "nodageshdefmem": False,
        "patachma": False,
        "keepmetagim": True,
    }
    headers = {
        'content-type': 'text/plain;charset=UTF-8'
    }

    r = requests.post(url, json=payload, headers=headers)
    return ''.join(extract_word(k) for k in r.json())


if __name__ == '__main__':
    text = 'מוקדם יותר היום, צה"ל ערך טקס חילופי מפקדים באוגדת עזה, כשהמפקד היוצא, תת אלוף אליעזר טולדנו, אמר כי "לחמאס לא אכפת מילדי הרצועה. עליהם לחדול מלחפור מנהרות כחיות השדה ולהשקיע את הבטון בבניית תשתיות לחסרי בית. עליהם לחדול מלייצר רקטות ולהשתמש בצינורות להקמת תשתית מים וביוב ראויה לילדיהם. עליהם לחדול מלזרוע משגרים ולקצור רקטות ולזרוע חיטה ולקצור תבואה".'
    print(call_dicta(text))
