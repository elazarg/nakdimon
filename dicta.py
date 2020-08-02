import requests

METEG = '\u05BD'
KUBUTZ = '\u05BB'
SHURUK = '\u05BC'
HOLAM = '\u05b9'

url = 'https://nakdan-2-0.loadbalancer.dicta.org.il/api'


def extract_word(k):
    if k['options']:
        res = k['options'][0][0]
        res = res.replace('|', '')
        res = res.replace(KUBUTZ + 'ו' + METEG, 'ו' + SHURUK)
        res = res.replace(HOLAM + 'ו' + METEG, 'ו' + HOLAM)
        res = res.replace(METEG, '')
        return res
    return k['word']


def call_dicta(data):
    payload = {
        "task": "nakdan",
        "genre": "modern",
        "data": data,
        "addmorph": True,
        "keepqq": False,
        "nodageshdefmem": False,
        "patachma": False,
        "keepmetagim": True,
    }

    r = requests.post(url, json=payload, headers={'content-type': 'text/plain;charset=UTF-8'})
    return ''.join(extract_word(k) for k in r.json())


data = 'מוקדם יותר היום, צה"ל ערך טקס חילופי מפקדים באוגדת עזה, כשהמפקד היוצא, תת אלוף אליעזר טולדנו, אמר כי "לחמאס לא אכפת מילדי הרצועה. עליהם לחדול מלחפור מנהרות כחיות השדה ולהשקיע את הבטון בבניית תשתיות לחסרי בית. עליהם לחדול מלייצר רקטות ולהשתמש בצינורות להקמת תשתית מים וביוב ראויה לילדיהם. עליהם לחדול מלזרוע משגרים ולקצור רקטות ולזרוע חיטה ולקצור תבואה".'
print(call_dicta(data))
