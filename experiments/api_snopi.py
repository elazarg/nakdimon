import requests


def call_snopi(text: str):
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


if __name__ == '__main__':
    print(call_snopi("שלום לכם"))
