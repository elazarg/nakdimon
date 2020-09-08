from experiments import external_apis
import hebrew
from experiments.metrics import *
# import nakdimon

SYSTEMS = {
    # 'Snopi': external_apis.fetch_snopi,  # Too slow
    # 'Morfix': external_apis.fetch_morfix,  # terms-of-use issue
    'Nakdan': external_apis.fetch_dicta,
    # r'\sysname{}': nakdimon.call_nakdimon
}


def format_latex(all_results):
    for system, results in all_results.items():
        print('{system} & {cha:.2%} & {dec:.2%} & {wor:.2%} & {voc:.2%} \\\\'.format(system=system, **results)
              .replace('%', ''))


def collect_metrics(actual, expected):
    return {
        'cha': metric_cha(actual, expected),
        'dec': metric_dec(actual, expected),
        'wor': metric_wor(actual, expected),
        'voc': metric_wor(actual, expected, vocalize=True)
    }


def run(filename):
    with open(filename, 'r', encoding='utf8') as f:
        expected = f.read()
    cleaned = hebrew.remove_niqqud(expected)
    all_results = {system: collect_metrics(api(cleaned), expected)
                   for system, api in SYSTEMS.items()}
    format_latex(all_results)


if __name__ == '__main__':
    run('test/hillel.txt')
