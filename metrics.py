from typing import Tuple, List
import hebrew


def get_items(actual: str, expected: str) -> Tuple[List[hebrew.HebrewItem], List[hebrew.HebrewItem]]:
    expected_hebrew = list(hebrew.iterate_dotted_text(expected))
    actual_hebrew = list(hebrew.iterate_dotted_text(actual))
    expected_letters = [c.letter for c in expected_hebrew]
    actual_letters = [c.letter for c in actual_hebrew]
    assert expected_letters == actual_letters
    return actual_hebrew, expected_hebrew


def metric_cha(actual: str, expected: str) -> float:
    actual_hebrew, expected_hebrew = get_items(actual, expected)

    total = sum(hebrew.can_any(c.letter) for c in expected_hebrew)
    actual = sum(x == y for x, y in zip(actual_hebrew, expected_hebrew) if hebrew.can_any(x.letter))
    return actual / total


def metric_dec(actual: str, expected: str) -> float:
    actual_hebrew, expected_hebrew = get_items(actual, expected)

    total = (
        sum(hebrew.can_niqqud(c.letter) for c in expected_hebrew)
      + sum(hebrew.can_dagesh(c.letter) for c in expected_hebrew)
      + sum(hebrew.can_sin(c.letter) for c in expected_hebrew)
    )
    actual = (
        sum(x.niqqud == y.niqqud for x, y in zip(actual_hebrew, expected_hebrew) if hebrew.can_niqqud(x.letter))
      + sum(x.dagesh == y.dagesh for x, y in zip(actual_hebrew, expected_hebrew) if hebrew.can_dagesh(x.letter))
      + sum(x.sin == y.sin for x, y in zip(actual_hebrew, expected_hebrew) if hebrew.can_sin(x.letter))
    )
    return actual / total


def metric_wor(actual: str, expected: str) -> float:
    actual_hebrew, expected_hebrew = get_items(actual, expected)
    actual_hebrew = hebrew.tokenize(actual_hebrew)
    expected_hebrew = hebrew.tokenize(expected_hebrew)

    def is_hebrew(token):
        return len([c for c in token.items if c.letter in hebrew.HEBREW_LETTERS]) > 1

    total = sum(is_hebrew(token) for token in expected_hebrew)
    actual = sum(x == y for x, y in zip(actual_hebrew, expected_hebrew) if is_hebrew(x))
    return actual / total


def read_expected_actual(actual_filename, expected_filename):
    with open(expected_filename, encoding='utf8') as f:
        expected = f.read().strip()

    with open(actual_filename, encoding='utf8') as f:
        actual = f.read().strip()

    return actual, expected


if __name__ == '__main__':
    actual, expected = read_expected_actual('tmp_actual.txt', 'tmp_expected.txt')
    print(metric_cha(actual, expected))
    print(metric_dec(actual, expected))
    print(metric_wor(actual, expected))
