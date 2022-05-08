import os
from collections import Counter

import hebrew


def make_table_2():
    folder_to_source = {
        'blogs': r'Blogs \& Forums',
        'books': r"Books \& Forums",
        'dont_panic': r"Books \& Forums",
        'forums': r'Blogs \& Forums',
        'govil': "gov.il",
        'modernTestCorpus': 'DictaTest',
        'news': 'Online Outlets',
        'random_internet': r'Books \& Forums',
        'wiki': 'he.wikipedia',
        'yanshuf': 'Yanshuf',
        'ynet': 'Online Outlets'
    }
    source_to_type = {
        r"Blogs \& Forums": 'User-gen.',
        r"Books \& Forums": 'Literary',
        "Yanshuf": "News",
        'he.wikipedia': "Wiki",
        'gov.il': 'Official',
        'Online Outlets': 'News / Mag',
        'DictaTest': 'Wiki',
    }

    count_words = Counter()
    count_docs = Counter()
    for folder, source in folder_to_source.items():
        path = f'hebrew_diacritized/modern/{folder}'
        tokens = [
            t for t in hebrew.collect_tokens([path])
            if len(t.strip_nonhebrew().items) > 1
        ]
        count_words[source] += len(tokens)
        count_docs[source] += len(os.listdir(path))
    print(r"& Genre & Sources & \# Docs & \# Tokens \\")
    print(r"\midrule")
    for source in count_words:
        print(rf"& {source_to_type[source]} & {source} & {count_docs[source]} & {count_words[source]} \\")
    print(r" \midrule")
    print(rf"& Total &  & {sum(count_docs.values())} & {sum(count_words.values())} \\")
    print(r" \midrule")


if __name__ == '__main__':
    make_table_2()
