import os
import random

import naqdan

base = '../undotted_texts/ynet/'

if __name__ == '__main__':
    files = os.listdir(base)
    filename = base + random.choice(files)
    print(filename)
    naqdan.diacritize_file(filename)
