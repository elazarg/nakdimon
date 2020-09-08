import os
import random

import nakdimon

base = '../undotted_texts/ynet/'

if __name__ == '__main__':
    files = os.listdir(base)
    filename = base + random.choice(files)
    print(filename)
    nakdimon.diacritize_file(filename)
