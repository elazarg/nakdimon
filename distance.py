import os
import random

import utils

base = '../undotted_texts/ynet/'

files = os.listdir(base)
filename = base + random.choice(files)
print(filename)
with utils.smart_open(filename, encoding='utf-8') as f:
    print(f.read())
