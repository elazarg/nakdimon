import os
import random

import predict

base = '../undotted_texts/ynet/'

if __name__ == '__main__':
    files = os.listdir(base)
    filename = base + random.choice(files)
    print(filename)
    predict.main(filename)
