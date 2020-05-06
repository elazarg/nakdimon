import datetime

with open('ynet_ids.txt') as f:
    lines = f.read().split()

print(datetime.datetime.now())

i = 0
for n in lines:
    with open('../../undotted_texts/ynet/{}.txt'.format(n)):
        i += 1
        print(i, end='\r')

print()
print(datetime.datetime.now())
