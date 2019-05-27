import pandas as pd
import glob
import csv
import unicodedata
import pickle

characters = []
for filename in glob.glob('datasets/*.csv'):
    print(filename)
    with open(filename, encoding='utf-8') as csvfile:
        data = csv.DictReader(csvfile, fieldnames=["s1", "s2", "res", "id1", "id2", "l1", "l2", "i1", "i2", "d"], delimiter='|')
        for row in data:
            characters += list(bytearray(unicodedata.normalize('NFKD', row['s1'] + row['s2']), encoding='utf-8'))
            characters = list(set(characters))
            #characters = ''.join(list(set(characters)))

#characters = ''.join(list(set(characters)))
#characters = unicodedata.normalize('NFKD', characters)
characters = ['PAD'] + characters
print(characters)
print(len(characters))

with open("characters.pkl", 'wb') as char:
    pickle.dump(characters, char)
