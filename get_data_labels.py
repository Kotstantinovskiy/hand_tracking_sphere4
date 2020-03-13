import os
from shutil import copytree

STOP_SIGN = "Stop Sign"
SWIPING_LEFT = "Swiping Left"
SWIPING_RIGHT = "Swiping Right"
SWIPING_DOWN = "Swiping Down"
SWIPING_UP = "Swiping Up"
THUMB_UP = "Thumb Up"
DIR = "./20bn-jester-v1/"
NEW_DIR = "./new_lables/"

lables = set()
file = open("lables.csv", "r")
for line in file:
    lables.add(line.split("\t")[0])

lables = list(lables)
print(len(lables))

'''
lables_df = pd.read_csv("lables.csv", sep="\t", headers=None)
lables = set(list(lables_df[1]))
'''

i = 0
for name_dir in os.listdir(DIR):
    if name_dir in lables:
        i = i + 1
        if i % 100 == 0:
             print(i)

        copytree(DIR + name_dir, NEW_DIR + name_dir)

