import os
from shutil import copytree

STOP_SIGN = "Stop Sign"
SWIPING_LEFT = "Swiping Left"
SWIPING_RIGHT = "Swiping Right"
SWIPING_DOWN = "Swiping Down"
SWIPING_UP = "Swiping Up"
THUMB_UP = "Thumb Up"
DIR = "./20bn-jester-v1/"
NEW_DIR = "./new_labeles/"

labeles = set()
file = open("labeles.csv", "r")
for line in file:
    labeles.add(line.split("\t")[0])

labeles = list(labeles)
print(len(labeles))

'''
labeles_df = pd.read_csv("labeles.csv", sep="\t", headers=None)
labeles = set(list(labeles_df[1]))
'''

i = 0
for name_dir in os.listdir(DIR):
    if name_dir in labeles:
        i = i + 1
        if i % 100 == 0:
             print(i)

        copytree(DIR + name_dir, NEW_DIR + name_dir)

