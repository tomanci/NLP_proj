import os 
import pandas as pd
import random as rd

current_dir = os.getcwd()

data = pd.read_csv("/Users/tommasoancilli/Downloads/ita-eng/ita.txt", header=None, delimiter = "\t")

EPSILON = 0.3
MAX_LENGTH = 10

trans_file = open("ita-eng.txt", "w")

for item in range(data.shape[0]):
   text_it = data[1][item]
   text_eng =  data[0][item]

   text_it_split = text_it.split()
   text_eng_split = text_eng.split()

   if len(text_it_split) <= MAX_LENGTH and len(text_eng_split) <= MAX_LENGTH:
      if rd.random() < EPSILON:
         trans_file.write(text_eng + " -> " + text_it + "\n")

trans_file.close()