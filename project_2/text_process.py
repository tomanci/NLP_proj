import os 
import pandas as pd
import re
import random as rd
import numpy as np

current_dir = os.getcwd()

path_dataset = "/Users/tommasoancilli/Downloads/ita-eng/ita.txt"

def Text_creation (path_dataset, epsilon = 0.3, max_length = 10):

   data = pd.read_csv(path_dataset, header=None, delimiter = "\t")

   EPSILON = epsilon # decide the fraction of senteces to be included in the dataset
   MAX_LENGTH = max_length  #max length of sentences allowed

   trans_file_eng = open("eng.txt", "w")
   trans_file_ita = open("ita.txt","w")

   for item in range(data.shape[0]):
      text_it = data[1][item]
      text_eng =  data[0][item]

      text_it_split = text_it.split()
      text_eng_split = text_eng.split()

      if len(text_it_split) <= MAX_LENGTH and len(text_eng_split) <= MAX_LENGTH:
         if rd.random() < EPSILON:
            trans_file_eng.write(text_eng + "\n")
            trans_file_ita.write(text_it + "\n")

   trans_file_eng.close()
   trans_file_ita.close()

class Language ():

   def __init__(self, path_name) -> None:

      self.path_name = path_name

      self.word2index = {"EOS":0, "SOS":1}
      self.index2word = {0:"EOS", 1:"SOS"}
      self.n_words = 1 
      self.n_sentences = 0 
      self.n_max_length = 0 

   def normalize_string (self, s) -> str:
      s = re.sub(r"([.!? \, \' \" \% \-])", r" ", s) #remove puntuation 
      s = s.lower() #convert to lower 
      s = s.strip() # remove spaces from the beginning / end 
      s = "SOS" + " " + s + " "+ "EOS"
      return s    

   def add_string(self, s):
      partial_length = 0 
      for word in s.split(" "):
         self.add_word(word)
         partial_length = partial_length + 1
      
      self.n_sentences = self.n_sentences + 1

      if self.n_max_length < partial_length:
         self.n_max_length = partial_length

   def add_word(self, word):
   
      if word not in self.word2index:
         self.word2index[word] = self.n_words
         self.index2word[self.n_words] = word
         self.n_words = self.n_words + 1

   def string_processing (self):
      processed_file = open("processed-"+self.path_name, "w") 

      with open(str(self.path_name)) as file:
         for s in file:
            s = self.normalize_string(s)
            processed_file.write(s +"\n")
            self.add_string(s)
      
      processed_file.close()
   

   #TODO #2:


def Dataset_creation(lang_input, lang_output):
   
   MAX_LENGTH = max(lang_input.n_max_length, lang_output.n_max_length) + 1 # I've -> I ve so I have two words now 

   if lang_input.n_sentences != lang_output.n_sentences:
      raise KeyError ("Error, the number of examples does not match")

   #TODO #1:
   matrix_input = Matrix_creation(lang_input, MAX_LENGTH)
   matrix_output = Matrix_creation(lang_output, MAX_LENGTH)
   
   
def Matrix_creation(lang, MAX_LENGTH):
   matrix  = np.zeros( (lang.n_sentences, MAX_LENGTH) )


   path_input = "processed"+lang.path_name

   try:

      with open(str(path_input), 'r') as file:
         for idx, line in enumerate(file, start = 0):
            words = line.split()
            for id, word in enumerate(words, start = 0 ):
               matrix[idx][id] = lang.word2index[word]

   except Exception as e: #TODO #3
      print(f"{e},{words},{idx},{id}")

   np.save(f'matrix-{re.sub(".txt", "", lang.path_name)}.npy', matrix)
   return matrix


def main ():
   path_dataset = "/Users/tommasoancilli/Downloads/ita-eng/ita.txt"
   max_length = 10 

   Text_creation (path_dataset, epsilon = 0.3, max_length = max_length)
   
   input_lang = Language(path_name = "eng.txt")
   output_lang = Language(path_name = "ita.txt")   

   input_lang.string_processing()
   output_lang.string_processing()

   #Dataset_creation(input_lang, output_lang)

   
if __name__ == "__main__":
   main()
