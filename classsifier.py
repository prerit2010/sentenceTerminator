
import numpy as np
import csv
import pandas as pd
import time
import nltk
import re
import pickle
from nltk.tokenize import sent_tokenize, word_tokenize

dim = 25

def loadGloveModel(gloveFile):
    print("Loading Glove Model")
    f = open(gloveFile,'r')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print("Done.",len(model)," words loaded!")
    return model

def save_obj(obj, name ):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

# # start = time.time()
# # table = pd.read_table("27Bx25d.txt", sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)
# # nltk.download()

word_vec = loadGloveModel("27Bx25d.txt")

file = open("test.txt")
text = file.read()
token = word_tokenize(text)
token = set(token)
token = list(token)
my_word_vec = {}
count = 0

for i, tok in enumerate(token):
    re1 = re.search(r'(\w)+\W(\w)+', tok.lower())
    re2 = re.search(r'(\w)+',tok.lower())
    try: 
        if re1:
            my_word_vec[re1.group()] = word_vec[re1.group()]
        elif re2:
            my_word_vec[re2.group()] = word_vec[re2.group()]  
    except:
        rand_vec = np.random.rand(dim,)
        if re1:
            my_word_vec[re1.group()] = rand_vec
        elif re2:
            my_word_vec[re2.group()] = rand_vec
        # print("token not found: ", tok)
        count+=1

save_obj(my_word_vec, "my_word_vec")

my_word_vec = load_obj("my_word_vec")
# print(my_word_vec['blocked'])

# train_file = open("train.txt")
# train_text = train_file.read()

# print(count)

print(my_word_vec)








