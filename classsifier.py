import numpy as np
import csv
import pandas as pd
import time
import nltk
import re
import pickle
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
dim = 25

# def loadGloveModel(gloveFile):
#     print("Loading Glove Model")
#     f = open(gloveFile,'r')
#     model = {}
#     for line in f:
#         splitLine = line.split()
#         word = splitLine[0]
#         embedding = np.array([float(val) for val in splitLine[1:]])
#         model[word] = embedding
#     print("Done.",len(model)," words loaded!")
#     return model

# def save_obj(obj, name ):
#     with open('obj/'+ name + '.pkl', 'wb') as f:
#         pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

# # start = time.time()
# # table = pd.read_table("27Bx25d.txt", sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)
# # nltk.download()

# word_vec = loadGloveModel("27Bx25d.txt")

# file = open("test.txt")
# text = file.read()
# token = word_tokenize(text)
# token = set(token)
# token = list(token)
# my_word_vec = {}
# count = 0

# for i, tok in enumerate(token):
#     re1 = re.search(r'(\w)+\W(\w)+', tok.lower())
#     re2 = re.search(r'(\w)+',tok.lower())
#     try: 
#         if re1:
#             my_word_vec[re1.group()] = word_vec[re1.group()]
#         elif re2:
#             my_word_vec[re2.group()] = word_vec[re2.group()]  
#     except:
#         rand_vec = np.random.rand(dim,)
#         if re1:
#             my_word_vec[re1.group()] = rand_vec
#         elif re2:
#             my_word_vec[re2.group()] = rand_vec
#         # print("token not found: ", tok)
#         count+=1

# save_obj(my_word_vec, "my_word_vec")

my_word_vec = load_obj("my_word_vec")
# print(my_word_vec['blocked'])

# train_file = open("train.txt")
# train_text = train_file.read()

# print(count)

# print(my_word_vec)



file=open("data/train.txt",'r')
story=file.read()
file.close()
convo= re.findall(r'\"(.+?)\"',story,re.S)
for i,str1 in enumerate(convo):
    # if(story.find("\""+str1+"\"")) ==-1:
    #     print("\""+str1+"\""+"NOT FOUND")

    story=story.replace("\""+str1+"\"","id"+str(i))
for i in range(len(convo)):
    if(convo[i][-1]=="."):
        convo[i]=convo[i][0:-1]+"@"
    convo[i]=convo[i].replace(".","#")
    convo[i]=convo[i].replace("@",".")
i=len(convo)-1
for str1 in reversed(convo):
    story=story.replace("id"+str(i),"\""+str1+"\"")
    i-=1
# print(story)
sentences=re.findall(r'<s>(.+?)</s>',story,re.S)
tokens=[]
for i in range(len(sentences)):
    tokens.append(nltk.word_tokenize(sentences[i]))
X=[]
Y=[]
#print(tokens)
for i in range(len(tokens)):
    for j in range((len(tokens[i]))):

        if(tokens[i][j].find(".")!=-1 and j!=len(tokens[i])-1) and tokens[i][j+1]!="''":
            X.append([i, j])
            Y.append(0)
        elif tokens[i][j].find(".")!=-1 :
            X.append([i, j])
            Y.append(1)


X=np.array(X)
Y=np.array(Y)
# print(X,Y)

# print(X.shape)
# print(X)

win_size = 5
w = np.random.rand(dim,) #intialize model
test = 50

X_final = np.empty((0,dim))
for indices in X:
    total_vec = np.zeros(dim,)
    count = 0
    if tokens[indices[0]][indices[1]] == ".":
        for i in range(win_size):
            index_left = indices[1]-i-1
            if index_left >= 0:
                re1 = re.search(r'(\w)+\W(\w)+', tokens[indices[0]][index_left].lower())
                re2 = re.search(r'(\w)+',tokens[indices[0]][index_left].lower())
                if re1:
                    try:
                        vec = my_word_vec[re1.group()]
                        total_vec = np.add(total_vec, vec)
                        count+=1
                    except:
                        count+=0
                elif re2:
                    try:
                        vec = my_word_vec[re2.group()]
                        total_vec = np.add(total_vec, vec)
                        count+=1
                    except:
                        count+=0
            index_right = indices[1]+i+1

            if index_right < len(tokens[indices[0]]):
                re1 = re.search(r'(\w)+\W(\w)+', tokens[indices[0]][index_right].lower())
                re2 = re.search(r'(\w)+',tokens[indices[0]][index_right].lower())
                if re1:
                    vec = my_word_vec[re1.group()]
                    total_vec = np.add(total_vec, vec)
                    count+=1
                elif re2:
                    vec = my_word_vec[re2.group()]
                    total_vec = np.add(total_vec, vec)
                    count+=1

    else:
        split_token = tokens[indices[0]][indices[1]].split('.')
        i = 0
        j=0
        while(j < len(split_token)):
                i = i+1
                re1 = re.search(r'(\w)+\W(\w)+', split_token[j].lower())
                re2 = re.search(r'(\w)+',split_token[j].lower())
                if re1:
                    try:
                        vec = my_word_vec[re1.group()]
                        total_vec = np.add(total_vec, vec)
                        count+=1

                    except:
                        count+=0
                elif re2:
                    try:
                        vec = my_word_vec[re2.group()]
                        total_vec = np.add(total_vec, vec)
                        count+=1
                    except:
                        count+=0
                j = j+1

        while(i < win_size):
            index_left = indices[1]-i-1
            if index_left >= 0:
                re1 = re.search(r'(\w)+\W(\w)+', tokens[indices[0]][index_left].lower())
                re2 = re.search(r'(\w)+',tokens[indices[0]][index_left].lower())
                if re1:
                    try:
                        vec = my_word_vec[re1.group()]
                        total_vec = np.add(total_vec, vec)
                        count+=1
                    except:
                        count+=0
                elif re2:
                    try:
                        vec = my_word_vec[re2.group()]
                        total_vec = np.add(total_vec, vec)
                        count+=1
                    except:
                        count+=0
            index_right = indices[1]+i+1

            if index_right < len(tokens[indices[0]]):
                re1 = re.search(r'(\w)+\W(\w)+', tokens[indices[0]][index_right].lower())
                re2 = re.search(r'(\w)+',tokens[indices[0]][index_right].lower())
                if re1:
                    vec = my_word_vec[re1.group()]
                    total_vec = np.add(total_vec, vec)
                    count+=1
                elif re2:
                    vec = my_word_vec[re2.group()]
                    total_vec = np.add(total_vec, vec)
                    count+=1

            i+=1


    if count != 0:
        total_vec = np.divide(total_vec, count)

    X_final = np.append(X_final, [total_vec], axis=0)



Y_true = Y
# print(X_final.shape, Y_final.shape)
clf = svm.SVC(class_weight="balanced")
# clf = svm.SVC(kernel='rbf')

X_train, X_test, Y_train, Y_test = train_test_split( X_final, Y_true, test_size=0.33, random_state=42)

try:
    clf.fit(X_train, Y_train)
except:
    print("Can't Classify : only 1 unique label found")
    exit()
Y_pred = clf.predict(X_test)
print("Test : ", Y_test, "\n\nPrediction : ", Y_pred)
print("Accuracy : ", accuracy_score(Y_test, Y_pred))
