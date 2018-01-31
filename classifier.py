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
dim = 200
win_size = 2

def loadGloveModel(gloveFile):
    print("Loading vectors")
    f = open(gloveFile,'r')
    model = {}
    for line in f:
        # split the lines to make a list
        splitLine = line.split()
        # store the first element as a word
        word = splitLine[0]
        # store the corresponding vector as embedding
        embedding = np.array([float(val) for val in splitLine[1:]])
        # store the key value pair of word and vector in a dictionary
        model[word] = embedding
    print("Done.",len(model)," words loaded!")
    return model

def save_obj(obj, name):
    # Save the word-vector dictionary in a pickle file
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, protocol=2)

def load_obj(name):
    # Load the word-vector dictionary from the pickle file 
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def create_word_vec(dimension):
    # obtain the word-vector dictionary.
    word_vec = loadGloveModel("data/27Bx" + str(dimension) + "d.txt")

    # Create a shorter dictionary with vocabulary only of the words in our data
    file = open("text/test2.txt")
    text = file.read()
    token = word_tokenize(text)
    # Unique words
    token = set(token)
    token = list(token)
    my_word_vec = {}
    # iterate over the tokens
    for i, tok in enumerate(token):
        # Cleaning the tokens by geting rid of the punctuations marks.
        re1 = re.search(r'(\w)+\W(\w)+', tok.lower())
        re2 = re.search(r'(\w)+',tok.lower())
        try: 
            if re1:
                my_word_vec[re1.group()] = word_vec[re1.group()]
            elif re2:
                my_word_vec[re2.group()] = word_vec[re2.group()]  
        except:
            # if the word-vector is not found in the dictionary, assign it a random vector
            rand_vec = np.random.rand(dim,)
            if re1:
                my_word_vec[re1.group()] = rand_vec
            elif re2:
                my_word_vec[re2.group()] = rand_vec
            # print("token not found: ", tok)

    # Save the word-vector dictionary in a pickle file
    save_obj(my_word_vec, "my_word_vec" + str(dimension))

# for creating our short word-vector dictionary.
# create_word_vec(dim)    

# load the dictionary from the pickle file
my_word_vec = load_obj("my_word_vec" + str(dim))


file =open("output/train.txt",'r')
text = file.read()
file.close()
# find the quoted text from the text file
quoted = re.findall(r'\"(.+?)\"',text,re.S)
for i,mystr in enumerate(quoted):
    # Replace the quoted strings with unique ids.
    text = text.replace("\""+mystr+"\"","id"+str(i))

for i in range(len(quoted)):
    if quoted[i][-1] == ".":
        # append @ if found .
        quoted[i] = quoted[i][0:-1] + "@"
    if quoted[i][-1] == "?":
        # append $ if found ?
        quoted[i] = quoted[i][0:-1] + "$"

    #repace ? and . inside the quoted text with something else to avoid confusion
    quoted[i] = quoted[i].replace("?","&")
    quoted[i] = quoted[i].replace(".","#")
    quoted[i] = quoted[i].replace("$","?")
    quoted[i] = quoted[i].replace("@",".")

i=len(quoted)-1
for mystr in reversed(quoted):
    # replace the text with unique ids
    text = text.replace("id"+str(i),"\""+mystr+"\"")
    i -= 1

text = re.sub(r'\?(\s*\"*\s*</s>)',r'??\1',text)
# find the sentences using the inserted tags
tagged_line = re.findall(r'<s>(.+?)</s>',text,re.S)
tokenized_line = []
# iterate over the sentences and tokenize them
for i in range(len(tagged_line)):
    tokenized_line.append(nltk.word_tokenize(tagged_line[i]))
X=[]
Y=[]

j=0
# iterate over the tokenized sentences
for i in range(len(tokenized_line)):
    while j < (len(tokenized_line[i])):
        # Assign a label 0 in case . is not a sentence terminator
        if(tokenized_line[i][j].find(".") != -1 and j != len(tokenized_line[i])-1) and \
            tokenized_line[i][j+1] != "''":
            X.append([i, j])
            Y.append(0)
        # Assign a label 1 in case . is a sentence terminator
        elif tokenized_line[i][j].find(".")!=-1 :
            X.append([i, j])
            Y.append(1)

        # Assign a label 0 in case ? is not a sentence terminator
        if (tokenized_line[i][j].find("?") != -1 and j != len(tokenized_line[i])-1) and \
            tokenized_line[i][j] == "?" and tokenized_line[i][j+1] != "?":
            X.append([i, j])
            Y.append(0)

        # Assign a label 1 in case ? is a sentence terminator
        elif (tokenized_line[i][j].find("?") != -1 and j != len(tokenized_line[i])-1) and \
            tokenized_line[i][j] == "?" and  tokenized_line[i][j+1]=="?":
            X.append([i, j])
            Y.append(1)
            j = j+1

        j = j+1
    j=0
X=np.array(X)
Y=np.array(Y)


X_final = np.empty((0,dim))
#iterate over the indices of . and ?
for indices in X:
    #initialize a vector with all zeros
    total_vec = np.zeros(dim,)
    count = 0
    if tokenized_line[indices[0]][indices[1]] == "." or "?":
        # Iterate over the window size near the . or ?
        for i in range(win_size):
            # indices towards the left of the token
            index_left = indices[1]-i-1
            if index_left >= 0:
                # Cleaning the tokens by geting rid of the punctuations marks.
                re1 = re.search(r'(\w)+\W(\w)+', tokenized_line[indices[0]][index_left].lower())
                re2 = re.search(r'(\w)+',tokenized_line[indices[0]][index_left].lower())
                if re1:
                    try:
                        # storing the vector corresponing to the word
                        vec = my_word_vec[re1.group()]
                        # Addding the vector to the initialized vector of zeroes
                        total_vec = np.add(total_vec, vec)
                        # Maintainig a count so as to take the average in the end.
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

            # going over the right side of the current token
            if index_right < len(tokenized_line[indices[0]]):
                # Cleaning the tokens by geting rid of the punctuations marks.
                re1 = re.search(r'(\w)+\W(\w)+', tokenized_line[indices[0]][index_right].lower())
                re2 = re.search(r'(\w)+',tokenized_line[indices[0]][index_right].lower())
                if re1:
                    try:
                        # storing the vector corresponing to the word
                        vec = my_word_vec[re1.group()]
                        # Addding the vector to the initialized vector of zeroes
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

    else:
        # split the token if the token is not just .
        split_token = tokenized_line[indices[0]][indices[1]].split('.')
        i = 0
        j=0
        # iterate over the spit token
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

        # Iterate over the window size
        while(i < win_size):
            index_left = indices[1]-i-1
            if index_left >= 0:
                re1 = re.search(r'(\w)+\W(\w)+', tokenized_line[indices[0]][index_left].lower())
                re2 = re.search(r'(\w)+',tokenized_line[indices[0]][index_left].lower())
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

            if index_right < len(tokenized_line[indices[0]]):
                re1 = re.search(r'(\w)+\W(\w)+', tokenized_line[indices[0]][index_right].lower())
                re2 = re.search(r'(\w)+',tokenized_line[indices[0]][index_right].lower())
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
            i += 1
    # To avoid division by 0
    if count != 0:
        # Take the average of the total_vector
        avg_vec = np.divide(total_vec, count)

    X_final = np.append(X_final, [avg_vec], axis=0)


Y_true = Y

clf = svm.SVC()
# clf = svm.SVC(kernel='rbf')
#Split the training and the test data
X_train, X_test, Y_train, Y_test = train_test_split(X_final, Y_true, test_size=0.33, random_state=42)

try:
    #Train the SVM on training data
    clf.fit(X_train, Y_train)
except:
    print("Can't Classify : only 1 unique label found")
    exit()
#Predict on the test data using the model trained by SVM in clf
Y_pred = clf.predict(X_test)
print("Test : ", Y_test, "\n\nPrediction : ", Y_pred)
print("Accuracy : ", accuracy_score(Y_test, Y_pred))
