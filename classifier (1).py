import numpy as np

import nltk
import re
import pickle
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

#Specify dimension of vectors and window size to be used
dimension = 200
windowsize = 1

def GloveVectors_Load(file_Glove):
    #Load full GloveVector file in model
    vector_file = open(file_Glove,'r')
    model = {}
    for vec in vector_file:
        splitvec = vec.split()
        word = splitvec[0]
        vector = np.array([float(value) for value in splitvec[1:]])
        model[word] = vector
    print("Done.",len(model)," words loaded!")
    return model

def Store_vectors(obj, name ):
    #Store the dictionary of wordvectors in pickle format
    with open('obj/'+ name + '.pkl', 'wb') as vector_file:
        pickle.dump(obj, vector_file, protocol=2)

def Load_vectors(name ):
    #Load the dictionary of word vectors from pickle format
    with open('obj/' + name + '.pkl', 'rb') as vector_file:
        return pickle.load(vector_file)

def Extract_wordvec(dimension):

    word_vec = GloveVectors_Load("data/27Bx" + str(dimension) + "d.txt")
    # Store all word vectors that are relevant to the story in a dictionary

    test_file = open("text/test2.txt")
    story = test_file.read()
    tokens = word_tokenize(story)
    tokens = set(tokens)
    tokens = list(tokens)
    story_wordvec = {}


    for i, token in enumerate(tokens):
        case1 = re.search(r'(\w)+\W(\w)+', token.lower())
        case2 = re.search(r'(\w)+',token.lower())
        try: 
            if case1:
                story_wordvec[case1.group()] = word_vec[case1.group()]
            elif case2:
                story_wordvec[case2.group()] = word_vec[case2.group()]  
        except:
            new_vec = np.random.rand(dimension,)
            if case1:
                story_wordvec[case1.group()] = new_vec
            elif case2:
                story_wordvec[case2.group()] = new_vec
            

    Store_vectors(story_wordvec, "story_word_vec" + str(dimension))


# Extract_wordvec(dimension)
story_wordvec = Load_vectors("story_word_vec" + str(dimension))

#Extract tagged sentences
file = open("output/train.txt",'r')
story = file.read()
file.close()

#Find all conversational text in a list
convo = re.findall(r'\"(.+?)\"',story,re.S)

#Replace every conversational text with a unique identifier
for i,str1 in enumerate(convo):
    story = story.replace("\""+str1+"\"","id"+str(i))

#Ignore all dots, question marks and exclamation marks inside conversational text except the ones occuring in the end
for i in range(len(convo)):
    if convo[i][-1] == ".":
        convo[i]=convo[i][0:-1]+"@"
    if convo[i][-1] == "?":
        convo[i] = convo[i][0:-1]+"$"
    if convo[i][-1] == "!":
        convo[i] = convo[i][0:-1]+"%"
    convo[i] = convo[i].replace("?","#")
    convo[i] = convo[i].replace(".","#")
    convo[i] = convo[i].replace("!","#")
    convo[i] = convo[i].replace("$","?")
    convo[i] = convo[i].replace("@",".")
    convo[i] = convo[i].replace("%", "!")

#Put back all conversational text
i = len(convo)-1
for str1 in reversed(convo) :
    story = story.replace( "id"+str(i) , "\""+str1+"\"")
    i -= 1

#Replace the conversational text ending with a question mark with a double question mark
story = re.sub( r'\?(\s*\"*\s*</s>)' , r'??\1' , story)
story = re.sub( r'\!(\s*\"*\s*</s>)' , r'!!\1' , story)

#Find all the sentences in story using sentence tag
sentences = re.findall( r'<s>(.+?)</s>' , story , re.S)
tokens = []
#Tokenize all sentences
for i in range(len(sentences)) :
    tokens.append(nltk.word_tokenize(sentences[i]))
X = []
Y = []
j = 0

#Give all the indices of dots and question mark except the ones ending with these symbols
for i in range(len(tokens)) :
    while j < (len(tokens[i])) :

        if( tokens[i][j].find(".") != -1 and j != len(tokens[i]) -1) and tokens[i][j+1] != "''" :
            #if token contains a dot and it does not occur in the end of converational text
            X.append([i, j])
            Y.append(0)
        elif tokens[i][j].find(".") != -1 :
            X.append([i, j])
            Y.append(1)

        if (tokens[i][j].find("?") != -1 and j != len(tokens[i]) - 1) and tokens[i][j] == "?" and  tokens[i][j+1] != "?" :
            #if token contains a question mark and it occurs in the end of conversational text
            X.append([i, j])
            Y.append(0)

        elif (tokens[i][j].find("?") != -1 and j != len(tokens[i]) - 1) and tokens[i][j] == "?" and  tokens[i][j+1] == "?" :
            X.append([i, j])
            Y.append(1)
            j = j+1

        if (tokens[i][j].find("!") != -1 and j != len(tokens[i]) - 1) and tokens[i][j] == "!" and tokens[i][j+1] != "!" :
            # if token contains a question mark and it occurs in the end of conversational text
            X.append([i, j])
            Y.append(0)

        elif (tokens[i][j].find("!") != -1 and j != len(tokens[i]) - 1) and tokens[i][j] == "!" and tokens[i][j + 1] == "!" :
            X.append([i, j])
            Y.append(1)
            j += 1
        j += 1
    j = 0
X = np.array(X)
Y = np.array(Y)

X_examples = np.empty((0,dimension))
#For every index of dot and question mark
for index in X:

    sum_vec = np.zeros(dimension,)
    count = 0
    #if token only has a single character dot, question mark or exclamation mark
    if tokens[index[0]][index[1]] == "." or "?" or "!":
        # sum all the word vec in left window
        for i in range(windowsize):
            left_index = index[1] - i - 1
            if left_index >= 0:
                case1 = re.search(r'(\w)+\W(\w)+', tokens[index[0]][left_index].lower())
                case2 = re.search(r'(\w)+',tokens[index[0]][left_index].lower())
                if case1:
                    try:
                        vec = story_wordvec[case1.group()]
                        sum_vec = np.add(sum_vec, vec)
                        count += 1
                    except:
                        count += 0
                elif case2:
                    try:
                        vec = story_wordvec[case2.group()]
                        sum_vec = np.add(sum_vec, vec)
                        count += 1
                    except:
                        count += 0
            right_index = index[1] + i + 1
            #sum of all word vectors in right window
            if right_index < len(tokens[index[0]]) :
                case1 = re.search(r'(\w)+\W(\w)+', tokens[index[0]][right_index].lower())
                case2 = re.search(r'(\w)+',tokens[index[0]][right_index].lower())
                if case1:
                    try:
                        vec = story_wordvec[case1.group()]
                        sum_vec = np.add(sum_vec, vec)
                        count += 1
                    except:
                        count += 0
                elif case2:
                    try:
                        vec = story_wordvec[case2.group()]
                        sum_vec = np.add(sum_vec, vec)
                        count += 1
                    except:
                        count += 0

    else:
        #if token has other characters other than dot
        break_tok = tokens[index[0]][index[1]].split('.')
        i = 0
        j = 0
        while(j < len(break_tok)):
                i = i+1
                case1 = re.search(r'(\w)+\W(\w)+', break_tok[j].lower())
                case2 = re.search(r'(\w)+',break_tok[j].lower())
                if case1:
                    try:
                        vec = story_wordvec[case1.group()]
                        sum_vec = np.add(sum_vec, vec)
                        count += 1

                    except:
                        count += 0
                elif case2:
                    try:
                        vec = story_wordvec[case2.group()]
                        sum_vec = np.add(sum_vec, vec)
                        count += 1
                    except:
                        count += 0
                j = j+1

        while(i < windowsize):
            left_index = index[1]-i-1
            if left_index >= 0:
                case1 = re.search(r'(\w)+\W(\w)+', tokens[index[0]][left_index].lower())
                case2 = re.search(r'(\w)+',tokens[index[0]][left_index].lower())
                if case1:
                    try:
                        vec = story_wordvec[case1.group()]
                        sum_vec = np.add(sum_vec, vec)
                        count += 1
                    except:
                        count += 0
                elif case2:
                    try:
                        vec = story_wordvec[case2.group()]
                        sum_vec = np.add(sum_vec, vec)
                        count += 1
                    except:
                        count += 0
            right_index = index[1] + i + 1

            if right_index < len(tokens[index[0]]):
                case1 = re.search(r'(\w)+\W(\w)+', tokens[index[0]][right_index].lower())
                case2 = re.search(r'(\w)+',tokens[index[0]][right_index].lower())
                if case1:
                    try:
                        vec = story_wordvec[case1.group()]
                        sum_vec = np.add(sum_vec, vec)
                        count += 1
                    except:
                        count += 0
                elif case2:
                    try:
                        vec = story_wordvec[case2.group()]
                        sum_vec = np.add(sum_vec, vec)
                        count += 1
                    except:
                        count += 0
            i += 1

    if count != 0:
        #Average out the sum of vectors
        sum_vec = np.divide(sum_vec, count)


    X_examples = np.append(X_examples, [sum_vec], axis=0)


#Initialize SVM model
model = svm.SVC()
# model = svm.SVC(kernel='rbf')
#Split training and testing data
X_train, X_test, Y_train, Y_test = train_test_split(X_examples, Y, test_size=0.33, random_state=42)
#Learn the model for training data
try:
    model.fit(X_train, Y_train)
except:
    print("Can't Classify : only 1 unique label found")
    exit()
#Predict outputs for test data
Y_pred = model.predict(X_test)
print("Test : ", Y_test)
print("Prediction : ", Y_pred)
#Compute acuracy
print("Accuracy : ", accuracy_score(Y_test, Y_pred))
