import re

file=open("output/output.txt",'r')
text=file.read()
file.close()

# used # when conversational text ends with a . 
text = re.sub( re.compile( r'\"(.+?)(\.)\"(\s*\n*[A_Z])' , re.S) , r'"\1\2"#\2\3' , text)
# if the conversational text ends with a ? and new line
text = re.sub( re.compile( r'\"(.+?)(\?)(\s*)\"(\n\n)',re.S) , r'"\1\2\3"#\4' , text)
# if the conversational text ends with a ! and new line
text = re.sub( re.compile( r'\"(.+?)(\!)(\s*)\"(\n\n)',re.S) , r'"\1\2\3"#\4' , text)
# if the conversational text ends with a ? and new sentence
text = re.sub( re.compile( r'\"(.+?)(\?)(\s*)\"(\s*[A-Z])' , re.S) , r'"\1\2\3"#.\4' , text)
# if the conversational text ends with a ! and new sentence
text = re.sub( re.compile( r'\"(.+?)(\!)(\s*)\"(\s*[A-Z])' , re.S) , r'"\1\2\3"#.\4' , text)

# find the quoted text.
matches = re.findall(r'\"(.+?)\"',text,re.S)
#print(text)
for i,mystr in enumerate(matches):
    # replace the quoted text with a unique id at every place
    text = text.replace("\""+mystr+"\"","id"+str(i))

#find the acronyms in the text
exceptions = re.findall(r'(?:[A-Z]\.\s)+',text)
# Create a list of possible negative examples of sentence terminators.
special = ['Mr. ', 'Ms. ', 'Mrs. ', 'Dr. ', 'No. ', 'Sr. ', 'Jr. ']
exceptions = exceptions + special
exceptions.sort()

i = len(exceptions)-1
for mystr in reversed(exceptions):
    # replace these 
    text = text.replace(mystr,"acc_"+str(i))
    i -= 1


text = re.sub( re.compile( r'(\w)(\!)(\s*[a-z])' , re.S) , r'\1*\2' , text)
# insert a dollar symbol before ? to differenciate.
text = re.sub(r'\?',r'$?',text)

text = re.sub( r'\!', r'%!' , text)
# split on \n\n, that is the paragraphs
paragraphs = text.split('\n\n')

tagged_lines=[]

for i in range(len(paragraphs)):
    # Split the paragraphs on . or ?
    tagged_lines.append(re.split('\.|\?|\!',paragraphs[i]))

# As the negative examples of . and ? have been removed, we can apply the tags.
for j in range(len(tagged_lines)):
    for k in range(len(tagged_lines[j])):
        if len(tagged_lines[j][k])>1:
            tagged_lines[j][k]="<s>" + tagged_lines[j][k] + "</s>"

text = ""

for j in range(len(tagged_lines)):
    for k in range(len(tagged_lines[j])):

        if len(tagged_lines[j][k]) > 1 and tagged_lines[j][k].find('%') == len(tagged_lines[j][k]) - 5:
            # Add excalmation mark symbol if percentage symbol is encountered
            text = text + tagged_lines[j][k][0:-5] + "!" + tagged_lines[j][k][-4:len(tagged_lines[j][k])]
        # insert a ? if $ is found in the text
        elif len(tagged_lines[j][k]) > 1 and tagged_lines[j][k].find('$') == len(tagged_lines[j][k])-5 :
            text = text + tagged_lines[j][k][0:-5] + "?" + tagged_lines[j][k][-4:len(tagged_lines[j][k])]
        # else insert .
        elif len(tagged_lines[j][k]) > 1 and k < len (tagged_lines[j])-1:
            text=text+tagged_lines[j][k][0:-4]+"."+tagged_lines[j][k][-4:len(tagged_lines[j][k])]
        elif len(tagged_lines[j][k]) > 1 and k == len(tagged_lines[j])-1:
            text=text+tagged_lines[j][k]

    text = text + "\n\n"

text=re.sub(r'\*',"!",text)
# Remove all #. inserted before
text = re.sub(r'\#(\.)',"",text)
# remove #
text = re.sub(r'\#',"",text)
i = len(matches)-1

for mystr in reversed(matches):
    # replace back the unique ids with strings
    text=text.replace("id"+str(i),"\""+mystr+"\"")
    i -= 1

i = len(exceptions)-1
for mystr in reversed(exceptions):
    # replace the unique ids with the accronyms
    text = text.replace("acc_"+str(i),mystr)
    i -= 1

# Save the final txt in the train.txt file
file = open("output/train.txt",'w')
file.write(text)
file.close()
