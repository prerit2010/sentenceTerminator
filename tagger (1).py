import re

file = open( "output/output.txt" , 'r')
story = file.read()
file.close()

story = re.sub( re.compile( r'\"(.+?)(\.)\"(\s*\n*[A_Z])' , re.S) , r'"\1\2"#\2\3' , story)
#Handle the case when conversational text ends with dot(.) by using #. It means start of a new sentence.

story = re.sub( re.compile( r'\"(.+?)(\?)(\s*)\"(\n\n)',re.S) , r'"\1\2\3"#\4' , story)
#Handle the case when conversational text ends with a ? followed by new paragraph.

story = re.sub( re.compile( r'\"(.+?)(\!)(\s*)\"(\n\n)',re.S) , r'"\1\2\3"#\4' , story)
#Handle the case when conversational text ends with a ! followed by new paragraph.

story = re.sub( re.compile( r'\"(.+?)(\?)(\s*)\"(\s*[A-Z])' , re.S) , r'"\1\2\3"#.\4' , story)
#Handle the case when conversational text ends with a ? followed by a new sentence.

story = re.sub( re.compile( r'\"(.+?)(\!)(\s*)\"(\s*[A-Z])' , re.S) , r'"\1\2\3"#.\4' , story)
#Handle the case when conversational text ends with a ! followed by a new sentence.


matches = re.findall( r'\"(.+?)\"' , story , re.S)
#Store all conversational text in a list

for i , str1 in enumerate(matches) :
    story = story.replace( "\"" + str1 + "\"" , "id" + str(i))
#Replace all conversational text with unique identifiers


acronyms = re.findall( r'(?:[A-Z]\.\s)+' , story)
#Find all acronyms in text
acronyms.append('Mr. ')
acronyms.append('Ms. ')
acronyms.append('Mrs. ')
acronyms.append('Dr. ')
acronyms.append('No. ')
acronyms.append('Sr. ')
acronyms.append('Jr. ')
acronyms.append('St. ')
#Add abbreviations to list of acronyms
acronyms.sort()

i = len(acronyms)-1
for str1 in reversed(acronyms):
    story = story.replace( str1,"acc_" + str(i))
    i -= 1
#Replace all acrnonyms with unique identifier


story = re.sub( re.compile( r'(\w)(\!)(\s*[a-z])' , re.S) , r'\1*\2' , story)
#Identify the places where ! in a sentence does not mean end of it

story = re.sub( r'\?', r'$?' , story)
story = re.sub( r'\!', r'%!' , story)
#Differentiate between dot, question mark and exclamation mark by a using a dollar symbol before question mark

para = story.split('\n\n')
#Split all paragraphs

sentences = []
#Split all sentences on dot, question mark and exclamation mark symbol.
for i in range(len(para)) :
    sentences.append( re.split( '\.|\?|\!' , para[i]))

#Add sentence tags in every sentence
for j in range(len(sentences)) :
    for k in range(len(sentences[j])) :
        if len(sentences[j][k])>1 :
            sentences[j][k] = "<s>" + sentences[j][k] + "</s>"

story = ""

#Add sentence terminator symbols before the end of sentence tag
for j in range(len(sentences)) :
    for k in range(len(sentences[j])) :
        if len(sentences[j][k]) > 1 and sentences[j][k].find('%') == len(sentences[j][k]) - 5:
            # Add excalmation mark symbol if percentage symbol is encountered
            story = story + sentences[j][k][0:-5] + "!" + sentences[j][k][-4:len(sentences[j][k])]

        elif len(sentences[j][k]) > 1 and sentences[j][k].find('$') == len(sentences[j][k])-5 :
            #Add question mark symbol if dollar symbol is encountered
            story = story + sentences[j][k][0:-5] + "?" + sentences[j][k][-4:len(sentences[j][k])]
        elif len(sentences[j][k]) > 1 and k < len(sentences[j])-1 :
            #Otherwise add a dot symbol
            story = story + sentences[j][k][0:-4] + "." + sentences[j][k][-4:len(sentences[j][k])]
        elif len(sentences[j][k]) > 1 and k == len(sentences[j])-1 :
            story = story + sentences[j][k]

    story = story + "\n\n"


#Remove all # from story
story=re.sub(r'\*',"!",story)
story = re.sub( r'\#(\.)', "" , story)
story = re.sub(r'\#' , "" , story)


#Replace all conversational unique identifiers by their text
i=len(matches)-1
for str1 in reversed(matches):
    story=story.replace("id"+str(i),"\""+str1+"\"")
    i-=1

#Replace all identifier of acronyms with orignal acronyms
i=len(acronyms)-1
for str1 in reversed(acronyms):
    story=story.replace("acc_"+str(i),str1)
    i-=1

#Write final story in a text file
print("Tagged Successfully!")
file=open("output/train.txt",'w')
file.write(story)
file.close()
