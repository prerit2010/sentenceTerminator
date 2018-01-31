import re

file=open("output/output.txt",'r')
story=file.read()
file.close()

story=re.sub(re.compile(r'\"(.+?)(\.)\"',re.S),r'"\1\2"#\2',story)
story=re.sub(re.compile(r'\"(.+?)(\?)(\s*)\"(\n\n)',re.S),r'"\1\2\3"#\4',story)
story=re.sub(re.compile(r'\"(.+?)(\?)(\s*)\"(\s*[A-Z])',re.S),r'"\1\2\3"#.\4',story)


matches=re.findall(r'\"(.+?)\"',story,re.S)
#print(story)
for i,str1 in enumerate(matches):
    if(story.find("\""+str1+"\"")) ==-1:
        print("\""+str1+"\""+"NOT FOUND")

    story=story.replace("\""+str1+"\"","id"+str(i))
#print(story)


acronyms=re.findall(r'(?:[A-Z]\.\s)+',story)

acronyms.append('Mr. ')
acronyms.append('Ms. ')
acronyms.append('Mrs. ')
acronyms.append('Dr. ')
acronyms.append('No. ')
acronyms.append('Sr. ')
acronyms.append('Jr. ')
acronyms.sort()
print(acronyms)
i=len(acronyms)-1
for str1 in reversed(acronyms):
    story=story.replace(str1,"acc_"+str(i))
    i-=1
#print(story)
story=re.sub(r'\?',r'$?',story)
para=story.split('\n\n')
# print(story)
sentences=[]

for i in range(len(para)):
    sentences.append(re.split('\.|\?',para[i]))
#print(sentences)
for j in range(len(sentences)):
    for k in range(len(sentences[j])):
        if len(sentences[j][k])>1:
            sentences[j][k]="<s>"+sentences[j][k]+"</s>"
story=""
#print(sentences)
for j in range(len(sentences)):
    for k in range(len(sentences[j])):
        if len(sentences[j][k]) > 1 and sentences[j][k].find('$')==len(sentences[j][k])-5  :
            story = story + sentences[j][k][0:-5] + "?" + sentences[j][k][-4:len(sentences[j][k])]
        elif len(sentences[j][k]) > 1 and k<len(sentences[j])-1:
            story=story+sentences[j][k][0:-4]+"."+sentences[j][k][-4:len(sentences[j][k])]
        elif len(sentences[j][k]) > 1 and k==len(sentences[j])-1:
            story=story+sentences[j][k]

    story=story+"\n\n"
#print(story)
story=re.sub(r'\#(\.)',"",story)
story=re.sub(r'\#',"",story)
i=len(matches)-1
for str1 in reversed(matches):
    story=story.replace("id"+str(i),"\""+str1+"\"")
    i-=1
#print(story)
i=len(acronyms)-1
for str1 in reversed(acronyms):
    story=story.replace("acc_"+str(i),str1)
    i-=1
#print(story)


#print(story)
file=open("output/train.txt",'w')
file.write(story)
file.close()
