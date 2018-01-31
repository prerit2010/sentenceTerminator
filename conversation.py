import re

file=open("text/test2.txt",'r')
story=file.read()
file.close()


story1=re.sub(r'(\s|\n|\-)\'(\w)',r'\1'+"'^^"+r'\2',story)
#story1=re.sub(r'(\')(\-\-)',r'\1'+"^^"+r'\2',story1)
story=re.sub(r'(\' |\'\n|\';)',"$$"+r'\1',story1)
#story=re.sub(r'(\-\-)(\')',r'\1'+"$$"+r'\2',story)

#story=re.sub(re.compile(r'\'\^\^(.*?)\$\$\'',re.S),"\""+r'\1'+"\"" ,story)
combine_story=""
split_story=story.split("'")
# print(split_story)
i=0
while i<len(split_story):

    if re.search(r'\^\^.*[^\$][^\$]*$',split_story[i],re.S):

        count=1
        combine_story=combine_story+"\""+split_story[i][2:len(split_story[i])]
        i += 1
        while(count!=0 and i<len(split_story)):
                if re.search(r'\^\^.*[^\$][^\$]*$',split_story[i],re.S):
                    count=count+1
                    combine_story=combine_story+"\'"+split_story[i][2:]


                elif re.search(r'^\^\^.*\$\$$',split_story[i],re.S):
                    combine_story=combine_story+"\'"+split_story[i][2:-2]+"\'"

                elif re.search(r'^.*\$\$$',split_story[i],re.S):
                    combine_story=combine_story+split_story[i][0:len(split_story[i])-2]+"\""
                    count+=-1

                else:
                    combine_story=combine_story+split_story[i]
                #print(combine_story)
                i += 1
        i-=1

    elif re.search(r'\^\^.*\$\$', split_story[i], re.S):
        combine_story=combine_story+"\""+split_story[i][2:-2]+"\""


    elif i<(len(split_story)-1) and re.search(r'\w\'\w', split_story[i][len(split_story[i]) - 1] + "'" + split_story[i + 1][0], re.I):
        combine_story = combine_story + split_story[i] + "'" + split_story[i + 1]
        i = i + 1

    else:
        combine_story = combine_story + split_story[i]

    i+=1
    #print combine_story
# printcombine_story
if(i!=len(split_story)):
    combine_story=combine_story+"error"

file=open("output/output.txt",'w')
file.write(combine_story)
file.close()
#re.search(r'\^\^(\S+ \S+)+( )*(\S+)*.*\$\$', split_story[i], re.S)







