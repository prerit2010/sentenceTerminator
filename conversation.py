import re

file = open("text/test2.txt",'r')
text = file.read()
file.close()
# Append ^^ in the beginning of the conversational text.
text1 = re.sub(r'(\s|\n|\-)\'(\w)',r'\1'+"'^^"+r'\2',text)
# Append $$ in the end of the conversational text.
text = re.sub(r'(\' |\'\n|\';)',"$$"+r'\1',text1)

total_text = ""
#Splitting the the complete text on "'" 
split_text = text.split("'")

i = 0
# iterating over the splitted text
while i<len(split_text):
    # Searching for the nested quotes
    if re.search(r'\^\^.*[^\$][^\$]*$', split_text[i],re.S):
        count = 1
        # inserting the double quote symbol
        total_text = total_text +"\"" + split_text[i][2:len(split_text[i])]
        i += 1
        while(count!=0 and i < len(split_text)):
                # Search for the quotes which are not at the end of the sentence
                if re.search(r'\^\^.*[^\$][^\$]*$',split_text[i],re.S):
                    count = count + 1
                    total_text = total_text+"\'"+split_text[i][2:]

                # Replace the ' as they are inside the conversational text
                elif re.search(r'^\^\^.*\$\$$',split_text[i],re.S):
                    total_text = total_text+"\'"+split_text[i][2:-2]+"\'"

                # Append " sybmol to mark the end of conversational text.
                elif re.search(r'^.*\$\$$',split_text[i],re.S):
                    total_text = total_text + split_text[i][0:len(split_text[i])-2] +"\""
                    count += -1

                else:
                    total_text = total_text + split_text[i]

                i += 1
        i -= 1

    # place the conversational text inside double quotes.
    elif re.search(r'\^\^.*\$\$', split_text[i], re.S):
        total_text = total_text+"\""+split_text[i][2:-2]+"\""


    elif i<(len(split_text)-1) and re.search(r'\w\'\w', split_text[i][len(split_text[i]) - 1] \
         + "'" + split_text[i+1][0], re.I):
        total_text = total_text + split_text[i] + "'" + split_text[i + 1]
        i = i + 1

    else:
        total_text = total_text + split_text[i]

    i += 1

if(i!=len(split_text)):
    total_text = total_text +"error"

file=open("output/output.txt",'w')
file.write(total_text)
file.close()
