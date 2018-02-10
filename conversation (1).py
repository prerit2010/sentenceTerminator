import re

file = open( "text/test2.txt" , 'r')
story = file.read()
file.close()


story1 = re.sub( r'(\s|\n|\-)\'(\w)' , r'\1'+"'^^"+r'\2' , story)
#Identify start of conversational text and add ^^ characters in beginning

story = re.sub( r'(\' |\'\n|\';)' , "$$"+r'\1' , story1)
#Identify end of conversatinal text and add $$ characters in end


combine_story = ""
#Combine the text split on quotes in combine_story variable
split_story = story.split("'")
#Split the text on ' character
i = 0

while i < len( split_story ):

    if re.search( r'\^\^.*[^\$][^\$]*$' , split_story[i] , re.S) :
    #Handle the case of nested conversational text(quotes inside quotes) using a counter
    #Text between quotes begins with ^^ but does not end with $$
        count = 1
        combine_story = combine_story + "\"" + split_story[i][2:len(split_story[i])]
        i += 1
        while(count != 0 and i < len(split_story)):

                if re.search( r'\^\^.*[^\$][^\$]*$' , split_story[i] , re.S) :
                    count += 1
                    #Increment the counter when only start of quote is faced
                    combine_story = combine_story + "\'" + split_story[i][2:]


                elif re.search( r'^\^\^.*\$\$$' , split_story[i] , re.S) :
                    #No effect on counter when the text between quotes does not have any nested quotes
                    combine_story = combine_story + "\'" + split_story[i][2:-2] + "\'"

                elif re.search( r'^.*\$\$$' , split_story[i] , re.S) :
                    count += -1
                    #Decrement the counter when only end of quote is faced
                    combine_story = combine_story + split_story[i][0:len(split_story[i])-2] + "\""


                else :
                    combine_story = combine_story + split_story[i]
                i += 1
        i -= 1

    elif re.search( r'\^\^.*\$\$' , split_story[i] , re.S) :
    #Handle the case when conversational text does not have nested quotes.Simply add double quotes in beginning and end
        combine_story = combine_story+"\"" + split_story[i][2:-2] + "\""


    elif i < (len(split_story) -1) and re.search( r'\w\'\w' , split_story[i][ len( split_story[i] ) - 1] + "'" + split_story[i + 1][0], re.I) :
    #Handle the apostrophe case. Concatenate the adjacent strings with a '
        combine_story = combine_story + split_story[i] + "'" + split_story[i + 1]
        i += 1

    else :
    #Add non conversational text without the quotes
        combine_story = combine_story + split_story[i]

    i += 1

#Write final story with replaced quotes in output story
print("Replaced quotes!")
file=open("output/output.txt",'w')
file.write(combine_story)
file.close()







