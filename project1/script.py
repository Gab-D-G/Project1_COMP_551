import os
os.chdir('/home/gabriel/Desktop/comp551/project1')
import numpy as np
import json
with open("data/proj1_data.json") as fp:
    data = json.load(fp)

# Now the data is loaded.
# It a list of data points, where each datapoint is a dictionary with the following attributes:
# popularity_score : a popularity score for this comment (based on the number of upvotes) (type: float)
# children : the number of replies to this comment (type: int)
# text : the text of this comment (type: string)
# controversiality : a score for how "controversial" this comment is (automatically computed by Reddit)
# is_root : if True, then this comment is a direct reply to a post; if False, this is a direct reply to another comment 



def parse_text(string):
    #first convert the string to lower case
    string=string.lower()
    
    #make a list of the words delimited by whitespaces
    words=[]
    i=0
    while(i<len(string)):
        word=''
        while(i<len(string) and not string[i]==' '):
            word+=string[i]
            i+=1
        if not word=='':
            words.append(word)
        i+=1
        
    return words

#stack all the words from the training data (first 10000 samples) in a single list
full_word_list=[]
for i in range(10000):
    string=data[i].get('text')
    words=parse_text(string)
    for word in words:
        full_word_list.append(word)

counted_words=[]
word_occurences=[]
c=0
for target_word in full_word_list:
    counted=False
    for word in counted_words:
        if word==target_word:
            counted=True
            break

    if not counted:        
        #iterate through the full list and count the number of occurences for the target word
        num_occurences=0
        for word in full_word_list:
            if word==target_word:
                num_occurences+=1
        word_occurences.append(num_occurences)
        counted_words.append(target_word)
    print(c)
    c+=1

import pickle
pickle.dump(counted_words,open('counted_words.txt','wb'))
pickle.dump(word_occurences,open('word_occurences','wb'))

#create 160-long list of empty string
top_words=[]
for i in range(160):
    top_words.append('')
top_occurences=np.zeros(160)
#now iterate through the list and build progressively a numpy array 
#of the 160 most frequent words in increasing order of occurence
c=0
for target_word in counted_words:
    num_occurences=word_occurences[c]
    #if the word has a higher number of occurences than the least occuring word in the current list,
    #place the word in the array, reorganize the list and drop the least occuring word
    if num_occurences>top_occurences[0]:
        #first find the index to place the new word
        i=0
        while(i<len(top_occurences)-1 and num_occurences>top_occurences[i+1]):
            i+=1
        for j in range(i):
            top_words[j]=top_words[j+1]
            top_occurences[j]=top_occurences[j+1]
        top_words[i]=target_word
        top_occurences[i]=num_occurences
    print(c)
    c+=1
    
pickle.dump(top_words,open('top_words.txt','wb'))
pickle.dump(top_occurences,open('top_occurences','wb'))
top_words=pickle.load(open('top_words.txt','rb'))
top_occurences=pickle.load(open('top_occurences','rb'))

def count_top_words(string,top_words):
    words=parse_text(string)
    word_counts=np.zeros(len(top_words))
    i=0
    for target_word in top_words:
        count=0
        for word in words:
            if word==target_word:
                count+=1
        word_counts[i]=count
        i+=1
    return word_counts
                
        

#fill a numpy array with the scores in the first column, followed by the selected
    #features to feed to the model
num_features=4 + 160 #has the 4 basic features + the top word count
sorted_data=np.zeros([len(data),num_features])
for i in range(len(data)):
    data_point = data[i]
    sorted_data[i,0]=data_point.get('popularity_score')
    sorted_data[i,1]=data_point.get('children')
    sorted_data[i,2]=data_point.get('controversiality')
    sorted_data[i,3]=int(data_point.get('is_root'))    
    comment=data_point.get('text')
    sorted_data[i,4:]=count_top_words(comment,top_words)

#training dataset    
X_train=sorted_data[:10000,1:]
Y_train=sorted_data[:10000,0]
#validation dataset
X_valid=sorted_data[10000:11000,1:]
Y_valid=sorted_data[10000:11000,0]
#test dataset
X_test=sorted_data[11000:,1:]
Y_test=sorted_data[11000:,0]
