'''
COMP551 PROJECT 1

PREPROCESSING

Vincent Bazinet
Gabriel Desrosiers-Grégoire
Kevin Ma

**MAKE SURE THAT ALL THE FILES FOR THE PROJECT ARE IN THE SAME DIRECTORY

'''
import numpy as np
import json
with open("proj1_data.json") as fp:
    data = json.load(fp)

word_count = {}

 #function that counts the occurence of the top 160 words in a string
 
def count_top_words(string,top_words):
    words= string.split(" ")
    word_counts = np.zeros(len(top_words))
    i=0
    for target_word in top_words:
        count=0
        for word in words:
            if word==target_word:
                count+=1
        word_counts[i] = count
        i+=1
    return word_counts

#MAIN

#count the occurence of each word in the first 10000 comments
for i in range(10000):
    print(i)
    comment = data[i]['text']
    comment = comment.lower()
    words = comment.split(" ")
    
    for word in words:
        if word in word_count:
            word_count[word] = word_count[word] + 1
            
        else:
            word_count[word] = 1
  
#store the top 160 words in a list
top_words = []        
for i in range(160):
    word = max(word_count, key=word_count.get)
    top_words.append(word)
    word_count.pop(word)
    
#build the dataset
num_features= 164
sorted_data=np.zeros([len(data),num_features])
for i in range(len(data)):
    data_point = data[i]
    sorted_data[i,0]=data_point['popularity_score']
    sorted_data[i,1]=data_point['children']
    sorted_data[i,2]=data_point['controversiality']
    sorted_data[i,3]=int(data_point['is_root']) 
    comment=data_point['text']
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
 