'''
COMP551 PROJECT 1

PREPROCESSING

Vincent Bazinet
Gabriel Desrosiers-Grégoire
Kevin Ma

**MAKE SURE THAT ALL THE FILES FOR THE PROJECT ARE IN THE SAME DIRECTORY

'''

#load the data for this assignment from the json file
import numpy as np
import json
with open("proj1_data.json") as fp:
    data = json.load(fp)

'''
Functions useful for our pre-processing:
'''

#functions that tells whether a comment is spam or a bot
def is_spam(string):
    
    words = string.split(" ")
    
    spam_words = []
    with open('spam_words.txt','r') as f:
        for line in f:
            for word in line.split(", "):
                spam_words.append(word)
    
    for word in words:
        for spam_word in spam_words:
            if word.startswith(spam_word):
                return True           
    return False

#function that counts the number of "negative" words
def count_negativity(string):
    
    negative_words = []
    with open('negative_words.txt','r') as f:
        for line in f:
            for word in line.split(", "):
                negative_words.append(word)
    
    words = string.split(" ")
    
    totalnb = 0
    for word in words:
        for negative_word in negative_words:
            if word == negative_word:
                totalnb += 1
    return totalnb

#functions that computes the length of the comment
def length(string):
    words = string.split(" ")
    return len(words)

#function that computes the average length of the words in the comment
def ave_word_length(string):
    words = string.split(" ")
    return (len(string)/len(words))

#function that counts the number of capitalized words in the comment
def capitalized_nb(string):
    words = string.split(" ")
    count = 0
    for word in words:
        if word.istitle():
            count+=1
    return count

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

'''MAIN'''

'''
COMPUTE THE LIST OF THE TOP 160 WORDS
'''
#count the occurence of each word in the first 10000 comments
word_count = {}  #store results
for i in range(10000):
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
    
'''
BUILD THE DATASET
'''
    
num_features= 1+1+3+160+2   #popularity/intercept/3_main/text_features/extra_two
sorted_data=np.zeros([len(data),num_features])
for i in range(len(data)):
    
    data_point = data[i]
    comment = data_point['text']
    
    sorted_data[i,0]= data_point['popularity_score']
    sorted_data[i,1]= 1 # the intercept is set up to values of 1
    sorted_data[i,2]= data_point['children']
    sorted_data[i,3]= data_point['controversiality']
    sorted_data[i,4]= int(data_point['is_root']) 
    sorted_data[i,5:165]= count_top_words(comment,top_words)
    
    '''
    EXTRA FEATURES

        ***UN-COMMENT ANY FEATURE YOU WOULD LIKE TO TRY
        
        ***NOTE : the count_negativity() function is not
        ***optimized and therefore take a ~5 minutes to run
    '''

    sorted_data[i,165] = (sorted_data[i,2]) * length(comment)
    sorted_data[i,166] = (sorted_data[i,2])**(1 + sorted_data[i,4])
    #sorted_data[i,165] = is_spam(comment)
    #sorted_data[i,166] = count_negativity(comment)

#training dataset    
X_train=sorted_data[:10000,1:]
Y_train=sorted_data[:10000,0]
#validation dataset
X_valid=sorted_data[10000:11000,1:]
Y_valid=sorted_data[10000:11000,0]
#test dataset
X_test=sorted_data[11000:,1:]
Y_test=sorted_data[11000:,0]
 