#import the ntlk package
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

# import all the packagess
import numpy as np
import tflearn
import tensorflow

import random
import json
import pickle
#need to open the intents.json file in main.py
#intents is the directory of all the main tags, patterns of speech and responses expected

with open("intents.json") as file:
    data = json.load(file)

#looping through the json list
#
try:
    with open("data.pickle", "rb") as f:
        words, labels, training,output = pickle.load(f)
except:
    words=[]
    labels =[]
    # creating new list doc_y, reason is that for each pattern , to know what tag /intent it is a part of
    docs_x = []
    docs_y = []

    #for every single intent in the json file,
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
    #we are gonna do stemming, it will take teach of the patterns and come to the root word
    # we do this to understand the main meaning of the word and remove all things which will change the meaning
    # nltk has a nice function which can do stemming for us
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
                labels.append(intent["tag"])

    #all words in words list, remove duplicates, know vocubulary size of model
    #w.lower to make all words into lower case to reduce confusion
    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    #make list of all the sorted words
    words = sorted(list(set(words)))

    # sorting of labels, not necessary but its ok
    labels = sorted(labels)

    #neural networks only recognise numbers,
    # so we create bag of words,
    # called one hot encoding,
    # which is length of words we have
    #we will have a list of the length of the words we have
    #[0,0,0,0,1,1,1,1,0,1] - encoding the word in this form,
    #add the word to a number in the hot encoding.
    #one hot is only if the word exists.

    training = []
    output = []
    #input data will be the full list with all the words, 0 and 1 for word that exists
    #our output has to be bunch of stringg
    #we need to convert our output also into one hot encoding.
    #our input one hot and output one hot will compare
    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []
    # this is our bag of words, one hot encoding

        wrds = [stemmer.stem(w) for w in doc] # we are going to stem them words

        # now we go through all the diff words in the stem list
        #then we put 1 or 0 in the main bag list

        for w in words:
            if w in wrds:
                bag.append(1) # which means word exists, otherwise
            else:
                bag.append(0)
# now we generate the output
#output has ot be generated as 0 and 1's
        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1
        #look through labels list , see where is the tag is and set that value to 1 in output

        training.append(bag)
        output.append(output_row)


    #we training them into np arrays, change them into arrays to feed into the model
training = np.array(training)
output = np.array(output)

with open("data.pickle", "wb") as f:
    pickle.dump((words, labels, training, output), f)
# now we start building the model using tflearn , we are going to uild the model
# reset is to make sure there are no underlying code pending in tensorflow, to clear tensorflow
tensorflow.reset_default_graph()
#define the input shape that we expect for the model
# the length is 0 , the model should expect us to have an array of 45 or whatever is expected length
net = tflearn.input_data(shape=[None, len(training[0])])
#we are going to add the fully connected layer to neural network, we will have each neuron for this hidden layer with 8 neurons
net = tflearn.fully_connected(net, 8)
#we also need one more hidden layers,
net = tflearn.fully_connected(net,8)
#then we create the output layer , what does is allow to get probability for each output,
#softmax will give us a probaility for each neuron in the layer which wil be output for network.
net = tflearn.fully_connected(net,len(output[0]), activation="softmax")
#
net = tflearn.regression(net)
#to train the model
model = tflearn.DNN(net)

try:
    model.load("model.tflearn")
except:

#time to fit, time to pass all of the training data
#pass all the training data, epoch is the number of times we show the words to pc the same data
    model.fit(training, output, n_epoch=1000, batch_size= 8, show_metric=True)
#then save the model
    model.save("model.tflearn")

#Now we need to get the model working
#we have to ensure that we do not use too much of the code re-running every time we type any requireent

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i , w in enumerate(words):
            if w == se:
                bag[i] = 1
    return np.array(bag)

def chat():
    print("start talking with the HR bot")
    while True:
        inp = input("You: ")
        if inp.lower() =="quit":
            break

        results = model.predict([bag_of_words(inp, words)])
        results_index = np.argmax(results)
        tag = labels[results_index]
        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']
        print(random.choice(responses))

chat()