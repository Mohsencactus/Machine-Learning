import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np 
import json
import random
from sklearn.neural_network import MLPClassifier
from sklearn.externals import joblib
###############################################################
def percent(X):
    percent = int(X*1000)/10
    return percent 

def bagger(msg, words):
    bag = [0 for _ in range(len(words))]
    wrds = msg.split(' ')
    wrds = [word.lower() for word in wrds]
    for x in wrds:
        for i,w in enumerate(words):
            if w == x:
                bag[i] = 1
    return np.array(bag)
###############################################################
model = MLPClassifier()
stemmer = LancasterStemmer()

words = []
labels = []
docs_x = []
docs_y = []
WFeatures = []
WLabels = []
result = ['']
###############################################################
with open("intents.json") as file:
    data = json.load(file)
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        wrds = pattern.split(' ')
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent["tag"])
    labels.append(intent["tag"])
words = [w.lower() for w in words]
words = sorted(list(set(words)))
output_row = []
for x,doc in enumerate(docs_x):
    kwords = []
    wrds = [w.lower() for w in doc]
    for w in words:
        if w in wrds:
            kwords.append(1)
        else:
            kwords.append(0)
    WFeatures.append(kwords)
    WLabels.append(docs_y[x]) 
###############################################################
WFeatures = np.array(WFeatures)
WLabels = np.array(WLabels)
###############################################################
try :
    model = joblib.load('words_model.pkl') 
except:
    model.fit(np.array(WFeatures),WLabels)
    joblib.dump(model, 'words_model.pkl') 

print("Hi I am R3VERSE!")
print("I am a chatbot. ^_^")
print("I will learn to speaking soon")
print("So Whats up? ")
while True:
    msg = input("Your massage: ")
    feature = bagger(msg,words)
    result = model.predict([feature])
    score = model.predict_proba([feature])
    score = percent(max(score[0]))
    tag = result[0]
    for tg in data["intents"]:
        if tg['tag'] == tag:
            responses = tg['responses']
            response = random.choice(responses)
    Total = (response,tag,score)
    
    print(response)