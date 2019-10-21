#Supervised learning
#Note!: these are fake examples for fruits
#Details of the fruits. sorted by:[color,weigh]
#A classifier is a function that compares the input data with our dataset 
#by finding a pattern between them
##########################################################################
from sklearn import tree

#classifiers dont accept strings so:
#red = 0
#green = 1
#yelllow = 2 
#orange = 3
#apple = 0
#banana = 1
#orange = 2 
descriptions = [[0,150],[1,150],[2,150],[3,200]]
labels = [0,0,1,2]

#create a classifier
classifier = tree.DecisionTreeClassifier()
#assigning the data-set for it
classifier = classifier.fit(descriptions,labels)

#Getting a description from the user
input_color = int(input("What color is the fruit you see? "))
input_size = int(input("How much does the fruit you see weigh? "))

#comparing the input-data with the dataset and finding what data-input is
result = classifier.predict([[input_color,input_size]])

print("Your fruit is:",result)
