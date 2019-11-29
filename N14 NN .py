import sklearn as sk
from sklearn.neural_network import MLPClassifier

features = []
labels = []

model = MLPClassifier()
model.fit(features, labels)

result = model.predict(X)
print(result[0])