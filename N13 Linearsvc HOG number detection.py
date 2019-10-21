from sklearn.datasets import fetch_openml
from skimage import feature
from sklearn.svm import LinearSVC
import numpy as np 

mnist_dataset = fetch_openml("MNIST_784")

tasvir = mnist_dataset.data
label = mnist_dataset.target 

hogl = []

for item in tasvir:
    f_vec = feature.hog(item.reshape(28,28)) 
    hogl.append((f_vec))

model = LinearSVC()
model.fit(np.array(hogl),label)
score = model.score(np.array(hogl),label)

print(score)