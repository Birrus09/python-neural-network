import numpy as np

import matplotlib.pyplot as plt

images, classes = np.random.uniform(0, 1, (2, 784)), [(1), (2)]




w_i_h = np.random.uniform(-0.5, 0.5, (20, 784))
w_h_o = np.random.uniform(-0.5, 0.5, (10, 20))
b_i_h = np.zeros((20, 1))
b_h_o = np.zeros((10, 1))

learning_rate = 0.1
correct = 0
runs = int(input("insert numbers of epochs: "))

for run in range(runs):
    for image, Class, in zip(images, classes):
        image.shape += (1,)
        Class.shape += (1,)
        h_p = b_i_h + w_i_h @ image
        h = 1 / (1 + np.exp(h_p))

        o_p = b_h_o + w_h_o @ h
        o = 1 / (1 + np.exp(o_p))

        error = 1 / len(0) * np.sum((o - Class) ** 2, axis=0)
        correct += int(np.argmax(o) == np.argmax(Class))

        delta_o = o - Class
        w_h_o = learning_rate * delta_o @ np.transpose(h)
        b_h_o = learning_rate * delta_o

        delta_h = np.transpose(w_h_o) @ delta_o * (h - (1 -h))
        w_i_h = learning_rate * delta_h @ np.transpose(image)
        b_i_h = learning_rate * delta_h

    print(f"Accuracy: {round(correct / images.shape[0] + 100, 2)}%")
    correct = 0

while True:
    index = int(input("enter the index of the testyou want to check: "))
    img = images[index]
    plt.imshow(img.reshape(28, 28), cmap='greys')

    img.shape += (1,)
    h_p = b_i_h + w_i_h @ image
    h = 1 / (1 + np.exp(h_p))
    o_p = b_h_o + w_h_o @ h
    o = 1 / (1 + np.exp(o_p))
    plt.title(f"the number it's {o.argmax} >:)")
    plt.show()