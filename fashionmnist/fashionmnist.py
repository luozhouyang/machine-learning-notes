import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (
  test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# plt.figure()
# plt.imshow(train_images[0])
# plt.colorbar()
# plt.grid(False)
#
# plt.figure(figsize=(10, 10))
# for i in range(25):
#   plt.subplot(5, 5, i + 1)
#   plt.xticks([])
#   plt.yticks([])
#   plt.grid(False)
#   plt.imshow(train_images[i], cmap=plt.cm.binary)
#   plt.xlabel(class_names[train_labels[i]])

model = keras.Sequential()
model.add(keras.layers.Flatten(input_shape=(28, 28)))
model.add(keras.layers.Dense(128, activation=keras.activations.relu))
model.add(keras.layers.Dense(10, activation=keras.activations.softmax))

model.compile(optimizer=keras.optimizers.Adam(),
              loss=keras.losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])

# train
model.fit(train_images, train_labels, epochs=5)

# eval
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)

predictions = model.predict(test_images)
predict_num_classes = []
for i in range(len(test_labels)):
  prediction = np.argmax(predictions[i])
  print(prediction)
  predict_num_classes.append(prediction)

print(predict_num_classes)
