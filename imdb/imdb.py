import keras
import matplotlib.pyplot as plt

imdb = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(
  num_words=10000)
print(
  "Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))

print(train_labels)

word2idx = imdb.get_word_index()
word2idx = {k: (v + 3) for k, v in word2idx.items()}
word2idx['<PAD>'] = 0
word2idx['<START>'] = 1
word2idx['<UNK>'] = 2
word2idx['<UNUSED>'] = 3

idx2word = dict([(value, key) for key, value in word2idx.items()])
print("%s,%s,%s,%s" % (
  idx2word.get(0, "?"), idx2word.get(1, "?"), idx2word.get(2, "?"),
  idx2word.get(3, "?")))


def decode_review(text_ids):
  return " ".join([idx2word.get(i, "?") for i in text_ids])


train_data = keras.preprocessing.sequence.pad_sequences(
  train_data,
  value=word2idx["<PAD>"],
  padding='post',
  maxlen=256)
print(len(train_data[0]), len(train_data[1]))
print(train_data[2])
print(decode_review(train_data[2]))

test_data = keras.preprocessing.sequence.pad_sequences(
  test_data,
  value=word2idx["<PAD>"],
  padding='post',
  maxlen=256)

vocab_size = 10000

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation=keras.activations.relu))
model.add(keras.layers.Dense(1, activation=keras.activations.sigmoid))

model.summary()

model.compile(optimizer=keras.optimizers.Adam(),
              loss=keras.losses.binary_crossentropy,
              metrics=['accuracy'])

x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)
results = model.evaluate(test_data, test_labels)

print(results)

history_dict = history.history
history_dict.keys()

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

plt.clf()  # clear figure
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()
