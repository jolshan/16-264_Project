'''
Classifies audio files based on training from CNN. 
Dataset comes from:
http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html
Labels for emotions come from:
http://www.robots.ox.ac.uk/~vgg/research/cross-modal-emotions/
Structure and snippets of code come from:
https://towardsdatascience.com/how-to-apply-machine-learning-and-deep-learning-methods-to-audio-analysis-615e286fcbbc
'''

import pickle
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import Adam

# Load in pickled file. X and yy are defaults, but could be one of many files.
X = pickle.load(open("X.p", "rb"))
yy = pickle.load(open("yy.p", "rb"))

x_train, x_test, y_train, y_test = train_test_split(X, yy, test_size=0.1, random_state = 127)

print(len(x_test[1]))

num_labels = yy.shape[1]
filter_size = 2
model = Sequential()
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_labels))
model.add(Activation('softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

score = model.evaluate(x_test, y_test, verbose=0)
accuracy = 100*score[1]

print("Pre-training accuracy: %.4f%%" % accuracy)

num_epochs = 100
model.fit(x_train, y_train, epochs=num_epochs, validation_data=(x_test, y_test), verbose=1)

# Evaluating the model on the training and testing set
score = model.evaluate(x_train, y_train, verbose=0)
print("Training Accuracy: {0:.2%}".format(score[1]))
score = model.evaluate(x_test, y_test, verbose=0)
print("Testing Accuracy: {0:.2%}".format(score[1]))

# Save model to file; save weights to file. 
# File name should change based on pickled dataset loaded.
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

predictions = model.predict(x_test)
predictionstrain = model.predict(x_train)
# Confusion matrices
print(confusion_matrix(predictionstrain.argmax(axis=1),y_train.argmax(axis=1)))
print(confusion_matrix(predictions.argmax(axis=1),y_test.argmax(axis=1)))