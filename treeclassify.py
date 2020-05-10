'''
Classifies audio files based on RandomForestClassifier. 
Dataset comes from:
http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html
Labels for emotions come from:
http://www.robots.ox.ac.uk/~vgg/research/cross-modal-emotions/
'''

import pickle
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier 

# Load in pickled file. X and yy are defaults, but could be one of many files.
X = pickle.load(open("X.p", "rb"))
yy = pickle.load(open("yy.p", "rb"))

x_train, x_test, y_train, y_test = train_test_split(X, yy, test_size=0.1, random_state = 127)

parameters = {'bootstrap': True,
              'min_samples_leaf': 1,
              'n_estimators': 50, 
              'min_samples_split': 2,
              'max_features': 'sqrt',
              'max_depth': 200,
              'max_leaf_nodes': None}

RF_model = RandomForestClassifier(**parameters)
RF_model.fit(x_train, y_train)


RF_predictions = RF_model.predict(x_train)
score = accuracy_score(y_train ,RF_predictions)
print("Training accuracy: %.4f%%" % score)
RF_predictions = RF_model.predict(x_test)
score = accuracy_score(y_test ,RF_predictions)
print("Testing accuracy: %.4f%%" % score)

# Save model to file. Name should change based on pickled dataset loaded
filename = 'tree_model.sav'
pickle.dump(RF_model, open(filename, 'wb'))

predictions = RF_model.predict(x_test)
predictionstrain = RF_model.predict(x_train)
# Confusion matrices
print(confusion_matrix(predictionstrain.argmax(axis=1),y_train.argmax(axis=1)))
print(confusion_matrix(predictions.argmax(axis=1),y_test.argmax(axis=1)))