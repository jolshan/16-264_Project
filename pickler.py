'''
Extracts features from files and pickles them into files for input and output. 
Dataset comes from:
http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html
Labels for emotions come from:
http://www.robots.ox.ac.uk/~vgg/research/cross-modal-emotions/
Structure and snippets of code come from:
https://towardsdatascience.com/how-to-apply-machine-learning-and-deep-learning-methods-to-audio-analysis-615e286fcbbc
'''


import librosa
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

# Text file containing all files and emotion labels. 
# Default is "filesAndLabels.txt" but may change based on sample
fd = open('filesAndLabels.txt', 'r')

def extract_features(file_name):
    audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_processed = np.mean(mfccs.T,axis=0)
     
    return mfccs_processed

features = []
# Iterate through each sound file and extract the features 
for line in fd:
    nameAndLabel = line.split()
    
    class_label = nameAndLabel[1]
    data = extract_features('wav/' + nameAndLabel[0])
    
    features.append([data, class_label])
# Convert into a Panda dataframe 
featuresdf = pd.DataFrame(features, columns=['feature','class_label'])

featuresdf.head()

featuresdf.iloc[0]['feature']

# Convert features and corresponding classification labels into numpy arrays
X = np.array(featuresdf.feature.tolist())
y = np.array(featuresdf.class_label.tolist())
# Encode the classification labels
le = LabelEncoder()
yy = to_categorical(le.fit_transform(y))

pickle.dump(X, open("X.p", "wb"))
pickle.dump(yy, open("yy.p", "wb"))