# Load libraries
import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
import pandas
from pandas import read_csv
from pandas.plotting import scatter_matrix
import scipy
import scipy.stats
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets.samples_generator import make_blobs
import csv
import importlib
import importlib.util
import numpy
import matplotlib.pyplot as plt
import pickle
import keras
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras.models import model_from_json
import os

# Load dataset
pathy = input("Enter the dataset directory path: ")
pa1=pathy+"/"+"MLENGLISH.csv"
pa2=pathy+"/"+"stats.csv"
pa3=pathy+"/"+"datacorrP.csv"
pa4=pathy+"/"+"datanewchi.csv"
pa5=pathy+"/"+"datanewchi77.csv"
pa6=pathy+"/"+"datanewchi33.csv"
pa7=pathy+"/"+"datanewchi44.csv"
pa8=pathy+"/"+"datanewchi88.csv"
ns=['average_syll_pause_duration','No._long_pause','speaking_time','ave_No._of_words_in_minutes','articulation_rate','No._words_in_minutes','formants_index','f0_index','f0_quantile_25_index',
                              'f0_quantile_50_index','f0_quantile_75_index','f0_std','f0_max','f0_min','No._detected_vowel','perc%._correct_vowel','(f2/f1)_mean','(f2/f1)_std',
                                'no._of_words','no._of_pauses','intonation_index',
				    '(voiced_syll_count)/(no_of_pause)','TOEFL_Scale_Score','Prosody_CFER_Score','Score_Shannon_index','CEFR_Scale_Score','speaking_rate']
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

dataset = pandas.read_csv(pa1, names=ns)    
     
#shape
print(dataset.shape)

# descriptions
stat=dataset.describe()

stat.to_csv(pa2)
print(dataset.describe())

names= ['avepauseduratin','avelongpause','speakingtot','avenumberofwords','articulationrate','inpro','f1norm','mr','q25',
                              'q50','q75','std','fmax','fmin','vowelinx1','vowelinx2','formantmean','formantstd','nuofwrds','npause','ins',
							  'fillerratio','xx','xxx','totsco','xxban','speakingrate']

dataset = pandas.read_csv(pa1, names=names)    

# class distribution
print(dataset.groupby('speakingrate').size())
print(dataset.groupby('articulationrate').size())
print(dataset.groupby('vowelinx2').size())
print(dataset.groupby('formantmean').size())
print(dataset.groupby('formantstd').size())
print(dataset.groupby('xxban').size())

# box and whisker plots
dataset.plot(kind='box', subplots=True, layout=(8,4), sharex=False, sharey=False)
plt.show()

# histograms
dataset.hist()
plt.show()

#Correlation
df = pandas.read_csv(pa1,
                     names = names)
corMx=df.drop(['avenumberofwords','f1norm','inpro','q25','q75','vowelinx1','nuofwrds','npause','xx','xxx','totsco','xxban'], axis=1).corr(method='pearson')
print(corMx)
s = corMx.unstack()
so = s.sort_values(kind="quicksort")
corMx.to_csv(pa3, index = False)
print(so)

# Feature Extraction with Univariate Statistical Tests (Chi-squared for classification)
newMLdataset=dataset.drop(['avenumberofwords','f1norm','inpro','q25','q75','vowelinx1','nuofwrds','npause','xx','totsco','xxban','speakingrate','fillerratio'], axis=1)
newMLdataset.to_csv(pa4, header=False,index = False)
namess=nms = ['avepauseduratin','avelongpause','speakingtot','articulationrate','mr',
                              'q50','std','fmax','fmin','vowelinx2','formantmean','formantstd','ins',
							  'xxx']
df1 = pandas.read_csv(pa4,
                        names = namess)

scatter_matrix(df1,alpha=0.2) # scatter plot matrix
plt.show()
print(df1.shape)


lisss = ['avepauseduratin','avelongpause','speakingtot','articulationrate','mr',
                              'q50','std','fmax','fmin','vowelinx2','formantmean','formantstd','ins']
dfx= df1.drop(['xxx'],axis=1)
dfx.to_csv(pa5, header=False,index = False)
dfy= df1.drop(['avepauseduratin','avelongpause','speakingtot','articulationrate','mr',
                              'q50','std','fmax','fmin','vowelinx2','formantmean','formantstd','ins'],axis=1)
dfy.to_csv(pa8, header=False,index = False)

dfx = pandas.read_csv(pa5, names = None)
dfy = pandas.read_csv(pa8, names = None)

arrayy= dfy.values
array = dfx.values
array=numpy.log(array)
array=numpy.absolute(array)
X = array[:,0:12].astype(float)
Y = arrayy[:,0]
Y=Y.astype(str)


#create model
model = Sequential()

#get number of columns in training data
n_cols = 12

#add model layers
model.add(Dense(500,input_shape=(n_cols,)))
keras.layers.LeakyReLU(alpha=0.03)
model.add(Dense(250))
keras.layers.LeakyReLU(alpha=0.05)
model.add(Dense(125))
keras.layers.LeakyReLU(alpha=0.1)
model.add(Dense(70))
keras.layers.LeakyReLU(alpha=0.6)
model.add(Dense(35,activation='relu'))
model.add(Dense(15, activation='relu'))
model.add(Dense(1,activation='sigmoid'))
#Print model Summary
model.summary()

#compile model using mse as a measure of model performance
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])

# checkpoint
filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

#set early stopping monitor so the model stops training when it won't improve anymore
early_stopping_monitor = EarlyStopping(patience=400)
#train model
model.fit(X, Y, validation_split=0.2, epochs=600, callbacks=[early_stopping_monitor])
history=model.fit(X, Y, validation_split=0.2, epochs=600, callbacks=[early_stopping_monitor])
# evaluate the model
scores = model.evaluate(X,Y)
plt.plot(history.history['mean_squared_error'])
plt.show()
print("#############################################")
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

namess=nms = ['avepauseduratin','avelongpause','speakingtot','articulationrate','mr',
                              'q50','std','fmax','fmin','vowelinx2','formantmean','formantstd','ins',
							  'xxx']

df3 = pandas.read_csv(pa6,
                        names = namess)
df33=df3.drop(['xxx'], axis=1)

array = df33.values
array=numpy.log(array)
array=numpy.absolute(array)
x = array[:,0:12]


#Use the trained model and predict the user's spoken language proficiency.
y= model.predict(x)
y=y[0]

if y>.55:
    yh="c"
    yH=(100-(0.6-y)*1000)
if y>.5 and y<=.55:
    yh="b2"
    yH=(100-(0.55-y)*1000)
if y>.4 and y<=.5:
    yh="b1"
    yH=(100-(0.5-y)*1000)
if y>.3 and y<=.4:
    yh="a2"
    yH=(100-(0.4-y)*1000)
if y>.2 and y<=.3:
    yh="a1"
    yH=(100-(0.3-y)*1000)
if y>=0 and y<=.2:
    yh="a"
    yH=(100-(0.2-y)*1000)

print("#############################################")
print("Probability %:    ",yH)
print("Your spoken language proficiency Level:  ",yh)

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
 
# later...
 
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = loaded_model.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
