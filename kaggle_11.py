# -*- coding: utf-8 -*-
"""Created on Fri Feb  9 12:34:12 2024 @author: deepak"""

import os
import sys
pwd
os.chdir("C:/DEEPAK/Personal/Kaggle-Multi-Class Prediction of Obesity Risk/Data")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt # for data visualization purposes
import seaborn as sns # for statistical data visualization
%matplotlib inline
import imblearn
import datetime

# =============================================================================
# ################ train ['id', 'Gender', 'Age', 'Height', 'Weight','family_history_with_overweight', 'FAVC', 'FCVC', 'NCP', 'CAEC','SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE', 'CALC', 'MTRANS', 'NObeyesdad']
# =============================================================================

train=pd.read_csv("train.csv", header=0)
train.sample(5)
train.head(5)
train

train.shape
train.columns
train['NObeyesdad'].value_counts()


# =============================================================================
# ################ Merge into one file
# =============================================================================

data_train = train.copy()

data_train.shape
data_train.head


# =============================================================================
# ################ Describe the data file
# =============================================================================

data_train.info()

# Statistical description of numerical variables
data_train.describe()

# Boxplot of numerical variables

# Handling outlier

# =============================================================================
# ################ Remove Text columns
# =============================================================================

data_train.info()

data_train.drop(['id'], axis=1, inplace=True)

# Boolean variable to integer
data_train['Gender']
data_train['family_history_with_overweight']
data_train['FAVC']
data_train['CAEC'].value_counts()
data_train['SMOKE']
data_train['SCC']
data_train['CALC'].value_counts()
data_train['MTRANS'].value_counts()


data_train['Gender'] = data_train['Gender'].map({'Male': 1, 'Female': 0})
data_train['family_history_with_overweight'] = data_train['family_history_with_overweight'].map({'yes': 1, 'no': 0})
data_train['FAVC'] = data_train['FAVC'].map({'yes': 1, 'no': 0})
data_train['CAEC'] = data_train['CAEC'].map({'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3})
data_train['SMOKE'] = data_train['SMOKE'].map({'yes': 1, 'no': 0})
data_train['SCC'] = data_train['SCC'].map({'yes': 1, 'no': 0})
data_train['CALC'] = data_train['CALC'].map({'no': 0, 'Sometimes': 1, 'Frequently': 2})
data_train['MTRANS'] = data_train['MTRANS'].map({'Bike': 1, 'Motorbike': 2, 'Walking': 3, 'Automobile': 4, 'Public_Transportation': 5})

data_train.info()


# =============================================================================
# ################ target label and models
# =============================================================================
---------->

data_train['NObeyesdad']

from sklearn.preprocessing import LabelEncoder
le2=LabelEncoder()
X_train=data_train.drop('NObeyesdad',axis=1)
Y_train=le2.fit_transform(data_train['NObeyesdad'])

X_train.shape
Y_train.shape
type(X_train)
type(Y_train)
type(data_train)

X_train.info()


# =============================================================================
# ################ PCA --------1
# =============================================================================

# -----Scale the data to be between -1 and 1
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X1=scaler.fit_transform(X_train)
X1

from sklearn.decomposition import PCA
pca = PCA()
pca.fit_transform(X1)

pca.get_covariance()

explained_variance=pca.explained_variance_ratio_
explained_variance

with plt.style.context('dark_background'):
    plt.figure(figsize=(6, 4))

    plt.bar(range(17), explained_variance, alpha=0.5, align='center',
            label='individual explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.legend(loc='best')
    plt.tight_layout()

pca=PCA(n_components=7)
X_train_pca=pca.fit_transform(X1)
X_train_pca

pca.get_covariance()

explained_variance=pca.explained_variance_ratio_
explained_variance

with plt.style.context('dark_background'):
    plt.figure(figsize=(6, 4))

    plt.bar(range(7), explained_variance, alpha=0.5, align='center',
            label='individual explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.legend(loc='best')
    plt.tight_layout()

X_train_pca.shape
Y_train.shape

Y_train.value_counts()

y1=pd.DataFrame(Y_train)
y1.value_counts()


# =============================================================================
# ################ LDA --------1
# =============================================================================


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X1 = scaler.fit_transform(X_train)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis()
X_train_lda = lda.fit_transform(X1, Y_train)


X_train_lda.shape
Y_train.shape
Y_train.value_counts()




# =============================================================================
# ################ t-SNE --------2
# =============================================================================

--------PENDING

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Misalkan X adalah data Anda dengan dimensi yang lebih tinggi
# Inisialisasi objek t-SNE
tsne = TSNE(n_components=3, random_state=0)

# Melakukan reduksi dimensionalitas dengan t-SNE
X_train_tsne = tsne.fit_transform(X_train)

X_train_tsne.shape
type(X_train_tsne)


# =============================================================================
# ################ Merge PCA and t-SNE
# =============================================================================

X_train_pca_tsne = np.hstack((X_train_pca, X_train_tsne))
X_train_pca_tsne.shape
type(X_train_pca_tsne)


# =============================================================================
# ################ Class imbalance
# =============================================================================

from imblearn.combine import SMOTE


###### SMOTE + ENN

from imblearn.combine import SMOTEENN

#smenn = SMOTEENN(sampling_strategy=1.0)
#smenn = SMOTEENN(random_state=101)
smenn = SMOTEENN(random_state=42)
X_train_pca_tsne_class, Y_train_class = smenn.fit_resample(X_train_pca_tsne, Y_train)
#X_sm, y_sm = smenn.fit_resample(X_sm, y_sm)

X_train_pca_tsne_class.shape
Y_train_class.shape
Y_train_class.value_counts()



# =============================================================================
# ################ Logistic
# =============================================================================

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

#classifier = LogisticRegression(penalty='l2', C=0.6, solver='liblinear', random_state=42)
classifier = LogisticRegression(multi_class='multinomial', random_state=42)

############# Cross validation score #######
skfold=StratifiedKFold(n_splits=5)
scores=cross_val_score(classifier,X_train_pca,Y_train,cv=skfold)

scores
print(np.mean(scores))
print("Accuracy: %.3f%% (%.3f%%)" % (scores.mean()*100.0, scores.std()*100.0))

Y_train_pred = cross_val_predict(classifier,X_train_pca,Y_train,cv=skfold)

f1=f1_score(Y_train, Y_train_pred, average='weighted')
f1


# =============================================================================
# ################ XGBoost
# =============================================================================

import xgboost as xgb
#xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=42)

xgb_model = xgb.XGBClassifier(objective='multi:softmax', random_state=42)

scores=cross_val_score(xgb_model,X_train_pca,Y_train,cv=skfold)
scores
print(np.mean(scores))
print("Accuracy: %.3f%% (%.3f%%)" % (scores.mean()*100.0, scores.std()*100.0))

Y_train_pred = cross_val_predict(xgb_model, X_train_pca,Y_train, cv=skfold)

f1=f1_score(Y_train, Y_train_pred, average='weighted')
f1


# =============================================================================
# ################ Random Forest
# =============================================================================


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 100, random_state = 1)
scores=cross_val_score(rf,X_train_pca,Y_train,cv=skfold)
scores
print(np.mean(scores))
print("Accuracy: %.3f%% (%.3f%%)" % (scores.mean()*100.0, scores.std()*100.0))

Y_train_pred = cross_val_predict(rf, X_train_pca,Y_train, cv=skfold)

f1=f1_score(Y_train, Y_train_pred, average='weighted')
f1


# =============================================================================
# ################ SVM
# =============================================================================

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV

param_grid = {'C': [0.1, 1, 10, 100],
              'gamma': [1, 0.1, 0.01, 0.001],
              'kernel': ['rbf', 'linear', 'poly', 'sigmoid']}

grid_search = GridSearchCV(SVC(random_state=1), param_grid, cv=skfold)
grid_search.fit(X_train_pca, Y_train)
best_params = grid_search.best_params_




skfold=StratifiedKFold(n_splits=5)
from sklearn.svm import SVC
svm = SVC(random_state = 1,kernel='linear',gamma=1,C=0.1)
scores=cross_val_score(svm, X_train_pca, Y_train, cv=skfold)
scores
print(np.mean(scores))
print("Accuracy: %.3f%% (%.3f%%)" % (scores.mean()*100.0, scores.std()*100.0))

Y_train_pred = cross_val_predict(svm, X_train_pca,Y_train, cv=skfold)

f1=f1_score(Y_train, Y_train_pred, average='weighted')
f1


# =============================================================================
# ################ Light GBM
# =============================================================================

import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics

model = lgb.LGBMClassifier(learning_rate=0.09,max_depth=-5,random_state=42)

scores=cross_val_score(model, X_train_pca, Y_train, cv=skfold)
scores
print(np.mean(scores))
print("Accuracy: %.3f%% (%.3f%%)" % (scores.mean()*100.0, scores.std()*100.0))

Y_train_pred = cross_val_predict(model, X_train_pca,Y_train, cv=skfold)

f1=f1_score(Y_train, Y_train_pred, average='weighted')
f1

########### Feature Importance Plot

model.fit(X_train_pca, Y_train)
lgb.plot_importance(model)

lgb.plot_metric(model)

import graphviz
lgb.plot_tree(model,figsize=(30,40))







# =============================================================================
# ################ Test Data
# =============================================================================

# =============================================================================
# ################ test ['song_id', 'album', 'artist', 'lyrics', 'track']
# =============================================================================

test=pd.read_csv("test.csv", header=0)
test.sample(5)
test.head(5)
test

test.shape

test.columns
test['target'].value_counts()
test['album'].value_counts()
test['artist'].value_counts()
test['track'].value_counts()


# =============================================================================
# ################ sample_submission ['song_id', 'album', 'artist', 'lyrics', 'track']
# =============================================================================

sample_submission=pd.read_csv("sample_submission.csv", header=0)
sample_submission.sample(5)
sample_submission.head(5)
sample_submission

sample_submission.shape

sample_submission.columns
sample_submission['target'].value_counts()


# =============================================================================
# ################ Merge into one file
# =============================================================================

metadata1.shape

metadata1.rename(columns={'id':'song_id'}, inplace=True)
metadata2.rename(columns={'id':'song_id'}, inplace=True)

data_test=test.merge(metadata1)
data_test=data_test.merge(metadata2)

data_test.shape

data_test.head

# Categorical Data
data_test['album'].value_counts()
data_test['artist'].value_counts()
data_test['lyrics'].value_counts()
data_test['track'].value_counts()

# =============================================================================
# ################ Describe the data file
# =============================================================================

data_test.info()

missing_data = pd.DataFrame({'total_missing': data_test.isnull().sum(), 'perc_missing': (data_test.isnull().sum()/3472)*100})
missing_data

# Statistical description of numerical variables
data_test.describe()

# Boxplot of numerical variables


# Handling outlier



# =============================================================================
# ################ Remove Text columns
# =============================================================================

data_test.info()

data_test.drop(['song_id', 'artist','album','lyrics','track','release_date'], axis=1, inplace=True)

'''
from sklearn.preprocessing import LabelEncoder
#le=LabelEncoder()
data_test['artist']=le1.fit_transform(data_test['artist'])
'''


# Boolean variable to integer
data_test['explicit']

data_test['explicit'] = data_test['explicit'].astype(int)
data_test.info()


# =============================================================================
# ################ Fill NA (Mean filling)-------->   1
# =============================================================================

data_test.info()

data_test.replace([np.inf, -np.inf], np.nan, inplace=True)


data_test['danceability'] = data_test['danceability'].fillna(data_train['danceability'].mean())
data_test['duration'] = data_test['duration'].fillna(data_train['duration'].mean())
data_test['energy'] = data_test['energy'].fillna(data_train['energy'].mean())
data_test['loudness'] = data_test['loudness'].fillna(data_train['loudness'].mean())
data_test['positiveness'] = data_test['positiveness'].fillna(data_train['positiveness'].mean())
data_test['speechiness'] = data_test['speechiness'].fillna(data_train['speechiness'].mean())
data_test['tempo'] = data_test['tempo'].fillna(data_train['tempo'].mean())
data_test['a1'] = data_test['a1'].fillna(data_train['a1'].mean())
data_test['a11'] = data_test['a11'].fillna(data_train['a11'].mean())
data_test['a12'] = data_test['a12'].fillna(data_train['a12'].mean())
data_test['a13'] = data_test['a13'].fillna(data_train['a13'].mean())
data_test['a14'] = data_test['a14'].fillna(data_train['a14'].mean())
data_test['a15'] = data_test['a15'].fillna(data_train['a15'].mean())
data_test['a16'] = data_test['a16'].fillna(data_train['a16'].mean())
data_test['a17'] = data_test['a17'].fillna(data_train['a17'].mean())
data_test['a19'] = data_test['a19'].fillna(data_train['a19'].mean())
data_test['a2'] = data_test['a2'].fillna(data_train['a2'].mean())
data_test['a20'] = data_test['a20'].fillna(data_train['a20'].mean())
data_test['a21'] = data_test['a21'].fillna(data_train['a21'].mean())
data_test['a22'] = data_test['a22'].fillna(data_train['a22'].mean())
data_test['a23'] = data_test['a23'].fillna(data_train['a23'].mean())
data_test['a24'] = data_test['a24'].fillna(data_train['a24'].mean())
data_test['a25'] = data_test['a25'].fillna(data_train['a25'].mean())
data_test['a26'] = data_test['a26'].fillna(data_train['a26'].mean())
data_test['a27'] = data_test['a27'].fillna(data_train['a27'].mean())
data_test['a28'] = data_test['a28'].fillna(data_train['a28'].mean())
data_test['a3'] = data_test['a3'].fillna(data_train['a3'].mean())
data_test['a31'] = data_test['a31'].fillna(data_train['a31'].mean())
data_test['a32'] = data_test['a32'].fillna(data_train['a32'].mean())
data_test['a33'] = data_test['a33'].fillna(data_train['a33'].mean())
data_test['a35'] = data_test['a35'].fillna(data_train['a35'].mean())
data_test['a36'] = data_test['a36'].fillna(data_train['a36'].mean())
data_test['a37'] = data_test['a37'].fillna(data_train['a37'].mean())
data_test['a38'] = data_test['a38'].fillna(data_train['a38'].mean())
data_test['a39'] = data_test['a39'].fillna(data_train['a39'].mean())
data_test['a4'] = data_test['a4'].fillna(data_train['a4'].mean())
data_test['a40'] = data_test['a40'].fillna(data_train['a40'].mean())
data_test['a41'] = data_test['a41'].fillna(data_train['a41'].mean())
data_test['a42'] = data_test['a42'].fillna(data_train['a42'].mean())
data_test['a43'] = data_test['a43'].fillna(data_train['a43'].mean())
data_test['a45'] = data_test['a45'].fillna(data_train['a45'].mean())
data_test['a46'] = data_test['a46'].fillna(data_train['a46'].mean())
data_test['a47'] = data_test['a47'].fillna(data_train['a47'].mean())
data_test['a48'] = data_test['a48'].fillna(data_train['a48'].mean())
data_test['a49'] = data_test['a49'].fillna(data_train['a49'].mean())
data_test['a5'] = data_test['a5'].fillna(data_train['a5'].mean())
data_test['a50'] = data_test['a50'].fillna(data_train['a50'].mean())
data_test['a6'] = data_test['a6'].fillna(data_train['a6'].mean())
data_test['a7'] = data_test['a7'].fillna(data_train['a7'].mean())
data_test['a8'] = data_test['a8'].fillna(data_train['a8'].mean())

'''
data_test['danceability'] = data_test['danceability'].fillna(data_test['danceability'].mean())
data_test['duration'] = data_test['duration'].fillna(data_test['duration'].mean())
data_test['energy'] = data_test['energy'].fillna(data_test['energy'].mean())
data_test['loudness'] = data_test['loudness'].fillna(data_test['loudness'].mean())
data_test['positiveness'] = data_test['positiveness'].fillna(data_test['positiveness'].mean())
data_test['speechiness'] = data_test['speechiness'].fillna(data_test['speechiness'].mean())
data_test['tempo'] = data_test['tempo'].fillna(data_test['tempo'].mean())
data_test['a1'] = data_test['a1'].fillna(data_test['a1'].mean())
data_test['a11'] = data_test['a11'].fillna(data_test['a11'].mean())
data_test['a12'] = data_test['a12'].fillna(data_test['a12'].mean())
data_test['a13'] = data_test['a13'].fillna(data_test['a13'].mean())
data_test['a14'] = data_test['a14'].fillna(data_test['a14'].mean())
data_test['a15'] = data_test['a15'].fillna(data_test['a15'].mean())
data_test['a16'] = data_test['a16'].fillna(data_test['a16'].mean())
data_test['a17'] = data_test['a17'].fillna(data_test['a17'].mean())
data_test['a19'] = data_test['a19'].fillna(data_test['a19'].mean())
data_test['a2'] = data_test['a2'].fillna(data_test['a2'].mean())
data_test['a20'] = data_test['a20'].fillna(data_test['a20'].mean())
data_test['a21'] = data_test['a21'].fillna(data_test['a21'].mean())
data_test['a22'] = data_test['a22'].fillna(data_test['a22'].mean())
data_test['a23'] = data_test['a23'].fillna(data_test['a23'].mean())
data_test['a24'] = data_test['a24'].fillna(data_test['a24'].mean())
data_test['a25'] = data_test['a25'].fillna(data_test['a25'].mean())
data_test['a26'] = data_test['a26'].fillna(data_test['a26'].mean())
data_test['a27'] = data_test['a27'].fillna(data_test['a27'].mean())
data_test['a28'] = data_test['a28'].fillna(data_test['a28'].mean())
data_test['a3'] = data_test['a3'].fillna(data_test['a3'].mean())
data_test['a31'] = data_test['a31'].fillna(data_test['a31'].mean())
data_test['a32'] = data_test['a32'].fillna(data_test['a32'].mean())
data_test['a33'] = data_test['a33'].fillna(data_test['a33'].mean())
data_test['a35'] = data_test['a35'].fillna(data_test['a35'].mean())
data_test['a36'] = data_test['a36'].fillna(data_test['a36'].mean())
data_test['a37'] = data_test['a37'].fillna(data_test['a37'].mean())
data_test['a38'] = data_test['a38'].fillna(data_test['a38'].mean())
data_test['a39'] = data_test['a39'].fillna(data_test['a39'].mean())
data_test['a4'] = data_test['a4'].fillna(data_test['a4'].mean())
data_test['a40'] = data_test['a40'].fillna(data_test['a40'].mean())
data_test['a41'] = data_test['a41'].fillna(data_test['a41'].mean())
data_test['a42'] = data_test['a42'].fillna(data_test['a42'].mean())
data_test['a43'] = data_test['a43'].fillna(data_test['a43'].mean())
data_test['a45'] = data_test['a45'].fillna(data_test['a45'].mean())
data_test['a46'] = data_test['a46'].fillna(data_test['a46'].mean())
data_test['a47'] = data_test['a47'].fillna(data_test['a47'].mean())
data_test['a48'] = data_test['a48'].fillna(data_test['a48'].mean())
data_test['a49'] = data_test['a49'].fillna(data_test['a49'].mean())
data_test['a5'] = data_test['a5'].fillna(data_test['a5'].mean())
data_test['a50'] = data_test['a50'].fillna(data_test['a50'].mean())
data_test['a6'] = data_test['a6'].fillna(data_test['a6'].mean())
data_test['a7'] = data_test['a7'].fillna(data_test['a7'].mean())
data_test['a8'] = data_test['a8'].fillna(data_test['a8'].mean())
'''


data_test.info()
data_train.info()

X_train.shape
type(X_train)
X_train.info()
data_test.shape
type(data_test)
data_test.info()

data_test.to_csv('data_test.csv')




# =============================================================================
# ################ LDA --------1
# =============================================================================
data_test_scaled = scaler.transform(data_test)

X_test_lda = lda.transform(data_test_scaled)

# =============================================================================
# ################ PCA and t-SNE
# =============================================================================
# Scale the test data using the same StandardScaler object
data_test_scaled = scaler.transform(data_test)

# Apply PCA transformation to the scaled test data
X_test_pca = pca.transform(data_test_scaled)




##### For Test data transformation------------t-SNE

X_test_tsne = tsne.fit_transform(data_test)


X_test_pca_tsne = np.hstack((X_test_pca, X_test_tsne))


# =============================================================================
# ################ PREDICT
# =============================================================================

classifier.fit(X_train_lda, Y_train)
Y_test_pred = classifier.predict(X_test_lda)

original_labels=le2.inverse_transform(Y_test_pred)

sample_submission['target']=original_labels
sample_submission.to_csv('submission_logistic_lda.csv', index=False)



xgb_model.fit(X_train_pca, Y_train)
Y_test_pred = xgb_model.predict(X_test_pca)

original_labels=le2.inverse_transform(Y_test_pred)

sample_submission['target']=original_labels
sample_submission.to_csv('submission_xgm_11.csv', index=False)



svm.fit(X_train_lda, Y_train)
Y_test_pred = svm.predict(X_test_lda)

original_labels=le2.inverse_transform(Y_test_pred)

sample_submission['target']=original_labels
sample_submission.to_csv('submission_svm_3.csv', index=False)





# =============================================================================
# ###### Split
# =============================================================================

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_train_lda, Y_train, test_size=0.3, random_state=1, stratify=y)
X_train.shape
X_test.shape
y_train.shape

y_test.shape
y_test.value_counts()


# =============================================================================
# ################ DL using Keras
# =============================================================================
###     https://www.kaggle.com/karthik7395/binary-classification-using-neural-networks

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers

hidden_units=100
learning_rate=0.01
#hidden_layer_act='tanh'
hidden_layer_act='relu'
output_layer_act='softmax'
no_epochs=100

model = Sequential()

model.add(Dense(hidden_units, input_dim=21, activation=hidden_layer_act))
model.add(Dense(hidden_units, activation=hidden_layer_act))
model.add(Dense(16, activation=output_layer_act))

from tensorflow.keras import optimizers
sgd=optimizers.Adam(lr=learning_rate)
#sgd=optimizers.SGD(lr=learning_rate)
model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['acc'])

len(X_train)
model.fit(X_train, y_train, epochs=no_epochs, batch_size=len(X_train), verbose=0)

#test_x=test.iloc[:,1:]
predictions = model.predict(X_test)
predictions

rounded = [int(round(x[0])) for x in predictions]
print(rounded)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, rounded)
cm




















# =============================================================================
# ################ SVM
# =============================================================================

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

skfold=StratifiedKFold(n_splits=5)
from sklearn.svm import SVC
svm = SVC(random_state = 1)
scores=cross_val_score(svm,X_new,y,cv=skfold)
scores
print(np.mean(scores))
print("Accuracy: %.3f%% (%.3f%%)" % (scores.mean()*100.0, scores.std()*100.0))

y_pred = cross_val_predict(svm, X_new, y, cv=skfold)
conf_mat = confusion_matrix(y, y_pred)
conf_mat


# =============================================================================
# ################ Naive Bayes
# =============================================================================

from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
scores=cross_val_score(nb,X_new,y,cv=skfold)
scores
print(np.mean(scores))
print("Accuracy: %.3f%% (%.3f%%)" % (scores.mean()*100.0, scores.std()*100.0))

y_pred = cross_val_predict(nb, X_new, y, cv=skfold)
conf_mat = confusion_matrix(y, y_pred)
conf_mat

# =============================================================================
# ################ Decision Tree
# =============================================================================

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
scores=cross_val_score(dt,X_new,y,cv=skfold)
scores
print(np.mean(scores))
print("Accuracy: %.3f%% (%.3f%%)" % (scores.mean()*100.0, scores.std()*100.0))

y_pred = cross_val_predict(dt, X_new, y, cv=skfold)
conf_mat = confusion_matrix(y, y_pred)
conf_mat


# =============================================================================
# ################ Random Forest
# =============================================================================

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 100, random_state = 1)
scores=cross_val_score(rf,X_new,y,cv=skfold)
scores
print(np.mean(scores))
print("Accuracy: %.3f%% (%.3f%%)" % (scores.mean()*100.0, scores.std()*100.0))

y_pred = cross_val_predict(rf, X_new, y, cv=skfold)
conf_mat = confusion_matrix(y, y_pred)
conf_mat

# =============================================================================
# ################ AdaBoost
# =============================================================================

from sklearn.ensemble import AdaBoostClassifier
abc = AdaBoostClassifier(n_estimators=50, learning_rate=1, random_state=0)
scores=cross_val_score(abc,X_new,y,cv=skfold)
scores
print(np.mean(scores))
print("Accuracy: %.3f%% (%.3f%%)" % (scores.mean()*100.0, scores.std()*100.0))

y_pred = cross_val_predict(abc, X_new, y, cv=skfold)
conf_mat = confusion_matrix(y, y_pred)
conf_mat




# =============================================================================
# ###### Split
# =============================================================================

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.3, random_state=1, stratify=y)
X_train.shape
X_test.shape
y_train.shape

y_test.shape
y_test.value_counts()


# =============================================================================
# ################ DL using Keras
# =============================================================================
###     https://www.kaggle.com/karthik7395/binary-classification-using-neural-networks

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers

hidden_units=100
learning_rate=0.01
#hidden_layer_act='tanh'
hidden_layer_act='relu'
output_layer_act='softmax'
no_epochs=100

model = Sequential()

model.add(Dense(hidden_units, input_dim=21, activation=hidden_layer_act))
model.add(Dense(hidden_units, activation=hidden_layer_act))
model.add(Dense(16, activation=output_layer_act))

from tensorflow.keras import optimizers
sgd=optimizers.Adam(lr=learning_rate)
#sgd=optimizers.SGD(lr=learning_rate)
model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['acc'])

len(X_train)
model.fit(X_train, y_train, epochs=no_epochs, batch_size=len(X_train), verbose=0)

#test_x=test.iloc[:,1:]
predictions = model.predict(X_test)
predictions

rounded = [int(round(x[0])) for x in predictions]
print(rounded)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, rounded)
cm













# =============================================================================
# ################ ROUGH
# =============================================================================

# =============================================================================
# ################ F1 Score
# =============================================================================

## Confusion Matrix
conf_mat = confusion_matrix(y_sm, y_pred)
conf_mat

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Membuat confusion matrix
cm_df = pd.DataFrame(conf_mat, index=range(0, 16), columns=range(0, 16))

## Graphical
# Menampilkan confusion matrix dalam bentuk heatmap

plt.figure(figsize=(10,7))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

classification_report(y_sm, y_pred)

############# Cross validation score #######
skfold = StratifiedKFold(n_splits=5)
scores = cross_val_score(classifier, X_sm, y_sm, cv=skfold, scoring='f1_macro')

print("F1 Scores:", scores)
print("Mean F1 Macro Score: %.3f" % np.mean(scores))
print("Mean Accuracy: %.3f%% (%.3f%%)" % (scores.mean()*100.0, scores.std()*100.0))


####
from sklearn.metrics import precision_recall_fscore_support as score

precision, recall, fscore, support = score(y_sm, y_pred)

print('precision: {}'.format(precision))
print('recall: {}'.format(recall))
print('fscore: {}'.format(fscore))
print('support: {}'.format(support))


# =============================================================================
# ################ Logistic Regression (Variable Significant----Check)
# =============================================================================
# --------- First Method ----------

import statsmodels.api as sm
########## Here, we have to add constant separately
X = sm.add_constant(X)
logit_model=sm.Logit(y,X)

logit_model=sm.Probit(y,X)


result=logit_model.fit()
print(result.summary2())




###########
data[data['target']==1].shape
data[data['target']==0].shape

# Remove column name 'A'
data.drop(['Outcome'], axis=1)

# drop variable
data.drop(["ID","Phase","State","district","ageNormal","familyNormal","incomeNormal","roomsNormal"], axis = 1, inplace = True)

data.columns

X = data.drop('Outcome',axis='columns')
y = data['Outcome']
X.shape
y.shape
type(X)
type(y)
type(data)


# Visualisasi hasil reduksi dimensionalitas
plt.figure(figsize=(8, 6))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis')  # y adalah label kelas
plt.colorbar()
plt.title('t-SNE Visualization')
plt.show()




