import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

#Read the training & test data
liver_df = pd.read_csv('./sample_data/liver_patient_data.csv')

# encode the categorial features
def partition(x):
    if x =='Male':
        return 0
    return 1

liver_df['Gender'] = liver_df['Gender'].map(partition)

liver_df["Albumin_and_Globulin_Ratio"] = liver_df.Albumin_and_Globulin_Ratio.fillna(liver_df['Albumin_and_Globulin_Ratio'].mean())
X = liver_df.drop('Dataset', axis=1)
y = liver_df['Dataset']
# finX = liver_df[['Total_Protiens','Albumin', 'Gender_Male']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)

# Random Forst Classifier
from sklearn import ensemble
rfc_model = ensemble.RandomForestClassifier()
rfc_model.fit(X_train, y_train)
y_pred = rfc_model.predict(X_test)

print('Accuracy: \n', accuracy_score(y_test,y_pred))

# Saving model to disk
pickle.dump(rfc_model, open('liver_model.pkl','wb'))

# Loading model to compare the results
model2 = pickle.load(open('liver_model.pkl','rb'))
xp=np.array([38,1,0.8,0.2,185,25,21,7.0,3.0,0.7]).reshape(-1,10)

result = model2.predict(xp)