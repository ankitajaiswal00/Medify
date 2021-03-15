import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import pickle

df = pd.read_csv(r'./sample_data/heart.csv')

X = df.drop(['target','thal'], axis=1)
y = df.target
#df.head()

# Encode the categorial feature
# X_encoded = pd.get_dummies(X, colmns=['sex', 'cp','fbs','restecg','exang','slope','ca'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 1)

# Random Forest Classifier
classifier = DecisionTreeClassifier()
classifier.fit(X_train,y_train )
y_pred_classifier = classifier.predict(X_test)
classifier_cm = confusion_matrix(y_test, y_pred_classifier)
print(classifier_cm)

# Saving model to disk
pickle.dump(classifier, open('heart_model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('heart_model.pkl','rb'))