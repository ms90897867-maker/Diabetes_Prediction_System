#upload file
import pandas as pd
import numpy as np

#Read file
df = pd.read_csv('diabetes.csv')
df.head()

#make copy of dataset
df_copy = df.copy()
print(df_copy)

#print dataset copy
print(df_copy.head())

#check null values
df_copy.isnull().sum()

#check duplicate values
df_copy.duplicated().sum()

#Remove unnecassary columns
#df_copy.drop('Pregnancies',axis=1,inplace=True)

print(df_copy.head())

num_cols = df_copy.select_dtypes(include=['int64','float64']).columns
for col in num_cols:
 df_copy[col]=df_copy[col].fillna(df_copy[col].median())
 print(df_copy[col])

#distributing X and y
X = df_copy.drop('Outcome',axis=1)
y = df_copy['Outcome']


print(X.shape)

print(y.shape)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y,
    test_size=0.2,
    random_state=42)


print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()

X_train = scaler.fit_transform(X_train)
print(X_train)

X_test = scaler.transform(X_test)
print(X_test)

#import knn
from sklearn.neighbors import KNeighborsClassifier


#make knn model
knn = KNeighborsClassifier(n_neighbors=4)

#train knn model
knn.fit(X_train,y_train)


#prediction
y_pred = knn.predict(X_test)
print(y_pred)

#comparision between actual and predicted
comparison = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': y_pred
})

print(comparison.head(10))

# Check accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
print("Accuracy:",accuracy)

#making cinfusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)

#making classification report
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))

#making pkl file
import joblib
#knn.fit(X_train,y_train)
#Model save karo
joblib.dump(knn,'diabetes.pkl')
joblib.dump(scaler,"scaler.pkl")
print("Model saved successfully as 'diabetes.pkl'")