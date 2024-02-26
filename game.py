print(1)
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
data = pd.read_csv('/content/diabetes.csv')
data.head()
X=data.drop('Outcome',axis=1)
y=data['Outcome']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
svm=SVC(kernel='linear',random_state=42)
svm.fit(X_train,y_train);
y_pred=svm.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)
print(accuracy)
print(classification_report(y_test, y_pred))
print("benny")
