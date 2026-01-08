# Deep-Learning-2025-2026
#SIMPLE LINEAR REGRESSION (PROJECT 1)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#create simple dataset manually
data_set= {
    'sqft_living': [650,800,1000,1400,1600,1800,2000,2200,2500, 2800],
    'price': [90000, 120000, 150000, 170000, 190000, 210000, 240000, 270000, 290000, 320000]
}

#convert to dataframe
home_data = pd.DataFrame(data_set)
print("Sample dataset created successfully")
print(home_data)

#prepare feature (x) and target (y)
X = home_data ['sqft_living'].values.reshape(-1, 1) #feature must be 2D
Y = home_data ['price'].values #target 1D

#splitting the dataset into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/3, random_state=0) #have to import model selection library for training and testing the data

#fitting the simple linear regression to the training dataset
from sklearn.linear_model import LinearRegression
monali = LinearRegression()
monali.fit(X_train, Y_train)

#prediction of test and training set results
Y_train_pred = monali.predict(X_train)
Y_test_pred = monali.predict(X_test)

#print model parameters and simple metrics
print("\nModel slope (coefficient):", monali.coef_[0])
print("Model intercept:", monali.intercept_)

from sklearn.metrics import mean_squared_error, r2_score
print("\nTrain R^2", r2_score(Y_train, Y_train_pred))
print("Test R^2:", r2_score(Y_test, Y_test_pred))
print("Train MSE:", np.sqrt(mean_squared_error(Y_test, Y_test_pred)))

#visualize the training results with a smooth regression line
plt.figure(figsize=(8,5))
plt.scatter(X_train, Y_train, label ='Training data', color='green')

#create smooth line for regression
line_x= np.linspace(X.min(), X.max(), 100).reshape(-1,1)
line_y= monali.predict(line_x)
plt.plot(line_x, line_y, color= 'red', linewidth =2, label ='regression line')

plt.title('House price vs Living Area (Training Set)')
plt.xlabel('Living Area (sqft)')
plt.ylabel('House Price ($)')
plt.legend()
plt.grid(True)
plt.show()

#visualize the test results (with same regression line)
plt.figure(figsize=(8,5))
plt.scatter(X_test, Y_test, label= 'Test data' , color= 'blue')
plt.plot(line_x, line_y, color= 'red', linewidth =2, label ='regression line')
plt.title("House price vs Living Area (Test set)")
plt.xlabel("Living Area (sqft)")
plt.ylabel("House Price ($)")


#LOGISTIC REGRESSION (PROJECT 2)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#load the Titanic dataset
data = pd.read_csv("/content/sample_data/tested.csv")

#display dataset info and statistics
print(data.info())
print(data.describe())

#select features
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
data = data[features + ['Survived']]

#handling  missing values
data['Age'].fillna(data['Age'].median(), inplace=True)
data['Fare'].fillna(data['Fare'].median(), inplace=True)


#convert categorical coloumn 'Sex' to numerical
le = LabelEncoder()
data['Sex'] = le.fit_transform(data['Sex']) #male = 0, female = 1

#split features and target
x = data[features]
y = data['Survived']

#Train- Test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

#Train Logistic Regression
model = LogisticRegression(max_iter=200)
model.fit(x_train, y_train)

#make predictions
y_pred = model.predict(x_test)

#evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Did not Survived', 'Survived'], yticklabels=['Did Not Survived', 'Survived'])
plt.xlabel('Predicted')
plt.ylabel("Actual")
plt.title('Confusion Matrix')
plt.show()


#DECISION TREES (PROJECT 3)
import pandas as pd
#create your own dataset
data = {
    'Study _Hours' : [1,2,3,4,5,6,7,8,9,10],
    'attendance(%)' : [55,60,65,70,75,80,85,90,95,98],
    'internal_marks' : [35, 45, 50, 55,60,70,75,80,85,90],
    'result' : ['pass', 'pass', 'pass', 'pass', 'pass', 'pass', 'fail', 'pass', 'pass', 'pass']

    }

# convert to dataframe
df = pd.DataFrame(data)

#show dataset
print("Student Performance dataset")
print(df)

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

#split the data into features (X) and target (Y)
x = df[['Study _Hours', 'attendance(%)', 'internal_marks']]
y = df['result']

#split data
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=0)

#train model
model = DecisionTreeClassifier(criterion = 'entropy', random_state=0)
model.fit(x_train, y_train)

#make predictions
y_pred = model.predict(x_test)

#evaluate
print("\nAccuracy: ", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix: ", confusion_matrix(y_test, y_pred))

#visualize
plt.figure(figsize=(10,6))
plot_tree(model, feature_names=x.columns, class_names=['fail', 'pass'], filled=True)
plt.show()
plt.legend()
plt.grid(True)
plt.show()



#RANDOM FOREST (PROJECT 4)
#importing necessary libraries
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#Load the famous Iris dataset
iris = load_iris()

#create dataframe for better visualize
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target

#display first 5 rows
print("Sample of dataset: ")
print(df.head())

#split data into features (x) and target (y)
x = df.iloc[:, :-1] #all coloumns except the last one
y = df.iloc[:, -1] #

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Train a new RandomForestClassifier model with the Iris data
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(x_train, y_train)

#prediction
y_pred = rf_model.predict(x_test)

#evaluation
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))



#RANDOM FOREST (PROJECT 4)
#importing necessary libraries
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#Load the famous Iris dataset
iris = load_iris()

#create dataframe for better visualize
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target

#display first 5 rows
print("Sample of dataset: ")
print(df.head())

#split data into features (x) and target (y)
x = df.iloc[:, :-1] #all coloumns except the last one
y = df.iloc[:, -1] #

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Train a new RandomForestClassifier model with the Iris data
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(x_train, y_train)

#prediction
y_pred = rf_model.predict(x_test)

#evaluation
print("Accuracy: ", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))




# KNN ALGORITHM

# Import required libraries
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, \
    classification_report, accuracy_score

# Load the Iris dataset
iris = load_iris()

# Convert dataset to DataFrame
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

#display 5 rows
print("sample dataset:")
print(df.head())

# split into features (X) and target(y)
X = df.iloc[:, :-1] # all columns except 'species'
y = df.iloc[:, -1]  # only the species column

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# feature scaling(important for knn)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# create and train knn classifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# make predictions
y_pred = knn.predict(X_test)

#evaluate
print("Accuracy of knn model:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))




#BAGGING
# Step 1: Import libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Step 2: Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Step 3: Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 4: Create base model (weak learner)
base_model = DecisionTreeClassifier()

# Step 5: Create bagging model
bag_model = BaggingClassifier(estimator=base_model, n_estimators=10, random_state=42)

# Step 6: Train both models
base_model.fit(X_train, y_train)
bag_model.fit(X_train, y_train)

# Step 7: Compare performance
y_pred_base = base_model.predict(X_test)
y_pred_bag = bag_model.predict(X_test)

print("Accuracy of single Decision Tree:", accuracy_score(y_test, y_pred_base))
print("Accuracy after Bagging:", accuracy_score(y_test, y_pred_bag))




#BAGGING
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

#define base learners
baselearners = [
    ('svm',  SVC(probability=True, random_state=42)), #added random_state for reproducibility
    ('knn' , KNeighborsClassifier(n_neighbors=5))
]

#meta learner (final model)
meta_model = LogisticRegression()

#create stacking model
stack_model = StackingClassifier(estimators=baselearners, final_estimator=meta_model)

#train
stack_model.fit(X_train, y_train)

#evaluate
y_pred_stack = stack_model.predict(X_test)
print("Accuracy with stacking: ", accuracy_score(y_test, y_pred_stack))




#Boosting
#1) AdaBoost
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#Create base model (weak learner)
base_model = DecisionTreeClassifier(max_depth=1)

#create adaboost model
boost_model = AdaBoostClassifier(base_model, n_estimators=50, learning_rate=1.0, random_state=42)

#train
boost_model.fit(X_train, y_train)

#evaluate
y_pred_boost = boost_model.predict(X_test)
print("Accuracy with AdaBoost:", accuracy_score(y_test, y_pred_boost))




#Boosting
#2) Gradient Boosting

#Import libraries
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

#Load dataset
iris = load_iris()
X, y = iris.data, iris.target

#Split into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create Gradient Boosting Model
gb_model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

# Train (fit) the model
gb_model.fit(X_train, Y_train)

# Make predictions
y_pred = gb_model.predict(X_test)

#Evaluate Performance
print("Confusion Matrix: ")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report: ")
print(classification_report(y_test, y_pred))
print("\nAccuracy of Gradient Boosting Model: ", accuracy_score(y_test, y_pred))


#Boosting
#3) XGBoost
!pip install XGBoost

import  xgboost as xgb
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


#Create XGBoost Model
xgb_model = xgb.XGBClassifier(
    n_estimators=100,  #number of trees
    learning_rate=0.1, #step size shrinkage
    max_depth=3,       #maximum depth of a tree
    subsample=0.8,     #subsample ratio of the training instances/ fraction pf training samples to use
    colsample_bytree=0.8, #subsample ratio(fractions) of features per tree
    random_state=42,
    eval_metric='mlogloss' #multiclass log loss metric (matrix to calculate xgboost)
)
 #Train(fit) the model
xgb_model.fit(X_train, y_train)

#make predictions
y_pred = xgb_model.predict(X_test)

#evaluate performance
print("Confusion Matrix")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report")
print(classification_report(y_test, y_pred))
print("\nAccuracy of XGBoost Model: ", accuracy_score(y_test, y_pred))




# ---SVM---

# Step 1: Import libraries
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

# Step 2: Load dataset (Iris)
iris = datasets.load_iris()
X = iris.data[:, :2]   # Take first two features for easy 2D visualization
y = iris.target

# Step 3: Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 4: Train SVM model
model = SVC(kernel='linear')   # Linear SVM
model.fit(X_train, y_train)

# Step 5: Predict and check accuracy
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Step 6: Visualize decision boundaries
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.title('SVM Decision Boundary (Linear Kernel)')
plt.show()
