import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score


# Loading data
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# Data preprocessing: Identifying features, missing values, converting categorical data to numeric values
X_train = train_df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']].copy()
X_train["Age"].fillna(X_train["Age"].median(), inplace=True)
X_train["Embarked"].fillna(X_train["Embarked"].mode()[0], inplace=True)
X_train['Sex'] = X_train['Sex'].map({'female': 0, 'male': 1}).astype(int)
X_train['Embarked'] = X_train['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

# Specifying the target variable
y_train = train_df['Survived'].copy()

# Creating the model
model = DecisionTreeClassifier(max_depth=3)
model.fit(X_train, y_train)

# Calculating the accuracy of the model
scores = cross_val_score(model, X_train, y_train, cv=10)
print("Accuracy: {:.2f}% (+/- {:.2f}%)".format(scores.mean()*100, scores.std()*100))

# Predicting the test data
X_test = test_df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']].copy()
X_test["Age"].fillna(X_test["Age"].median(), inplace=True)
X_test["Fare"].fillna(X_test["Fare"].median(), inplace=True)
X_test['Sex'] = X_test['Sex'].map({'female': 0, 'male': 1}).astype(int)
X_test['Embarked'] = X_test['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

# Predicting the test data
predictions = model.predict(X_test)

# Saving the predictions to a csv file
submission = pd.DataFrame({
    "PassengerId": test_df["PassengerId"],
    "Survived": predictions
})
submission.to_csv('submission.csv', index=False)
