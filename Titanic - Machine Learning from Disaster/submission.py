import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Separate features and target variable
features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
x= df[features]
y = df["Survived"]
x_test = test_df[features]

# Split the data into training and testing sets #test_size=0.2 means 20% of the data will be used for testing and 80% for training
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

# Define the column transformer for preprocessing
numeric_features = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
numeric_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy='median'))])

categorical_features = ["Sex", "Embarked"]
categorical_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy='most_frequent')), ("onehot", OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(transformers=[("num", numeric_transformer, numeric_features), ("cat", categorical_transformer, categorical_features)])

model = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", LogisticRegression(max_iter=1000))])

model.fit(x_train, y_train)
y_pred = model.predict(x_val)
accuracy = accuracy_score(y_val, y_pred)

print("Validation Accuracy:",accuracy)

model.fit(x, y)
test_predictions = model.predict(x_test)

submission = pd.DataFrame({"PassengerId": test_df["PassengerId"], "Survived": test_predictions})
submission.to_csv("submission.csv", index=False)
print("Submission file created successfully.")