import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import pickle

# Load the dataset
titanic_data = pd.read_csv('tested.csv')

# Impute missing values
imputer = SimpleImputer(strategy='median')
titanic_data['Age'] = imputer.fit_transform(titanic_data[['Age']])
titanic_data['Fare'] = imputer.fit_transform(titanic_data[['Fare']])

# Drop the 'Cabin' column
titanic_data.drop(columns=['Cabin'], inplace=True)

# Encode categorical variables
label_encoder = LabelEncoder()
titanic_data['Sex'] = label_encoder.fit_transform(titanic_data['Sex'])
titanic_data['Embarked'] = label_encoder.fit_transform(titanic_data['Embarked'])

# Drop unnecessary columns
titanic_data.drop(columns=['PassengerId', 'Name', 'Ticket'], inplace=True)

# Prepare features and target
X = titanic_data.drop(columns=['Survived'])
y = titanic_data['Survived']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Save the model
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)
