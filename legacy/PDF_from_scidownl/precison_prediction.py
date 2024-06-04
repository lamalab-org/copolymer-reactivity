import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
import json


with open('enhanced_doi_list.json', 'r') as file:
    data = json.load(file)

# Filter for entries with precision & convert language to lower case
filtered_data = [entry for entry in data if 'precision' in entry and entry['precision'] is not None]
for entry in filtered_data:
    entry['language'] = entry['language'].lower()

df = pd.DataFrame(filtered_data)

# define features and target value
features = ['language', 'table_quality', 'quality_of_number', 'year']
target = 'precision'

# drop entries with missing feature or target value
df = df.dropna(subset=features + [target])

# Split dataset in train, test and validation set
train_data, test_data = train_test_split(df, test_size=0.2, random_state=None)


# define features
numeric_features = ['table_quality', 'quality_of_number', 'year']
categorical_features = ['language']

numeric_transformer = Pipeline(steps=[
    ('imputer', 'passthrough')  # Placeholder, da keine fehlenden Werte
])
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# define random forest model
model = RandomForestRegressor(n_estimators=200, random_state=42)
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', model)])

# train model
pipeline.fit(train_data[features], train_data[target])

# evaluate the model with test set
predictions = pipeline.predict(test_data[features])
print(test_data[target])
print(predictions)
mae = mean_absolute_error(test_data[target], predictions)
print(f'Mean Absolute Error on test data: {mae}')

train_predictions = pipeline.predict(train_data[features])
mae_train = mean_absolute_error(train_data[target], train_predictions)
print(f'Mean Absolute Error on training data: {mae_train}')


print("\nTest Data Predictions:")
for pred, actual in zip(predictions, test_data[target]):
    print(f"Predicted: {pred}, Test: {actual}")
