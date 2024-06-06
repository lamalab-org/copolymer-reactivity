import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def categorize_precision(precision):
    if precision <= 0.5:
        return 0  # under 50%
    elif 0.5 < precision <= 0.7:
        return 1  # 50-70%
    elif 0.7 < precision <= 0.9:
        return 2  # 70-90%
    else:
        return 3  # over 90%


file_path = 'enhanced_doi_list.json'
with open(file_path, 'r') as file:
    data = json.load(file)

# Filter for entries with precision & convert language to lower case
filtered_data = [entry for entry in data if 'precision' in entry and entry['precision'] is not None]
for entry in filtered_data:
    entry['language'] = entry['language'].lower()

df = pd.DataFrame(filtered_data)

# define features and target value
features = ['language', 'table_quality', 'quality_of_number', 'year', 'pdf_quality', 'rxn_number']
target = 'precision'

# drop entries with missing feature or target value
df = df.dropna(subset=features + [target])

# Create binary target variable
df['precision_class'] = (df[target] > 0.5).astype(int) # class '0': prec. < 70%; class '1': prec. > 70%
#df['precision_class'] = df[target].apply(categorize_precision)


# Split dataset in train, test and validation set
train_data, test_data = train_test_split(df, test_size=0.2, random_state=None)


# define features
numeric_features = ['table_quality', 'quality_of_number', 'year', 'pdf_quality', 'rxn_number']
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
#model = RandomForestRegressor(n_estimators=200, random_state=42)
#pipeline = Pipeline(steps=[('preprocessor', preprocessor),('model', model)])


# Define random forest classifier
model = RandomForestClassifier(n_estimators=200, random_state=None, max_depth=5, min_samples_leaf=2)
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', model)])


# Initialize lists to store metrics
accuracies = []
train_accuracies = []
conf_matrices = []
FP_rate = []

# Number of runs
n_runs = 50

for run in range(n_runs):
    # Split dataset into train and test set
    train_data, test_data = train_test_split(df, test_size=0.2, random_state=None)

    # Train model
    pipeline.fit(train_data[features], train_data['precision_class'])

    # Evaluate the model with test set
    predictions = pipeline.predict(test_data[features])
    accuracy = accuracy_score(test_data['precision_class'], predictions)
    accuracies.append(accuracy)

    train_predictions = pipeline.predict(train_data[features])
    accuracy_train = accuracy_score(train_data['precision_class'], train_predictions)
    train_accuracies.append(accuracy_train)

    conf_matrix = confusion_matrix(test_data['precision_class'], predictions)
    conf_matrices.append(conf_matrix)

    # Calculate FPR for class 0 (under 50%)
    if conf_matrix.shape[0] > 0:
        tn = conf_matrix.sum() - conf_matrix[0, :].sum() - conf_matrix[:, 0].sum() + conf_matrix[0, 0]
        fp = conf_matrix[:, 0].sum() - conf_matrix[0, 0]
        if (fp + tn) > 0:
            fpr = fp / (fp + tn)
        else:
            fpr = 0.0
        FP_rate.append(fpr)


    print("\nTest Data Predictions:")
    for pred, actual in zip(predictions, test_data['precision_class']):
        print(f"Predicted: {pred}, Test: {actual}")

    print(f'Run {run + 1}/{n_runs}')
    print(f'Accuracy on test data: {accuracy}')
    print(f'Accuracy on training data: {accuracy_train}')
    print(f'False Positive Rate (FPR): {fpr}')
    print('Confusion Matrix:')
    print(conf_matrix)
    print('-----------------------------')

# Calculate average metrics
average_accuracy = np.mean(accuracies)
average_train_accuracy = np.mean(train_accuracies)
#average_conf_matrix = np.mean(conf_matrices, axis=0)
average_FPR = np.mean(FP_rate)

print(f'Average accuracy on test data over {n_runs} runs: {average_accuracy}')
print(f'Average accuracy on training data over {n_runs} runs: {average_train_accuracy}')
print(f'Average false-positive-rate over {n_runs} runs: {average_FPR}')


for entry in data:
    if all(feature in entry for feature in features):
        feature_values = pd.DataFrame([{
            'language': entry['language'].lower(),
            'table_quality': entry['table_quality'],
            'quality_of_number': entry['quality_of_number'],
            'year': entry['year'],
            'pdf_quality': entry['pdf_quality'],
            'rxn_number': entry['rxn_number']
        }])
        prediction = pipeline.predict(feature_values)[0]
        entry['precision_score'] = int(prediction)
        print(prediction)
    else:
        # Wenn nicht alle Features vorhanden sind, setzen Sie 'precision_score' auf None oder lassen Sie den Eintrag unverändert
        entry['precision_score'] = None


with open('enhanced_doi_list.json', 'w') as file:
    json.dump(data, file, indent=4)


# train model
#pipeline.fit(train_data[features], train_data[target])

# evaluate the model with test set
#predictions = pipeline.predict(test_data[features])
#print(test_data[target])
#print(predictions)
#mae = mean_absolute_error(test_data[target], predictions)
#print(f'Mean Absolute Error on test data: {mae}')

#train_predictions = pipeline.predict(train_data[features])
#mae_train = mean_absolute_error(train_data[target], train_predictions)
#print(f'Mean Absolute Error on training data: {mae_train}')


#print("\nTest Data Predictions:")
#for pred, actual in zip(predictions, test_data[target]):
    #print(f"Predicted: {pred}, Test: {actual}")


# Train model
#pipeline.fit(train_data[features], train_data['precision_class'])

# Evaluate the model with test set
#predictions = pipeline.predict(test_data[features])
#print(test_data['precision_class'])
#print(predictions)
#accuracy = accuracy_score(test_data['precision_class'], predictions)
#print(f'Accuracy on test data: {accuracy}')
#print(classification_report(test_data['precision_class'], predictions))

# False Positive Rate (FPR)
#tn, fp, fn, tp = confusion_matrix(test_data['precision_class'], predictions).ravel()
#fpr = fp / (fp + tn)
#print(f'False Positive Rate (FPR): {fpr}')

#train_predictions = pipeline.predict(train_data[features])
#accuracy_train = accuracy_score(train_data['precision_class'], train_predictions)
#print(f'Accuracy on training data: {accuracy_train}')

#print("\nTest Data Predictions:")
#for pred, actual in zip(predictions, test_data['precision_class']):
    #print(f"Predicted: {pred}, Test: {actual}")