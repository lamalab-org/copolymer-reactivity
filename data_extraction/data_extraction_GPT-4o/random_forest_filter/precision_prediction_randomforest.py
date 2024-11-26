import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, accuracy_score, confusion_matrix
import json


def categorize_precision(precision):
    if precision <= 0.5:
        return 0  # under 50%
    elif 0.5 < precision <= 0.7:
        return 1  # 50-70%
    elif 0.7 < precision <= 0.9:
        return 2  # 70-90%
    else:
        return 3  # over 90%


def false_positive_rate(y_true, y_pred):
    conf_matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = conf_matrix.ravel()
    if (fp + tn) > 0:
        fpr = fp / (fp + tn)
    else:
        fpr = 0.0
    return fpr


def combined_score(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    fpr = false_positive_rate(y_true, y_pred)
    # We want to maximize accuracy and minimize FPR, so we subtract FPR from accuracy
    return acc - fpr


file_path = '../collected_data/enhanced_doi_list_unique.json'
with open(file_path, 'r') as file:
    data = json.load(file)

filtered_data = [entry for entry in data if 'precision' in entry and entry['precision'] is not None]
for entry in filtered_data:
    entry['language'] = entry['language'].lower()

df = pd.DataFrame(filtered_data)

features = ['language', 'table_quality', 'quality_of_number', 'year', 'pdf_quality', 'rxn_number']
target = 'precision'

df = df.dropna(subset=features + [target])
df['precision_class'] = (df[target] > 0.7).astype(int)

numeric_features = ['table_quality', 'quality_of_number', 'year', 'pdf_quality', 'rxn_number']
categorical_features = ['language']

numeric_transformer = Pipeline(steps=[('imputer', 'passthrough')])
categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

model = RandomForestClassifier(random_state=22)

pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', model)])

param_grid = {
    'model__n_estimators': [25, 50, 100, 200, 300],
    'model__max_depth': [None, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30],
    'model__min_samples_leaf': [1, 2, 3, 4]
}

cv = StratifiedKFold(n_splits=10)
scorer = make_scorer(combined_score, greater_is_better=True)

# Split data into training and test sets for final evaluation
train_data, test_data = train_test_split(df, test_size=0.2, random_state=22)

grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring=scorer, n_jobs=-1)
grid_search.fit(train_data[features], train_data['precision_class'])

print(f'Best parameters: {grid_search.best_params_}')
print(f'Best cross-validation score: {grid_search.best_score_}')

best_model = grid_search.best_estimator_

# Evaluating the final model on the whole training dataset using cross-validation
cross_val_predictions = cross_val_predict(best_model, train_data[features], train_data['precision_class'], cv=cv)

conf_matrices = []
fpr_values = []

# Compute confusion matrix and FPR for each fold
for train_index, test_index in cv.split(train_data[features], train_data['precision_class']):
    X_test, y_test = train_data[features].iloc[test_index], train_data['precision_class'].iloc[test_index]
    predictions = cross_val_predictions[test_index]

    conf_matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    conf_matrices.append(conf_matrix)

    tn, fp, fn, tp = conf_matrix.ravel()
    if (fp + tn) > 0:
        fpr = fp / (fp + tn)
    else:
        fpr = 0.0
    fpr_values.append(fpr)

average_accuracy = np.mean(accuracy_score(train_data['precision_class'], cross_val_predictions))
std_accuracy = np.std(accuracy_score(train_data['precision_class'], cross_val_predictions))
average_fpr = np.mean(fpr_values)
std_fpr = np.std(fpr_values)

print(f'Average cross-validated accuracy: {average_accuracy}')
print(f'Standard deviation of cross-validated accuracy: {std_accuracy}')
print(f'Average cross-validated FPR: {average_fpr}')
print(f'Standard deviation of cross-validated FPR: {std_fpr}')

# Train the model on the entire training dataset
best_model.fit(train_data[features], train_data['precision_class'])

# Predict on the training set
train_predictions = best_model.predict(train_data[features])
train_accuracy = accuracy_score(train_data['precision_class'], train_predictions)
train_conf_matrix = confusion_matrix(train_data['precision_class'], train_predictions)

# Predict on the test set
test_predictions = best_model.predict(test_data[features])
test_accuracy = accuracy_score(test_data['precision_class'], test_predictions)
test_conf_matrix = confusion_matrix(test_data['precision_class'], test_predictions)

# Display training data predictions with actual precision
print("\nTraining Data Predictions:")
for pred, actual, precision in zip(train_predictions, train_data['precision_class'], train_data['precision']):
    print(f"Predicted: {pred}, Actual Class: {actual}, Actual Precision: {precision}")

# Display test data predictions with actual precision
print("\nTest Data Predictions:")
for pred, actual, precision in zip(test_predictions, test_data['precision_class'], test_data['precision']):
    print(f"Predicted: {pred}, Actual Class: {actual}, Actual Precision: {precision}")


print(f'Training accuracy: {train_accuracy}')
print(f'Test accuracy: {test_accuracy}')
print('Training confusion matrix:')
print(train_conf_matrix)
print('Test confusion matrix:')
print(test_conf_matrix)

# Update the json file with the predictions
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
        prediction = best_model.predict(feature_values)[0]
        entry['precision_score'] = int(prediction)
    else:
        entry['precision_score'] = None

with open('../collected_data/enhanced_doi_list_unique.json', 'w') as file:
    json.dump(data, file, indent=4)
