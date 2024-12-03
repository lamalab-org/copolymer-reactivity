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
    """
    Categorize precision values into predefined classes.
    """
    if precision <= 0.5:
        return 0  # under 50%
    elif 0.5 < precision <= 0.7:
        return 1  # 50-70%
    elif 0.7 < precision <= 0.9:
        return 2  # 70-90%
    else:
        return 3  # over 90%


def false_positive_rate(y_true, y_pred):
    """
    Calculate the false positive rate.
    """
    conf_matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = conf_matrix.ravel()
    return fp / (fp + tn) if (fp + tn) > 0 else 0.0


def combined_score(y_true, y_pred):
    """
    Custom scoring function to maximize accuracy and minimize FPR.
    """
    acc = accuracy_score(y_true, y_pred)
    fpr = false_positive_rate(y_true, y_pred)
    return acc - fpr


def load_data(input_path):
    """
    Load and preprocess data from the specified JSON file.
    """
    with open(input_path, 'r') as file:
        data = json.load(file)

    filtered_data = [entry for entry in data if 'precision' in entry and entry['precision'] is not None]
    for entry in filtered_data:
        entry['language'] = entry['language'].lower()

    return pd.DataFrame(filtered_data), data


def preprocess_data(df, features, target):
    """
    Prepare the DataFrame for training by removing NaNs and creating a target class.
    """
    df = df.dropna(subset=features + [target])
    df['precision_class'] = (df[target] > 0.7).astype(int)
    return df


def build_pipeline(numeric_features, categorical_features):
    """
    Build a preprocessing and modeling pipeline.
    """
    numeric_transformer = Pipeline(steps=[('imputer', 'passthrough')])
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    model = RandomForestClassifier(random_state=22)
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
    return pipeline


def update_json_with_predictions(data, features, best_model):
    """
    Update the JSON file with predictions for each entry.
    """
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
    return data


def main(input_path, output_path):
    """
    Main function to handle data loading, preprocessing, model training, and saving results.
    """
    # Load data
    print("Loading data...")
    df, original_data = load_data(input_path)

    # Define features and target
    features = ['language', 'table_quality', 'quality_of_number', 'year', 'pdf_quality', 'rxn_number']
    target = 'precision'

    # Preprocess data
    print("Preprocessing data...")
    df = preprocess_data(df, features, target)

    # Build pipeline
    print("Building pipeline...")
    numeric_features = ['table_quality', 'quality_of_number', 'year', 'pdf_quality', 'rxn_number']
    categorical_features = ['language']
    pipeline = build_pipeline(numeric_features, categorical_features)

    # Define parameter grid for hyperparameter tuning
    param_grid = {
        'model__n_estimators': [25, 50, 100, 200],
        'model__max_depth': [None, 5, 10, 20],
        'model__min_samples_leaf': [1, 2, 3, 4]
    }

    # Split data into training and test sets
    train_data, test_data = train_test_split(df, test_size=0.2, random_state=22)

    # Cross-validation and grid search
    print("Starting grid search...")
    cv = StratifiedKFold(n_splits=10)
    scorer = make_scorer(combined_score, greater_is_better=True)
    grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring=scorer, n_jobs=-1)
    grid_search.fit(train_data[features], train_data['precision_class'])

    # Best model and performance
    print(f'Best parameters: {grid_search.best_params_}')
    print(f'Best cross-validation score: {grid_search.best_score_}')
    best_model = grid_search.best_estimator_

    # Evaluate on test data
    print("Evaluating model...")
    test_predictions = best_model.predict(test_data[features])
    test_accuracy = accuracy_score(test_data['precision_class'], test_predictions)
    print(f'Test accuracy: {test_accuracy}')
    print('Test confusion matrix:')
    print(confusion_matrix(test_data['precision_class'], test_predictions))

    # Update JSON with predictions
    print("Updating JSON with predictions...")
    updated_data = update_json_with_predictions(original_data, features, best_model)

    # Save updated JSON
    print("Saving updated data...")
    with open(output_path, 'w') as file:
        json.dump(updated_data, file, indent=4)
    print(f"Updated data saved to {output_path}")


if __name__ == "__main__":
    # Input and output file paths
    input_path = '../collected_data/enhanced_doi_list_unique.json'
    output_path = '../collected_data/enhanced_doi_list_unique_updated.json'

    # Run the main function
    main(input_path, output_path)
