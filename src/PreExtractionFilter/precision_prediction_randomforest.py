import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, accuracy_score, confusion_matrix


def load_data(file_path):
    """
    Load JSON data from a file.
    """
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def save_data(data, file_path):
    """
    Save JSON data to a file.
    """
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)


def preprocess_data(data, features, target):
    """
    Prepare data for training by converting it into a DataFrame.
    """
    filtered_data = [entry for entry in data if target in entry and entry[target] is not None]
    df = pd.DataFrame(filtered_data)
    df['precision_class'] = (df[target] > 0.7).astype(int)
    return df


def build_pipeline(numeric_features, categorical_features):
    """
    Build a preprocessing and modeling pipeline.
    """
    numeric_transformer = 'passthrough'
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    model = RandomForestClassifier(random_state=22)
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
    return pipeline


def train_model(training_data, features, target):
    """
    Train the Random Forest model.
    """
    numeric_features = ['table_quality', 'quality_of_number', 'year', 'pdf_quality', 'rxn_number']
    categorical_features = ['language']

    pipeline = build_pipeline(numeric_features, categorical_features)

    param_grid = {
        'model__n_estimators': [50, 100],
        'model__max_depth': [None, 10],
        'model__min_samples_leaf': [1, 2]
    }

    X = training_data[features]
    y = training_data[target]

    cv = StratifiedKFold(n_splits=5)
    scorer = make_scorer(accuracy_score)

    grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring=scorer, n_jobs=-1)
    print("Starting GridSearchCV...")
    grid_search.fit(X, y)
    print("GridSearchCV completed.")
    print("Best parameters found:", grid_search.best_params_)
    print("Best cross-validation score:", grid_search.best_score_)

    return grid_search.best_estimator_


def update_scores(data, model, features, pdf_folder):
    """
    Update entries with predictions and check if corresponding PDFs exist.
    """
    # Map feature names in the test data to match the training data
    feature_mapping = {'rxn_count': 'rxn_number'}

    for entry in data:
        # Rename feature keys in the entry to match training data
        for old_name, new_name in feature_mapping.items():
            if old_name in entry:
                entry[new_name] = entry.pop(old_name)

        # Check if all required features are present and valid
        missing_features = [feature for feature in features if feature not in entry or pd.isna(entry[feature])]
        if missing_features:
            print(f"Skipping entry due to missing features: {missing_features}")
            entry['precision_score'] = None
            continue

        # Create a DataFrame for prediction
        feature_values = pd.DataFrame([{
            feature: entry.get(feature, 0)  # Use 0 as the default value for missing features
            for feature in features
        }])

        print(f"Features for prediction: {feature_values}")  # Debug: Show feature data
        try:
            # Perform prediction
            prediction = model.predict(feature_values)[0]
            print(f"Prediction result: {prediction}")  # Debug: Show prediction result
            entry['precision_score'] = int(prediction)
        except Exception as e:
            # Handle any errors during prediction
            print(f"Error during prediction: {e}")
            entry['precision_score'] = None

    return data


def main(training_file, scoring_file, output_file, pdf_folder):
    """
    Main function to run the filtering pipeline.
    """
    # Load training and scoring data
    training_data = load_data(training_file)
    scoring_data = load_data(scoring_file)

    features = ['language', 'table_quality', 'quality_of_number', 'year', 'pdf_quality', 'rxn_number']
    target = 'precision'

    # Prepare training data
    training_df = preprocess_data(training_data, features, target)

    # Train the model
    print("Training the model...")
    model = train_model(training_df, features, 'precision_class')

    # Update scoring data
    print("Updating scores for scoring data...")
    updated_data = update_scores(scoring_data, model, features, pdf_folder)

    # Save updated scoring data
    save_data(updated_data, output_file)
    print(f"Updated scoring data saved to {output_file}")


if __name__ == "__main__":
    training_file = "../obtain_data/output/training_data.json"
    scoring_file = "../obtain_data/output/selected_200_papers.json"
    output_file = "./output/paper_list.json"
    pdf_folder = "../obtain_data/output/PDF"

    main(training_file, scoring_file, output_file, pdf_folder)
