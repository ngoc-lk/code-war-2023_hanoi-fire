# -*- coding: utf-8 -*-

# Import Packages
import pandas as pd
import numpy as np
import sklearn.preprocessing as preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from catboost import CatBoostClassifier

def load_data(filepath):
    # Load dataset
    data = pd.read_csv(filepath, skipinitialspace=True)
    # Remove any 'Unnamed:' columns
    data = data.loc[:, ~data.columns.str.startswith('Unnamed: ')]
    # Drop uninterested columns
    data = data.drop(['Fire_Occurrence', 'Type_of_Fire', 'Fire_Scale', 'Number_of_People_Rescued'], axis=1)

    return data

def preprocess_data(data, target_col):
    # Split X and y
    X = data.drop(target_col, axis=1)
    y = data[target_col].values

    # Encode categorical target
    label_encoder = preprocessing.LabelEncoder()
    y = label_encoder.fit_transform(y.astype(str).ravel())

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Classify column types
    cat_columns = X_train.select_dtypes(include='object').columns
    num_columns = X_train.select_dtypes(exclude='object').columns

    # Remove outliers
    trainset = pd.concat(
        [X_train, pd.Series(y_train, name=target_col, index=X_train.index)],
        axis=1)
    for col in num_columns:
        Q1 = trainset[col].quantile(0.25)
        Q3 = trainset[col].quantile(0.75)
        IQR = Q3 - Q1
        trainset = trainset[(trainset[col] >= Q1-5.0*IQR) &
                            (trainset[col] <= Q3+5.0*IQR)]
    X_train = trainset.drop(target_col, axis=1)
    y_train = trainset[target_col].values

    # Encode categorical columns
    encoder = preprocessing.OrdinalEncoder(
        handle_unknown='use_encoded_value', unknown_value=-1)
    train_cat_values = encoder.fit_transform(X_train[cat_columns])
    test_cat_values = encoder.transform(X_test[cat_columns])

    # Scale numeric columns
    scaler = preprocessing.StandardScaler()
    train_num_values = scaler.fit_transform(X_train[num_columns])
    test_num_values = scaler.transform(X_test[num_columns])

    # Rebuild train and test dataset
    X_train = np.hstack((train_cat_values, train_num_values))
    X_test = np.hstack((test_cat_values, test_num_values))

    return X_train, X_test, y_train, y_test, label_encoder

def train_model(X_train, y_train):
    # Build a model with best possible parameters
    model = CatBoostClassifier(**{
        "iterations": 100,
        "learning_rate": 0.1,
        "depth": 6,
        "l2_leaf_reg": 3.0,
        "verbose": 0
    })

    # And train (fit) it
    model.fit(X_train, y_train)

    return model

def evaluate_model(model, X_test, y_test, label_encoder):
    # Predict using the test dataset
    y_test = label_encoder.inverse_transform(y_test)
    y_pred = label_encoder.inverse_transform(model.predict(X_test))
    y_yhat = pd.DataFrame({'Real': y_test, 'Pred': y_pred})
    print(y_yhat.reset_index(drop=True).round(2), '\n')

    # Calculate performance metrics
    print("Classification Report")
    print(classification_report(y_test, y_pred), '\n')

def main():
    # Load and explore data
    data = load_data("hanoi_fire.csv")
    print('Dataset Shape:', data.shape, '\n')
    print(data.head().round(2), '\n')
    data.info()
    print()
    print(data.describe(), '\n')

    # Preprocess data
    X_train, X_test, y_train, y_test, label_encoder = preprocess_data(data, 'Damage_Scale')

    # Train and evaluate model
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test, label_encoder)

if __name__ == "__main__":
    main()
