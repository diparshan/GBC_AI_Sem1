import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

def load_data(file_path):
    """
    Load data from a CSV file.
    :param file_path: Path to the CSV file.
    :return: DataFrame containing the dataset.
    """
    return pd.read_csv(file_path)

def select_k_best_features(X, y, k=10):
    """
    Select K best features using the chi-squared test.
    :param X: Feature set.
    :param y: Target variable.
    :param k: Number of top features to select.
    :return: Selected feature names.
    """
    selector = SelectKBest(score_func=chi2, k=k)
    selector.fit(X, y)
    selected_features = X.columns[selector.get_support(indices=True)]
    return selected_features

def recursive_feature_elimination(X, y, n_features_to_select=10):
    """
    Perform recursive feature elimination (RFE) with Logistic Regression.
    :param X: Feature set.
    :param y: Target variable.
    :param n_features_to_select: Number of features to select.
    :return: Selected feature names.
    """
    model = LogisticRegression()
    rfe = RFE(estimator=model, n_features_to_select=n_features_to_select)
    rfe.fit(X, y)
    selected_features = X.columns[rfe.support_]
    return selected_features

def random_forest_feature_importance(X, y):
    """
    Select features based on Random Forest feature importance.
    :param X: Feature set.
    :param y: Target variable.
    :return: Feature importance scores and selected feature names.
    """
    model = RandomForestClassifier()
    model.fit(X, y)
    feature_importances = pd.Series(model.feature_importances_, index=X.columns)
    sorted_features = feature_importances.sort_values(ascending=False)
    return sorted_features

def lgbm_feature_importance(X, y):
    """
    Select features based on LightGBM feature importance.
    :param X: Feature set.
    :param y: Target variable.
    :return: Feature importance scores and selected feature names.
    """
    model = LGBMClassifier()
    model.fit(X, y)
    feature_importances = pd.Series(model.feature_importances_, index=X.columns)
    sorted_features = feature_importances.sort_values(ascending=False)
    return sorted_features

def main(file_path):
    """
    Main function to execute feature selection methods.
    :param file_path: Path to the dataset.
    """
    # Load the data
    data = load_data(file_path)
    X = data.iloc[:, :-1]  # Assuming the last column is the target
    y = data.iloc[:, -1]

    # Select K Best Features
    print("Selecting top K best features...")
    k_best_features = select_k_best_features(X, y)
    print("K Best Features:", k_best_features)

    # Recursive Feature Elimination
    print("Performing Recursive Feature Elimination...")
    rfe_features = recursive_feature_elimination(X, y)
    print("RFE Selected Features:", rfe_features)

    # Random Forest Feature Importance
    print("Evaluating Random Forest Feature Importance...")
    rf_importances = random_forest_feature_importance(X, y)
    print("Random Forest Feature Importances:")
    print(rf_importances)

    # LightGBM Feature Importance
    print("Evaluating LightGBM Feature Importance...")
    lgbm_importances = lgbm_feature_importance(X, y)
    print("LightGBM Feature Importances:")
    print(lgbm_importances)

if __name__ == "__main__":
    
    main("fifa19.csv")
