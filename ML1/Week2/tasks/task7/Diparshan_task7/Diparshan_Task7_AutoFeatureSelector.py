#!/usr/bin/env python3
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, RFE, SelectFromModel 
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
import argparse

def cor_selector(X, y, num_feats):
    """
    Select features based on Pearson correlation
    
    Parameters:
    -----------
    X : pd.DataFrame
        Input features
    y : pd.Series
        Target variable
    num_feats : int
        Number of top features to select
    
    Returns:
    --------
    tuple: Boolean support mask and selected feature names
    """
    cor_list = []
    feature_name = X.columns.tolist()
    
    # Calculate correlation between features and target
    for i in feature_name:
        cor = np.corrcoef(X[i], y)[0, 1]
        cor_list.append(cor)
        
    # Convert to dataframe
    feature_value = pd.DataFrame({
        'Feature': feature_name,
        'Correlation': np.abs(cor_list)
    })
    
    # Sort by absolute correlation value
    feature_value = feature_value.sort_values('Correlation', ascending=False)
    
    # Select top features
    topk_feature = feature_value.iloc[:num_feats, :]
    
    # Create boolean mask
    cor_support = [feat in topk_feature['Feature'].tolist() for feat in feature_name]
    cor_feature = topk_feature['Feature'].tolist()
    
    return cor_support, cor_feature

def chi_squared_selector(X, y, num_feats):
    """
    Select features using Chi-squared test
    
    Parameters:
    -----------
    X : pd.DataFrame
        Input features
    y : pd.Series
        Target variable
    num_feats : int
        Number of top features to select
    
    Returns:
    --------
    tuple: Boolean support mask and selected feature names
    """
    X_norm = MinMaxScaler().fit_transform(X)
    chi_selector = SelectKBest(chi2, k=num_feats)
    chi_selector.fit(X_norm, y)
    chi_support = chi_selector.get_support()
    chi_feature = X.loc[:,chi_support].columns.tolist()
    return chi_support, chi_feature

def rfe_selector(X, y, num_feats):
    """
    Select features using Recursive Feature Elimination
    
    Parameters:
    -----------
    X : pd.DataFrame
        Input features
    y : pd.Series
        Target variable
    num_feats : int
        Number of top features to select
    
    Returns:
    --------
    tuple: Boolean support mask and selected feature names
    """
    X_norm = MinMaxScaler().fit_transform(X)
    rfe_selector = RFE(
        estimator=LogisticRegression(random_state=42),
        n_features_to_select=num_feats,
        step=1
    )
    rfe_selector.fit(X_norm, y)
    rfe_support = rfe_selector.get_support()
    rfe_feature = X.loc[:,rfe_support].columns.tolist()
    return rfe_support, rfe_feature

def embedded_log_reg_selector(X, y, num_feats):
    """
    Select features using Logistic Regression Embedded method
    
    Parameters:
    -----------
    X : pd.DataFrame
        Input features
    y : pd.Series
        Target variable
    num_feats : int
        Number of top features to select
    
    Returns:
    --------
    tuple: Boolean support mask and selected feature names
    """
    X_norm = MinMaxScaler().fit_transform(X)
    embedded_lr_selector = SelectFromModel(
        LogisticRegression(penalty='l1', solver='liblinear', random_state=42),
        max_features=num_feats
    )
    embedded_lr_selector.fit(X_norm, y)
    embedded_lr_support = embedded_lr_selector.get_support()
    embedded_lr_feature = X.loc[:,embedded_lr_support].columns.tolist()
    return embedded_lr_support, embedded_lr_feature

def embedded_rf_selector(X, y, num_feats):
    """
    Select features using Random Forest Embedded method
    
    Parameters:
    -----------
    X : pd.DataFrame
        Input features
    y : pd.Series
        Target variable
    num_feats : int
        Number of top features to select
    
    Returns:
    --------
    tuple: Boolean support mask and selected feature names
    """
    X_norm = MinMaxScaler().fit_transform(X)
    embedded_rf_selector = SelectFromModel(
        RandomForestClassifier(n_estimators=100, random_state=42),
        max_features=num_feats
    )
    embedded_rf_selector.fit(X_norm, y)
    embedded_rf_support = embedded_rf_selector.get_support()
    embedded_rf_feature = X.loc[:,embedded_rf_support].columns.tolist()
    return embedded_rf_support, embedded_rf_feature

def embedded_lgbm_selector(X, y, num_feats):
    """
    Select features using LightGBM Embedded method
    
    Parameters:
    -----------
    X : pd.DataFrame
        Input features
    y : pd.Series
        Target variable
    num_feats : int
        Number of top features to select
    
    Returns:
    --------
    tuple: Boolean support mask and selected feature names
    """
    X_norm = MinMaxScaler().fit_transform(X)
    lgbc = LGBMClassifier(
        n_estimators=500, 
        learning_rate=0.05, 
        num_leaves=32, 
        random_state=42, 
        verbosity=-1
    )
    embedded_lgbm_selector = SelectFromModel(lgbc, max_features=num_feats)
    embedded_lgbm_selector.fit(X_norm, y)
    embedded_lgbm_support = embedded_lgbm_selector.get_support()
    embedded_lgbm_feature = X.loc[:,embedded_lgbm_support].columns.tolist()
    return embedded_lgbm_support, embedded_lgbm_feature

def preprocess_dataset(dataset_path):
    """
    Preprocess the dataset for feature selection
    
    Parameters:
    -----------
    dataset_path : str
        Path to the input CSV file
    
    Returns:
    --------
    tuple: Preprocessed features (X), target variable (y), and number of features
    """
    player_df = pd.read_csv(dataset_path)
    
    # Define numerical and categorical columns
    numcols = [
        'Overall', 'Crossing', 'Finishing', 'ShortPassing', 'Dribbling', 
        'LongPassing', 'BallControl', 'Acceleration', 'SprintSpeed', 'Agility', 
        'Stamina', 'Volleys', 'FKAccuracy', 'Reactions', 'Balance', 'ShotPower', 
        'Strength', 'LongShots', 'Aggression', 'Interceptions'
    ]
    catcols = ['Preferred Foot', 'Position', 'Body Type', 'Nationality', 'Weak Foot']
    
    # Select only the columns we want
    player_df = player_df[numcols+catcols]
    
    # Create the training dataframe with one-hot encoding for categorical variables
    traindf = pd.concat([player_df[numcols], pd.get_dummies(player_df[catcols])], axis=1)
    traindf = traindf.dropna()
    
    # Create X and y
    y = traindf['Overall'] >= 87
    X = traindf.copy()
    del X['Overall']
    
    # Set number of features to select
    num_feats = 30
    
    return X, y, num_feats

def autoFeatureSelector(dataset_path, methods=[], num_output_features=30):
    """
    Automatically select best features using multiple methods
    
    Parameters:
    -----------
    dataset_path : str
        Path to the input CSV file
    methods : list
        List of feature selection methods to use
    num_output_features : int
        Number of top features to return
    
    Returns:
    --------
    list: Best features selected
    """
    # Preprocessing
    X, y, num_feats = preprocess_dataset(dataset_path)
    feature_name = list(X.columns)
    
    # Dictionary to store support indicators for each method
    support_dict = {}
    feature_dict = {}
    
    # Run every method we outlined above from the methods list
    available_methods = {
        'pearson': cor_selector,
        'chi-square': chi_squared_selector,
        'rfe': rfe_selector,
        'log-reg': embedded_log_reg_selector,
        'rf': embedded_rf_selector,
        'lgbm': embedded_lgbm_selector
    }
    
    for method in methods:
        if method in available_methods:
            selector = available_methods[method]
            support, features = selector(X, y, num_feats)
            support_dict[method] = support
            feature_dict[method] = features
    
    # Create dataframe with all selection methods
    feature_selection_df = pd.DataFrame({'Feature': feature_name})
    for method in methods:
        feature_selection_df[method] = support_dict.get(method, [False]*len(feature_name))
    
    # Count the total votes for each feature
    feature_selection_df['Total'] = feature_selection_df.iloc[:, 1:].sum(axis=1)
    
    # Sort features by total votes and feature name
    feature_selection_df = feature_selection_df.sort_values(['Total', 'Feature'], ascending=False)
    
    # Select features with maximum votes
    max_votes = feature_selection_df['Total'].max()
    best_features = feature_selection_df[feature_selection_df['Total'] == max_votes]['Feature'].tolist()
    
    # Limit to num_output_features if needed
    return best_features[:num_output_features]

def main():
    """
    Main function to run the auto feature selector from command line
    """
    parser = argparse.ArgumentParser(description='Auto Feature Selector')
    parser.add_argument('dataset_path', type=str, help='Path to the dataset CSV file')
    parser.add_argument('--methods', nargs='+', 
                        default=['pearson', 'chi-square', 'rfe', 'log-reg', 'rf', 'lgbm'],
                        help='List of feature selection methods to use')
    parser.add_argument('--num_features', type=int, default=30,
                        help='Number of top features to select')
    
    args = parser.parse_args()
    
    best_features = autoFeatureSelector(
        args.dataset_path, 
        methods=args.methods, 
        num_output_features=args.num_features
    )
    
    print("Best features selected:")
    for feature in best_features:
        print(feature)

if __name__ == "__main__":
    main()