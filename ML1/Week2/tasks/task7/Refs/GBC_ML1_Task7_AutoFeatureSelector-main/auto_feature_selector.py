import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, RFE, SelectFromModel 
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
import argparse

def cor_selector(X, y, num_feats):
    cor_list = []
    feature_name = X.columns.tolist()
    
    # calculate correlation between features and target
    for i in feature_name:
        cor = np.corrcoef(X[i], y)[0, 1]
        cor_list.append(cor)
        
    # convert to dataframe
    feature_value = pd.DataFrame(
        {'Feature': feature_name,
         'Correlation': cor_list})
    
    # sort by absolute correlation value
    feature_value['Correlation'] = feature_value['Correlation'].abs()
    feature_value = feature_value.sort_values('Correlation', ascending=False)
    
    # select top features
    topk_feature = feature_value.iloc[:num_feats, :]
    
    # create boolean mask
    cor_support = []
    for feat in feature_name:
        if feat in topk_feature['Feature'].tolist():
            cor_support.append(True)
        else:
            cor_support.append(False)
            
    # get selected feature names
    cor_feature = X.columns[cor_support].tolist()
    
    return cor_support, cor_feature

def chi_squared_selector(X, y, num_feats):
    X_norm = MinMaxScaler().fit_transform(X)
    chi_selector = SelectKBest(chi2, k=num_feats)
    chi_selector.fit(X_norm, y)
    chi_support = chi_selector.get_support()
    chi_feature = X.loc[:,chi_support].columns.tolist()
    return chi_support, chi_feature

def rfe_selector(X, y, num_feats):
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
    X_norm = MinMaxScaler().fit_transform(X)
    lgbc = LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=32, random_state=42, verbosity=-1) # Verbosity = -1 to suppress warnings
    embedded_lgbm_selector = SelectFromModel(lgbc, max_features=num_feats)
    embedded_lgbm_selector.fit(X_norm, y)
    embedded_lgbm_support = embedded_lgbm_selector.get_support()
    embedded_lgbm_feature = X.loc[:,embedded_lgbm_support].columns.tolist()
    return embedded_lgbm_support, embedded_lgbm_feature

def preprocess_dataset(dataset_path):
    player_df = pd.read_csv(dataset_path)
    
    # Define numerical and categorical columns
    numcols = ['Overall', 'Crossing','Finishing', 'ShortPassing', 'Dribbling','LongPassing', 
               'BallControl', 'Acceleration','SprintSpeed', 'Agility', 'Stamina','Volleys',
               'FKAccuracy','Reactions','Balance','ShotPower','Strength','LongShots',
               'Aggression','Interceptions']
    catcols = ['Preferred Foot','Position','Body Type','Nationality','Weak Foot']
    
    # Select only the columns we want
    player_df = player_df[numcols+catcols]
    
    # Create the training dataframe with one-hot encoding for categorical variables
    traindf = pd.concat([player_df[numcols], pd.get_dummies(player_df[catcols])], axis=1)
    traindf = traindf.dropna()
    
    # Create X and y
    y = traindf['Overall']>=87
    X = traindf.copy()
    del X['Overall']
    
    # Set number of features to select
    num_feats = 30
    
    return X, y, num_feats

def autoFeatureSelector(dataset_path, methods=[]):
    # Parameters
    # data - dataset to be analyzed (csv file)
    # methods - various feature selection methods we outlined before, use them all here (list)
    
    # preprocessing
    X, y, num_feats = preprocess_dataset(dataset_path)
    feature_name = list(X.columns)
    
    # Dictionary to store support indicators for each method
    support_dict = {}
    feature_dict = {}
    
    # Run every method we outlined above from the methods list and collect returned best features from every method
    if 'pearson' in methods:
        cor_support, cor_feature = cor_selector(X, y,num_feats)
        support_dict['pearson'] = cor_support
        feature_dict['pearson'] = cor_feature
    if 'chi-square' in methods:
        chi_support, chi_feature = chi_squared_selector(X, y,num_feats)
        support_dict['chi-square'] = chi_support
        feature_dict['chi-square'] = chi_feature
    if 'rfe' in methods:
        rfe_support, rfe_feature = rfe_selector(X, y,num_feats)
        support_dict['rfe'] = rfe_support
        feature_dict['rfe'] = rfe_feature
    if 'log-reg' in methods:
        embedded_lr_support, embedded_lr_feature = embedded_log_reg_selector(X, y, num_feats)
        support_dict['log-reg'] = embedded_lr_support
        feature_dict['log-reg'] = embedded_lr_feature
    if 'rf' in methods:
        embedded_rf_support, embedded_rf_feature = embedded_rf_selector(X, y, num_feats)
        support_dict['rf'] = embedded_rf_support
        feature_dict['rf'] = embedded_rf_feature
    if 'lgbm' in methods:
        embedded_lgbm_support, embedded_lgbm_feature = embedded_lgbm_selector(X, y, num_feats)
        support_dict['lgbm'] = embedded_lgbm_support
        feature_dict['lgbm'] = embedded_lgbm_feature
    
    # Create dataframe with all selection methods
    feature_selection_df = pd.DataFrame({'Feature':feature_name})
    for method in methods:
        feature_selection_df[method] = support_dict[method]
    
    # Count the total votes for each feature
    feature_selection_df['Total'] = feature_selection_df.iloc[:, 1:].sum(axis=1)
    
    # Sort features by total votes and feature name
    feature_selection_df = feature_selection_df.sort_values(['Total','Feature'], ascending=False)
    
    # Select features with maximum votes
    max_votes = feature_selection_df['Total'].max()
    best_features = feature_selection_df[feature_selection_df['Total'] == max_votes]['Feature'].tolist()
    
    return best_features

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Auto Feature Selector')
    parser.add_argument('dataset_path', type=str, help='Path to the dataset CSV file')
    parser.add_argument('--methods', nargs='+', default=['pearson', 'chi-square', 'rfe', 'log-reg', 'rf', 'lgbm'],
                        help='List of feature selection methods to use')
    
    args = parser.parse_args()
    
    best_features = autoFeatureSelector(args.dataset_path, args.methods)
    print("Best features selected:")
    for feature in best_features:
        print(feature)
