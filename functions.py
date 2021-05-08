#  Importing libraries from Algorithms file
from algorithms import get_class_algorithms, get_reg_algorithms
# Libraries for General Usage
import warnings
warnings.filterwarnings('ignore')
import math
from random import randint, uniform
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn.model_selection import KFold
from sklearn import preprocessing
imputers = [SimpleImputer, KNNImputer]
scalers = [preprocessing.StandardScaler, preprocessing.Normalizer, preprocessing.MinMaxScaler, preprocessing.RobustScaler, preprocessing.QuantileTransformer, preprocessing.PowerTransformer]


def find_best_scale_method(X_train, X_test):
    max_score_X_train = -100
    max_score_X_test = -100
    best_scaler_name = ''
    for scaler in scalers:
        if np.average(scaler().fit_transform(X_train)) > max_score_X_train:
            best_scaler_name = scaler().__class__.__name__
            max_score_X_train = np.average(scaler().fit_transform(X_train))
            X_train = scaler().fit_transform(X_train)
        if np.average(scaler().fit_transform(X_test)) > max_score_X_test:
            max_score_X_test = np.average(scaler().fit_transform(X_test))
            X_test = scaler().fit_transform(X_test)
    return best_scaler_name, X_train, X_test


def get_results(X, y, method, file_name):
    scores_before = []
    scores_after = []
    
    # K-Fold Cross Validation to Split Training and Test Sets
    folds = KFold(n_splits=10, random_state=21, shuffle=True)
    
    # Get all Classification or Regression Algorithms Setups Before and After Data Preprocessing
    algorithms = get_class_algorithms(folds) if method == 'Classification' else get_reg_algorithms(folds, file_name)
    
    
    for name, setup in algorithms.items():
        for train_index, test_index in folds.split(X, y):
            X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
            for status, algorithm in setup.items():
                if status == 'Before':
                    algorithm.fit(X_train, y_train)    
                    scores_before.append(algorithm.score(X_test, y_test))
                else:
                    # Find best scaling method
                    best_scaler_name, X_train, X_test = find_best_scale_method(X_train, X_test)
                    algorithm.fit(X_train, y_train)
                    scores_after.append(algorithm.score(X_test, y_test))
        print('=============================================')
        print(f"{name} {method}")
        print(f"Scaler: {best_scaler_name}")
        print(f"Before Data Preprocessing Score: {math.ceil(np.average(scores_before) * 100)}%")
        print(f"After Data Preprocessing Score: {math.ceil(np.average(scores_after) * 100)}%")
        print('=============================================')
        scores_before.clear()
        scores_after.clear()
        
           
def fill_with_missing_values(df):
    total_rows = len(df.index)
    percentage = round(uniform(0.1, 0.2), 2)
    samples = round(total_rows * percentage)
    print(f"Total Rows: {total_rows}")
    print(f"Percentage: {percentage}")
    print(f"Samples: {samples}")
    print('')
    for i in range(samples):
        row = randint(0, total_rows)
        column = randint(0, len(df.dtypes.index[:-1]) - 1)
        df.loc[row, df.dtypes.index[column]] = np.nan
    return df

def get_imputer_function(name_imputer, imputer):
    if name_imputer == 'SimpleImputer':
        return imputer(missing_values=np.nan, strategy='median')
    return imputer(missing_values=np.nan, metric='nan_euclidean')
    

def init(files, method):
    print(f"Method: {method}")
    for file in files:
        file_name = file.split('/')[1].split('.')[0]
        print(f"Dataset: {file_name}")
        
        # Import data
        df = pd.read_csv(file, error_bad_lines=False)
            
        # Target variable
        target = df.columns[-1]
        
        # List with dummy variables to encode
        dummy_variables = [column for column in df.dtypes.index[:-1] if df.dtypes[column] == object]
        
        # Fill data with missing values
        answer = input('Fill data with missing values ? (Y/N or y/n) : ')
        if answer.lower() == 'y':
            df = fill_with_missing_values(df)

        # Encoding dummy variables
        if dummy_variables: df = pd.get_dummies(df, columns=dummy_variables, drop_first=True)

        # Creating features and target arrays
        X = df.drop(target, axis=1).values
        y = df[target].values

        # Get results after imputing missing data with different imputers
        if df.isnull().values.any():
            for imputer in imputers:
                name_imputer = imputer().__class__.__name__
                print(f"Imputer: {name_imputer}")
                imp = get_imputer_function(name_imputer, imputer)
                X_fitted = imp.fit_transform(X)
                # Get results before data preprocessing
                get_results(X_fitted, y, method, file_name)
                print('')
            return

        # Get results without missing data
        get_results(X, y, method, file_name)
        print('')