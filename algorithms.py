from sklearn.model_selection import RandomizedSearchCV
import numpy as np

# Libraries for Classification Algorithms
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


# Libraries for Regregression Algorithms
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso


def get_class_algorithms(folds):
    svm_model = svm.SVC()
    dtc_model = DecisionTreeClassifier()
    mlp_model = MLPClassifier(max_iter=100)
    knn_model = KNeighborsClassifier()
    rfc_model = RandomForestClassifier()
    
    param_grid_svm = {
        'kernel': ['rbf'],
        'C': [25, 50, 100]
    }
    
    param_grid_dtc = {
        'criterion': ['gini', 'entropy'],
        'splitter': ['best', 'random'],
        'max_depth': range(1, 32),
        'min_samples_leaf': range(1, 10)
    }
    
    param_grid_mlp = {
        'solver': ['lbfgs'],
        'activation': ['tanh', 'relu']
    }
    
    param_grid_knn = {
        'algorithm':['auto', 'ball_tree','kd_tree','brute'],
        'weights':['uniform', 'distance'],
        'n_neighbors': list(range(5, 11)),
        'leaf_size': list(range(1, 6))
    }
    
    param_grid_rfc = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [int(x) for x in np.linspace(1, 45, num = 3)],
    }
    
    return {
        'SVM': {
            'Before': svm_model,
            'After': RandomizedSearchCV(svm_model, param_grid_svm, cv=folds, n_jobs=-1)
        },
        'Decision Tree': {
            'Before': dtc_model,
            'After': RandomizedSearchCV(dtc_model, param_grid_dtc, cv=folds, n_jobs=-1)
        },
        'MLP': {
            'Before': mlp_model,
            'After': RandomizedSearchCV(mlp_model, param_grid_mlp, cv=folds, n_jobs=-1)
        },
        'KNN': {
            'Before': knn_model,
            'After': RandomizedSearchCV(knn_model, param_grid_knn, cv=folds, n_jobs=-1)
        },
        'Random Forest': {
            'Before': rfc_model,
            'After': RandomizedSearchCV(rfc_model, param_grid_rfc, cv=folds, n_jobs=-1)
        },
    }

def get_reg_algorithms(folds, file_name):
    svr_model = SVR()
    linear_model = LinearRegression(fit_intercept=False)
    ridge_model = Ridge(normalize=True)
    lasso_model = Lasso(normalize=True)
    dtm_model = DecisionTreeRegressor()
    rfr_model = RandomForestRegressor()
    mlp_model = MLPRegressor(max_iter=100)
    knn_model = KNeighborsRegressor()
    
    param_grid_svr = {
        'C': [0.1 , 1, 10, 100, 1000],
        'epsilon': [0.0001, 0.001, 0.01, 0.1, 1]
    }
    
    param_grid_linear = {
            'kernel': ['linear'],
            'C': [1e-03, 1e-02, 0.1, 1],
            'gamma': [1e-03, 1e-02, 0.1, 1],
            'epsilon': [1e-02, 0.1, 1],
    }
    
    if file_name == 'computer_hardware':
        param_grid_linear = {
            'kernel': ['rbf'],
            'C': [10000, 100000, 500000]
        }
        
    
    param_grid_ridge_lasso = {
        'alpha': [1e-04, 1e-03, 1e-02, 0.1, 1]
    }
    
    param_grid_dtm = {
        'criterion': ['mse', 'mae'],
        'max_features': ['auto'],
        'max_depth': range(2, 32, 2),
        'random_state': range(0, 10)
    }
    
    param_grid_rfr = {
        'bootstrap': [True],
        'max_features': ['auto'],
        'max_depth': range(70, 120)
    }
    
    param_grid_mlp = {
        'solver': ['lbfgs'],
        'alpha': 10.0 ** -np.arange(1, 10), 
        'hidden_layer_sizes': np.arange(10, 15)
    }
    
    param_grid_knn = {
        'algorithm':['auto', 'ball_tree','kd_tree','brute'],
        'weights':['uniform', 'distance'],
        'n_neighbors': list(range(5, 11)),
        'leaf_size': list(range(1, 6))
    }
    
    return {
        'SVR': {
            'Before': svr_model,
            'After': RandomizedSearchCV(svr_model, param_grid_svr, cv=folds, n_jobs=-1)
        },
        'Linear': {
            'Before': linear_model,
            'After': RandomizedSearchCV(svr_model, param_grid_linear, cv=folds, n_jobs=-1)
        },
        'Ridge': {
            'Before': ridge_model,
            'After': RandomizedSearchCV(ridge_model, param_grid_ridge_lasso, cv=folds, n_jobs=-1)
        },
        'Lasso': {
            'Before': lasso_model,
            'After': RandomizedSearchCV(lasso_model, param_grid_ridge_lasso, cv=folds, n_jobs=-1),
        },
        'Decision Tree': {
            'Before': dtm_model,
            'After': RandomizedSearchCV(dtm_model, param_grid_dtm, cv=folds, n_jobs=-1)
        },
        'Random Forest': {
            'Before': rfr_model,
            'After': RandomizedSearchCV(rfr_model, param_grid_rfr, cv=folds, n_jobs=-1)
        },
        'MLP': {
            'Before': mlp_model,
            'After': RandomizedSearchCV(mlp_model, param_grid_mlp, cv=folds, n_jobs=-1)
        },
        'KNN': {
            'Before': knn_model,
            'After': RandomizedSearchCV(knn_model, param_grid_knn, cv=folds, n_jobs=-1)
        }
    }