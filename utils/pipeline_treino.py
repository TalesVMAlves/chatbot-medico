import time
import pandas as pd
import numpy as np
# Import various classifiers and utilities from scikit-learn and other libraries

# LightGBM classifier, a gradient boosting framework that uses tree-based learning algorithms
from lightgbm import LGBMClassifier

# CalibratedClassifierCV for probability calibration of classifiers
from sklearn.calibration import CalibratedClassifierCV

# Ensemble classifiers from scikit-learn
# ExtraTreesClassifier and RandomForestClassifier are ensemble methods that use multiple decision trees
# StackingClassifier allows combining multiple classifiers to improve performance
from sklearn.ensemble import (ExtraTreesClassifier, RandomForestClassifier, StackingClassifier)

# Linear models from scikit-learn
# LogisticRegression is a linear model for binary classification
# SGDClassifier is a linear classifier using stochastic gradient descent
from sklearn.linear_model import LogisticRegression, SGDClassifier

# Metrics for evaluating classification performance
# accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score, matthews_corrcoef
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score, matthews_corrcoef)

# Naive Bayes classifier for multinomially distributed data
from sklearn.naive_bayes import MultinomialNB

# K-Nearest Neighbors classifier
from sklearn.neighbors import KNeighborsClassifier

# Neural network-based classifier
from sklearn.neural_network import MLPClassifier

# Support Vector Machine classifiers
# SVC is a support vector classifier with a non-linear kernel
# LinearSVC is a support vector classifier with a linear kernel
from sklearn.svm import SVC, LinearSVC

# Decision tree classifier
from sklearn.tree import DecisionTreeClassifier

# XGBoost classifier, an optimized distributed gradient boosting library
from xgboost import XGBClassifier

from typing import List, Tuple

from typing import Dict, Any
from sklearn.metrics import make_scorer, matthews_corrcoef
from skopt import BayesSearchCV


def calculate_evaluation_metrics(y_true: pd.Series, y_pred: pd.Series) -> Tuple[float, float, float, str, float, np.ndarray]:
    """
    Calculate evaluation metrics for model predictions.

    Args:
        y_true (pd.Series): The true labels.
        y_pred (pd.Series): The predicted labels.

    Returns:
        Tuple[float, float, float, str, float, np.ndarray]: The calculated metrics including F1 score, balanced accuracy, accuracy, classification report, Matthews correlation coefficient, and confusion matrix.
    """
    # Calculate F1 score
    f1 = f1_score(y_true, y_pred, average='micro')
    # Calculate balanced accuracy
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)
    # Generate classification report
    classification_report_str = classification_report(y_true, y_pred)
    # Calculate Matthews correlation coefficient
    matthews_corr_coeff = matthews_corrcoef(y_true, y_pred)
    # Generate confusion matrix
    confusion_matrix_arr = confusion_matrix(y_true, y_pred)

    return f1, balanced_accuracy, accuracy, classification_report_str, matthews_corr_coeff, confusion_matrix_arr

def train_and_evaluate_models(X_train: pd.DataFrame,
                              y_train: pd.Series,
                              X_valid: pd.DataFrame, 
                              y_valid: pd.Series,
                              models = [
        ('Calibrated-LSVC', CalibratedClassifierCV(LinearSVC(random_state=1408, class_weight='balanced', dual='auto'))),
        ('LR', LogisticRegression(random_state=1408, n_jobs=-1, class_weight='balanced')),
        ('RF', RandomForestClassifier(random_state=1408, n_jobs=-1, class_weight='balanced')),
        ('LGBM', LGBMClassifier(random_state=1408, n_jobs=-1, class_weight='balanced', verbose=-1)),
        ('XGB', XGBClassifier(random_state=1408, n_jobs=-1, class_weight='balanced', verbosity=0)),
        ('MLP', MLPClassifier(random_state=1408)),
        ('SGD', SGDClassifier(random_state=1408, n_jobs=-1, class_weight='balanced')),
        ('NB', MultinomialNB()),
        ('LSVC', LinearSVC(random_state=1408, class_weight='balanced', dual='auto')),
        ('KNN', KNeighborsClassifier(n_jobs=-1)),
        ('DT', DecisionTreeClassifier(random_state=1408, class_weight='balanced')),
        ('ExtraTrees', ExtraTreesClassifier(random_state=1408, n_jobs=-1, class_weight='balanced'))
    ]) -> Tuple[pd.DataFrame, List[List]]:
    """
    Train multiple models and evaluate their performance.

    Args:
        X_train (pd.DataFrame): The training data.
        y_train (pd.Series): The training labels.
        X_valid (pd.DataFrame): The validation data.
        y_valid (pd.Series): The validation labels.
        n_jobs (int, optional): The number of jobs to run in parallel. Defaults to -1.

    Returns:
        Tuple[pd.DataFrame, List[List]]: A dataframe of the evaluation results and a list of classification reports.
    """
    
    evaluation_results = []
    classification_reports = []
    
    # Train each model and evaluate its performance
    for model_name, model in models:
        start_time = time.time()  # Record the start time

        try:
            # Train the model
            model.fit(X_train, y_train)
            # Make predictions on the validation set
            predictions = model.predict(X_valid)
        except Exception as e:
            # Handle any exceptions that occur during training or prediction
            print(f'Error {model_name} - {e}')
            continue 

        # Calculate evaluation metrics
        f1, balanced_accuracy, accuracy, classification_report_str, matthews_corr_coeff, confusion_matrix_arr = calculate_evaluation_metrics(y_valid, predictions)
        # Store the classification report and confusion matrix
        classification_reports.append([model_name, classification_report_str, confusion_matrix_arr])

        elapsed_time = time.time() - start_time  # Calculate the elapsed time
        # Append the evaluation results
        evaluation_results.append([model_name, f1, balanced_accuracy, accuracy, matthews_corr_coeff, elapsed_time, confusion_matrix_arr, classification_report_str])

        # Print the evaluation results
        print(f'Name: {model_name} - F1: {f1:.4f} - BACC: {balanced_accuracy:.4f} - ACC: {accuracy:.4f} - MCC: {matthews_corr_coeff:.4f} - Elapsed: {elapsed_time:.2f}s')
        print(classification_report_str)
        print(confusion_matrix_arr)
        print('*' * 20, '\n')

    # Create a DataFrame to store the evaluation results
    results_df = pd.DataFrame(evaluation_results, columns=['Model', 'F1', 'BACC', 'ACC', 'MCC', 'Total Time', 'Confusion Matrix', 'Classification Report'])
    # Convert the confusion matrix to a string for better readability in the DataFrame
    results_df['Confusion Matrix'] = results_df['Confusion Matrix'].apply(lambda x: str(x))

    return results_df, classification_reports


def perform_bayesian_optimization(model: LinearSVC, param_space: Dict[str, Any], X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> None:
    """
    Perform Bayesian optimization to find the best parameters for the model and evaluate it on the validation set.

    Args:
        model (LinearSVC): The model to be trained.
        param_space (Dict[str, Any]): The parameter space for the search.
        X_train (pd.DataFrame): The training data.
        y_train (pd.Series): The training labels.
        X_val (pd.DataFrame): The validation data.
        y_val (pd.Series): The validation labels.
    """
    # Define the scorer for the Bayesian optimization
    mcc_scorer = make_scorer(matthews_corrcoef)

    # Create the BayesSearchCV object with the LinearSVC and parameter space
    bayesian_search = BayesSearchCV(model, param_space, cv=3, scoring=mcc_scorer, 
                                    n_jobs=-1, n_iter=30, random_state=1408, n_points=10)

    # Perform the Bayesian optimization by fitting training data
    bayesian_search.fit(X_train, y_train)

    # Print the best parameters and corresponding MCC score found by the Bayesian search
    print("Best parameters: ", bayesian_search.best_params_)
    print("Best score: ", bayesian_search.best_score_)

    # Evaluate the model with chosen parameters on the validation set
    best_model = bayesian_search.best_estimator_
    y_val_pred = best_model.predict(X_val)

    metrics = {
        'MCC': matthews_corrcoef(y_val, y_val_pred),
        'F1 Score': f1_score(y_val, y_val_pred, average='weighted'),
        'Balanced Accuracy': balanced_accuracy_score(y_val, y_val_pred),
        'Accuracy': accuracy_score(y_val, y_val_pred),
    }

    print("\nValidation Metrics:")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")
    
    return best_model