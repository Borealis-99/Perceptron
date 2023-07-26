"""
Simple Classifiers for Cancer-Risk Screening

Problem Statement:
You have been given a data set containing some medical history information for
patients at risk for cancer [1]. This data has been split into various training and
testing sets; each set is given in CSV form, and is divided into inputs (x) and
outputs (y).

Each patient in the data set has been biopsied to determine their actual cancer
status. This is represented as a boolean variable, cancer in the y data sets,
where 1 means the patient has cancer and 0 means they do not. You will build
classifiers that seek to predict whether a patient has cancer, based on other
features of that patient. (The idea is that if we could avoid painful biopsies,
this would be preferred.)

[1] A. Vickers, Memorial Sloan Kettering Cancer Center
https://www.mskcc.org/sites/default/files/node/4509/documents/dca-tutorial-2015-2-26.pdf

================================================================================
See PDF for details of implementation and expected responses.
"""
import os
import numpy as np
import pandas as pd

import warnings

import sklearn.linear_model
import sklearn.metrics
import sklearn.calibration
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, confusion_matrix
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.calibration import CalibratedClassifierCV

plt.style.use('seaborn-v0_8')  # pretty matplotlib plots


def calc_confusion_matrix_for_threshold(y_true_N, y_proba1_N, thresh=0.5):
    ''' Compute the confusion matrix for a given probabilistic classifier and threshold

    Args
    ----
    y_true_N : 1D array of floats
        Each entry represents the binary value (0 or 1) of 'true' label of one example
        One entry per example in current dataset
    y_proba1_N : 1D array of floats
        Each entry represents a probability (between 0 and 1) that correct label is positive (1)
        One entry per example in current dataset
        Needs to be same size as ytrue_N
    thresh : float
        Scalar threshold for converting probabilities into hard decisions
        Calls an example "positive" if yproba1 >= thresh
        Default value reflects a majority-classification approach (class is the one that gets
        the highest probability)

    Returns
    -------
    cm_df : Pandas DataFrame
        Can be printed like print(cm_df) to easily display results
    '''
    from sklearn.metrics import confusion_matrix 
    cm = confusion_matrix(y_true_N, y_proba1_N >= thresh)
    cm_df = pd.DataFrame(data=cm, columns=[0, 1], index=[0, 1])
    cm_df.columns.name = 'Predicted'
    cm_df.index.name = 'True'
    return cm_df


def calc_binary_metrics(y_true_N, y_hat_N):
    """
    Calculate confusion metrics.
    Args
    ----
    y_true_N : 1D array of floats
        Each entry represents the binary value (0 or 1) of 'true' label of one example
        One entry per example in current dataset
    y_hat_N : 1D array of floats
        Each entry represents a predicted binary value (either 0 or 1).
        One entry per example in current dataset.
        Needs to be same size as y_true_N.

    Returns
    -------
    TP : int
        Number of true positives
    TN : int
        Number of true negatives
    FP : int
        Number of false positives
    FN : int
        Number of false negatives
    """
    # TODO
    TP = 0.0
    TN = 0.0
    FP = 0.0
    FN = 0.0

    for i in range(len(y_true_N)):
        if y_true_N[i] == 1 and y_hat_N[i] == 1:
            TP += 1
        elif y_true_N[i] == 0 and y_hat_N[i] == 0:
            TN += 1
        elif y_true_N[i] == 0 and y_hat_N[i] == 1:
            FP += 1
        elif y_true_N[i] == 1 and y_hat_N[i] == 0:
            FN += 1

    # TP = ((y_true_N == 1) & (y_hat_N == 1)).sum()
    # TN = ((y_true_N == 0) & (y_hat_N == 0)).sum()
    # FP = ((y_true_N == 0) & (y_hat_N == 1)).sum()
    # FN = ((y_true_N == 1) & (y_hat_N == 0)).sum()

    return TP, TN, FP, FN


def calc_percent_cancer(labels, cancer_label=1):
    """
    Calculate the number of instances that are equal to `cancer_label`
    :param labels: target variables of training of testing set
    :param cancer_label: the value indicating cancer positive
    :return: Percentage of labels marked as cancerous.
    """
    # TODO
    cancer_count = np.sum(labels == cancer_label)
    return 100 * cancer_count / len(labels)


def predict_0_always_classifier(X):
    """
    Implement a classifier that predicts 0 always.
    :param X: Samples to classify
    :return: predictions from the always-0 classifier
    """
    # TODO
    return np.zeros(X.shape[0])


def calc_accuracy(tp, tn, fp, fn):
    """
    Calculate the accuracy via confusion metrics.
    :param tp: Number of true positives
    :param tn: Number of true negative
    :param fp: Number of false negative
    :param fn: Number of false negative
    :return: Accuracy value from 0.0 to 1.0
    """
    # TODO
    return (tp + tn) / (tp + tn + fp + fn)


def standardize_data(X_train, X_test):
    """
    Transform features by scaling each feature to a given range.

    This estimator scales and translates each feature individually such that it
    is in the given range on the training set, e.g. between zero and one.

    :param X_train: training features
    :param X_test: testing features
    :return: standardize training features and testing features
    """
    from sklearn.preprocessing import MinMaxScaler
    # TODO
    # create scaler object
    scaler = MinMaxScaler()

    # fit scaler to training data and transform training data
    X_train_std = scaler.fit_transform(X_train)

    # transform test data using the same scaler
    X_test_std = scaler.transform(X_test)

    return X_train_std, X_test_std


def calc_perf_metrics_for_threshold(y_true_N, y_proba1_N, thresh=0.5):
    """
    Compute performance metrics for a given probabilistic classifier and threshold
    Args
    ----
    y_true_N : 1D array of floats
        Each entry represents the binary value (0 or 1) of 'true' label of one example
        One entry per example in current dataset
    y_proba1_N : 1D array of floats
        Each entry represents a probability (between 0 and 1) that correct label is positive (1)
        One entry per example in current dataset
        Needs to be same size as y_true_N
    thresh : float
        Scalar threshold for converting probabilities into hard decisions
        Calls an example "positive" if y_proba1 >= thresh
        Default value reflects a majority-classification approach (class is the one that gets
        the highest probability)

    Returns
    -------
    acc : accuracy of predictions
    tpr : true positive rate of predictions
    tnr : true negative rate of predictions
    ppv : positive predictive value of predictions
    npv : negative predictive value of predictions
    """
    # TODO
    # tp, tn, fp, fn = calc_binary_metrics(...)
    y_pred_N = (y_proba1_N >= thresh).astype(int)
    tp, tn, fp, fn = calc_binary_metrics(y_true_N, y_pred_N)

    # acc = 0.0
    # tpr = 0.0
    # tnr = 0.0
    # ppv = 0.0
    # npv = 0.0
    # TODO
    acc = calc_accuracy(tp, tn, fp, fn)
    if tp + fn == 0.0:
        tpr = 0.0
    else:
        tpr = tp / (tp + fn) # sensitivity/recall
    if tn + fp == 0.0:
        tnr = 0.0
    else:
        tnr = tn / (tn + fp) # specificity
    if tp + fp == 0.0:
        ppv = 0.0
    else:
        ppv = tp / (tp + fp) # precision
    if tn + fn == 0.0:
        npv = 0.0
    else:
        npv = tn / (tn + fn) # negative predictive value

    return acc, tpr, tnr, ppv, npv


def perceptron_classifier(x_train, y_train, x_test, y_test, penalty="l2",
                          alpha=0, random_state=42):
    """
    Trains a perceptron classifier on the given training data and returns the
    predicted values on both training and test data.
    
    Args
    ----
    x_train : 1D array of floats
    y_train : 1D array of floats
    x_test : 1D array of floats
    y_test : 1D array of floats
    random_state : Seed used by the random number generator
    
    Returns
    -------
    tuple: A tuple of predicted values for the training and test data.
    """
    # Basic Perceptron Models
    # Fit a perceptron to the training data.
    # Print out accuracy on this data, as well as on testing data.
    # Print out a confusion matrix on the testing data.

    pred_train = 0.0
    pred_test = 0.0

    # TODO: Use penalty, alpha, random_state for your perceptron classifier
    # BE SURE TO SET RANDOM SEED FOR CLASSIFIER TO BE DETERMINISTIC TO PASS TEST
    
    # Fit a perceptron to the training data.
    clf = Perceptron(penalty=penalty, alpha=alpha, random_state=random_state)
    clf.fit(x_train, y_train)

    # Predict on training and test data
    pred_train = clf.predict(x_train)
    pred_test = clf.predict(x_test)

    # # Print out accuracy on this data, as well as on testing data.
    # print(f"Accuracy on training data: {accuracy_score(y_train, pred_train):.3f}")
    # print(f"Accuracy on test data: {accuracy_score(y_test, pred_test):.3f}")

    # # Print out a confusion matrix on the testing data.
    # print("Confusion matrix on test data:")
    # print(confusion_matrix(y_test, pred_test))

    return pred_train, pred_test


def series_of_preceptrons(x_train, y_train, x_test, y_test, random_state=42, alphas=np.logspace(-5, 5, base=10, num=100)):
    """
    Train a series of Perceptron models with different regularization strengths and evaluate their performance on the given data.

    The function trains a sequence of Perceptron models with regularization strengths specified by the `alphas` parameter. For each model, the accuracy is computed on both the training and test sets.

    Parameters
    ----------

    x_train : array-like of shape (n_samples, n_features)
        Training input samples.
    y_train : array-like of shape (n_samples,)
        Target values for the training set.
    x_test : array-like of shape (n_samples, n_features)
        Test input samples.
    y_test : array-like of shape (n_samples,)
        Target values for the test set.
    random_state : int, optional (default=42)
        Random seed used for initializing the Perceptron models.
    alphas : array-like of shape (n_alphas,), optional (default=np.logspace(-5, 5, base=10, num=100))
        List of regularization strengths to be used for the Perceptron models.

    Returns
    -------
    train_accuracy_list : list of floats
        List of training accuracies for each of the Perceptron models.
    test_accuracy_list : list of floats
        List of test accuracies for each of the Perceptron models.
    """

    """Generate a series of regularized perceptron models

    Each model will use a different `alpha` value, multiplying that by the L2 penalty. 
    """
    # TODO

    # Initialize variables
    train_accuracy_list = list()
    test_accuracy_list = list()

    # Train a sequence of Perceptron models with different regularization strengths
    for alpha in alphas:
        # create the classifier with the current alpha value
        clf = Perceptron(penalty='l2', alpha=alpha, random_state=random_state)
        # fit the classifier on the training data
        clf.fit(x_train, y_train)
        # calculate accuracy on the training data and append to the train_accuracy_list
        train_accuracy = clf.score(x_train, y_train)
        train_accuracy_list.append(train_accuracy)
        # calculate accuracy on the test data and append to the test_accuracy_list
        test_accuracy = clf.score(x_test, y_test)
        test_accuracy_list.append(test_accuracy)

    # Return the training and test accuracies for each of the Perceptron models
    return train_accuracy_list, test_accuracy_list


def calibrated_perceptron_classifier(x_train, y_train, x_test, y_test,
                                     penalty="l2", alpha=0, random_state=42):
    """
    Train and evaluate a calibrated Perceptron classifier on the given data.

    The classifier is first trained using a standard Perceptron model, and the decision function is computed on the test set. Then a calibrated classifier is trained using the same Perceptron model and the isotonic calibration method. Finally, the calibrated classifier is used to compute the predicted probabilities for the test set.

    Parameters
    ----------
    x_train : array-like of shape (n_samples, n_features)
        Training input samples.
    y_train : array-like of shape (n_samples,)
        Target values for the training set.
    x_test : array-like of shape (n_samples, n_features)
        Test input samples.
    y_test : array-like of shape (n_samples,)
        Target values for the test set.
    penalty : str, optional (default='l2')
        Penalty to be used for the Perceptron model.
    alpha : float, optional (default=0)
        Constant that multiplies the regularization term.
    random_state : int, optional (default=42)
        Random seed used for initializing the Perceptron model.

    Returns
    -------
    decisions_test : ndarray of shape (n_samples_test,)
        Decision function output for the test samples.
    pred_prob_test : ndarray of shape (n_samples_test,)
        Predicted probabilities for the test samples.

    """
    # Use penalty, alpha, random_state for your perceptron classifier
    clf = Perceptron(penalty=penalty, alpha=alpha, random_state=random_state)

    # Train a Perceptron classifier
    clf.fit(x_train, y_train)

    # Compute the decision function on the test set
    decisions_test = clf.decision_function(x_test)

    # Train a calibrated classifier using the isotonic calibration method
    clf_calibrated = CalibratedClassifierCV(clf, cv=5, method='isotonic')
    clf_calibrated.fit(x_train, y_train)

    # Compute the predicted probabilities for the test set
    pred_prob_test = clf_calibrated.predict_proba(x_test)[:, 1]

    # Return the decision function output and the predicted probabilities
    return decisions_test, pred_prob_test

    pred_train = 0.0
    pred_test = 0.0

    # TODO: Use penalty, alpha, random_state for your perceptron classifier
    # BE SURE TO SET RANDOM SEED FOR CLASSIFIER TO BE DETERMINISTIC TO PASS TEST

    # Train a basic perceptron on the training data
    perceptron = Perceptron(penalty=penalty, alpha=alpha, random_state=random_state)
    perceptron.fit(x_train, y_train)

    # Calibrate the trained perceptron using CalibratedClassifierCV
    calibrated_perceptron = CalibratedClassifierCV(perceptron, cv=5, method='isotonic')
    calibrated_perceptron.fit(x_train, y_train)

    # Obtain the calibrated predictions on both the training and test data
    pred_train = calibrated_perceptron.predict_proba(x_train)[:, 1]
    pred_test = calibrated_perceptron.predict_proba(x_test)[:, 1]

    return pred_train, pred_test


def find_best_thresholds(y_test, pred_prob_test):
    """ Compare the probabilistic classifier across multiple decision thresholds
    Args
    ----
    y_test : 1D array of floats
    pred_prob_test : 1D array of floats

    Returns
    -------
    best_TPR : best true positive rate
    best_PPV_for_best_TPR : best positive predictive value for best true positive rate
    best_TPR_threshold : best true positive rate threshold
    best_PPV : best positive predictive value
    best_TPR_for_best_PPV : best true positive rate for best positive predictive value
    best_PPV_threshold : best positive predictive value threshold
    """
    # Try a range of thresholds for classifying data into the positive class (1).
    # For each threshold, compute the true postive rate (TPR) and positive
    # predictive value (PPV).  Record the best value of each metric, along with
    # the threshold that achieves it, and the *other*

    best_TPR = 0.0
    best_PPV_for_best_TPR = 0.0
    best_TPR_threshold = 0.0

    best_PPV = 0.0
    best_TPR_for_best_PPV = 0.0
    best_PPV_threshold = 0.0

    # TODO: test different thresholds to compute these values
    # thresholds = np.linspace(0, 1.001, 51)

    # Try different thresholds to compute these values
    thresholds = np.linspace(0, 1.001, 51)

    for threshold in thresholds:
        # # Predict labels for given threshold
        # pred_labels_test = (pred_prob_test >= threshold).astype(int)
        
        # # Calculate confusion matrix for predicted labels
        # tn, fp, fn, tp = confusion_matrix(y_test, pred_labels_test).ravel()
        
        # # Calculate true positive rate (TPR)
        # curr_TPR = tp / (tp + fn)
        
        # # Calculate positive predictive value (PPV)
        # curr_PPV = tp / (tp + fp)

        acc, curr_TPR, tnr, curr_PPV, npv = calc_perf_metrics_for_threshold(y_test, pred_prob_test, threshold)
        
        # # Update variables for best TPR threshold
        if curr_TPR > best_TPR:
            best_TPR = curr_TPR
            best_PPV_for_best_TPR = curr_PPV
            best_TPR_threshold = threshold
        elif curr_TPR == best_TPR and curr_PPV > best_PPV_for_best_TPR:
            best_PPV_for_best_TPR = curr_PPV
            best_TPR_threshold = threshold
        
        # Update variables for best PPV threshold
        if curr_PPV > best_PPV:
            best_PPV = curr_PPV
            best_TPR_for_best_PPV = curr_TPR
            best_PPV_threshold = threshold
        elif curr_PPV == best_PPV and curr_TPR > best_TPR_for_best_PPV:
            best_TPR_for_best_PPV = curr_TPR
            best_PPV_threshold = threshold

    return best_TPR, best_PPV_for_best_TPR, best_TPR_threshold, best_PPV, best_TPR_for_best_PPV, best_PPV_threshold


# You can use this function later to make printing results easier; don't change it.
def print_perf_metrics_for_threshold(y_true, y_proba1, thresh=0.5):
    """
    Pretty print perf. metrics for a given probabilistic classifier and threshold.

    See calc_perf_metrics_for_threshold() for parameter descriptions.
    """
    acc, tpr, tnr, ppv, npv = calc_perf_metrics_for_threshold(y_true,
                                                              y_proba1, thresh)

    # Pretty print the results
    print("%.3f ACC" % acc)
    print("%.3f TPR" % tpr)
    print("%.3f TNR" % tnr)
    print("%.3f PPV" % ppv)
    print("%.3f NPV" % npv)


if __name__ == '__main__':
    dir_data = "./data"
    all0 = np.zeros(10)
    all1 = np.ones(10)
    # Testing code
    # The following four calls to the function above test your results.
    TP, TN, FP, FN = calc_binary_metrics(all0, all1)
    print(f"TP: {TP}\t\tTN: {TN}\t\tFP: {FP}\tFN: {FN}\t")
    calc_binary_metrics(all1, all0)
    print(f"TP: {TP}\t\tTN: {TN}\t\tFP: {FP}\tFN: {FN}\t")
    calc_binary_metrics(all1, all1)
    print(f"TP: {TP}\t\tTN: {TN}\t\tFP: {FP}\tFN: {FN}\t")
    calc_binary_metrics(all0, all0)
    print(f"TP: {TP}\t\tTN: {TN}\t\tFP: {FP}\tFN: {FN}\t\n")

    # Load the dataset: creates arrays with the 2- or 3-feature input datasets.
    # Load the x-data and y-class arrays
    x_train = np.loadtxt(f'{dir_data}/x_train.csv',
                         delimiter=',',
                         skiprows=1)
    x_test = np.loadtxt(f'{dir_data}/x_test.csv',
                        delimiter=',',
                        skiprows=1)

    y_train = np.loadtxt(f'{dir_data}/y_train.csv',
                         delimiter=',',
                         skiprows=1)
    y_test = np.loadtxt(f'{dir_data}/y_test.csv',
                        delimiter=',',
                        skiprows=1)

    feat_names = np.loadtxt(f'{dir_data}/x_train.csv',
                            delimiter=',',
                            dtype=str,
                            max_rows=1)
    target_name = np.loadtxt(f'{dir_data}/y_train.csv',
                             delimiter=',',
                             dtype=str,
                             max_rows=1)

    print(f"features: {feat_names}\n")

    df_sampled_data = pd.DataFrame(x_test, columns=feat_names)
    df_sampled_data[str(target_name)] = y_test
    print(df_sampled_data.sample(15))
    print()

    # 2: Compute the fraction of patients with cancer.
    # Compute values for train and test sets (i.e., don't hand-count and print).
    # percent_cancer_train = calc_percent_cancer(y_train)
    # percent_cancer_test = calc_percent_cancer(y_test)
    # print("Percent of data that has_cancer on TRAIN: %.2f" %
    #       percent_cancer_train if percent_cancer_train else 0.0)
    # print("Percent of data that has_cancer on TEST : %.2f\n" %
    #       percent_cancer_test if percent_cancer_test else 0.0)

    # The predict-0-always baseline
    # Compute and print the accuracy of the always-0 classifier on val and test.
    # y_train_pred = predict_0_always_classifier(x_train)
    # y_test_pred = predict_0_always_classifier(x_test)
    # acc_always0_train = calc_accuracy(
    #     *calc_binary_metrics(y_train, y_train_pred)
    # )
    # acc_always0_test = calc_accuracy(*calc_binary_metrics(y_test, y_test_pred))
    # print("acc on TRAIN: %.3f" %
    #       acc_always0_train if acc_always0_train else 0.0)
    # print("acc on TEST : %.3f\n" %
    #       acc_always0_test if acc_always0_test else 0.0)

    # Print a confusion matrix for the always-0 classifier.
    # Generate a confusion matrix for the always-0 classifier on the test set.
    # print(calc_confusion_matrix_for_threshold(y_test, np.zeros_like(y_test)))

    # Basic Perceptron Models
    # Fit a perceptron to the training data.
    # Print out accuracy on this data, as well as on testing data.
    # Print out a confusion matrix on the testing data.
    # pred_train, pred_test = perceptron_classifier()
    # print("acc on TRAIN: %.3f" % accuracy_score(y_train, pred_train))
    # print("acc on TEST : %.3f" % accuracy_score(y_test, pred_test))
    # print("")
    # print("Confusion matrix for TEST:")
    # print(calc_confusion_matrix_for_threshold(y_test, pred_test))

    # Generate a series of regularized perceptron models
    # Each model will use a different `alpha` value multiplied by the L2 penalty.
    # Record and plot the accuracy of each model on both training and test data.

    # train_accuracy_list, test_accuracy_list = series_of_preceptrons(
    # alphas=np.logspace(-5, 5, base=10, num=100))
    # plt.plot(alphas, train_accuracy_list, label='Accuracy on training')
    # plt.plot(alphas, test_accuracy_list, label='Accuracy on testing')
    # plt.legend(loc='lower right')
    # plt.xscale('log')
    # plt.xlabel('log_10(alpha)')
    # plt.ylabel('Accuracy')
    # plt.show()

    # Decision functions and probabilistic predictions
    #
    # Create two new sets of predictions
    #
    # Fit `Perceptron` and `CalibratedClassifierCV` models to the data.
    # Use their predictions to generate ROC curves.

    # _, decisions_test = perceptron_classifier()
    # _, pred_prob_test = calibrated_perceptron_classifier()

    # fpr, tpr, thr = sklearn.metrics.roc_curve(y_test, decisions_test)
    # plt.plot(fpr, tpr, label='Decision function version')

    # fpr2, tpr2, thr2 = sklearn.metrics.roc_curve(y_test, pred_prob_test)
    # plt.plot(fpr2, tpr2, label='Probabilistic version')

    # plt.legend(loc='lower right')
    # plt.xlabel('False positive rate (FPR)')
    # plt.ylabel('True positive rate (TPR)')
    #
    # plt.show()
    #
    # print("AUC on TEST for Perceptron: %.3f" % sklearn.metrics.roc_auc_score(
    #     y_test, decisions_test))
    # print(
    #     "AUC on TEST for probabilistic model: %.3f" % sklearn.metrics.roc_auc_score(
    #         y_test, pred_prob_test))
    #
    # # Compare the probabilistic classifier across multiple decision thresholds
    # #
    # # Try a range of thresholds for classifying data into the positive class (1).
    # # For each threshold, compute the true postive rate (TPR) and positive
    # # predictive value (PPV).  Record the best value of each metric, along with
    # # the threshold that achieves it, and the *other*
    # # TODO: test different thresholds to compute these values
    # best_TPR, best_PPV_for_best_TPR, \
    #     best_TPR_threshold, best_PPV, \
    #     best_TPR_for_best_PPV, \
    #     best_PPV_threshold = find_best_thresholds(y_test, pred_prob_test)
    #
    # print("TPR threshold: %.4f => TPR: %.4f; PPV: %.4f" % (
    #     best_TPR_threshold, best_TPR, best_PPV_for_best_TPR))
    # print("PPV threshold: %.4f => PPV: %.4f; TPR: %.4f" % (
    #     best_PPV_threshold, best_PPV, best_TPR_for_best_PPV))
    #
    # # #### (e) Exploring diffrerent thresholds
    # #
    # # #### (i) Using default 0.5 threshold.
    # #
    # # Generate confusion matrix and metrics for probabilistic classifier, using threshold 0.5.
    # best_thr = 0.5
    # print("ON THE TEST SET:")
    # print("Chosen best threshold = %.4f" % best_thr)
    # print("")
    # print(calc_confusion_matrix_for_threshold(y_test, pred_prob_test, best_thr))
    # print("")
    # print_perf_metrics_for_threshold(y_test, pred_prob_test, best_thr)
    #
    # #  Using threshold with highest TPR.
    # #
    # # Generate confusion matrix and metrics for probabilistic classifier, using
    # # threshold that maximizes TPR.
    # best_thr = best_TPR_threshold
    # print("ON THE TEST SET:")
    # print("Chosen best threshold = %.4f" % best_thr)
    # print("")
    # print(calc_confusion_matrix_for_threshold(y_test, pred_prob_test, best_thr))
    # print("")
    # print_perf_metrics_for_threshold(y_test, pred_prob_test, best_thr)
    #
    # # #### (iii) Using threshold with highest PPV.
    # #
    # # Generate confusion matrix and metrics for probabilistic classifier, using threshold that maximizes PPV.
    # best_thr = best_PPV_threshold
    # print("ON THE TEST SET:")
    # print("Chosen best threshold = %.4f" % best_thr)
    # print("")
    # print(calc_confusion_matrix_for_threshold(y_test, pred_prob_test, best_thr))
    # print("")
    # print_perf_metrics_for_threshold(y_test, pred_prob_test, best_thr)
