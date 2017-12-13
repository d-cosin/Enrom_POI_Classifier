#!/usr/bin/python

import os
import sys
import pickle
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
from scipy.stats import randint as sp_randint
from scipy.stats import expon
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, \
    StratifiedShuffleSplit
from sklearn import tree
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier

sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from tester  import dump_classifier_and_data, test_classifier
from aux_functions import print_log_message, extract_df_features, \
    test_k_values, tune_hyper_parameters
print_log_message("Libraries Initialized.")

PLOTS_XTICKS = np.arange(0, 23, 1)
PLOTS_YTICKS = np.arange(0, 0.61, 0.05)

i = 1  # figure counter

# Load the file into Pandas Dataframe.
print_log_message("Loading file and features...")
train = pd.read_pickle("final_project_dataset.pkl")
df_train = pd.DataFrame.from_dict(train, orient="index")
df_train = df_train.replace("NaN", 0)
df_train = df_train.drop("email_address", axis=1)  # E-mail acts like an Id

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = extract_df_features(df_train)  # Initially using all features
print_log_message("Done.")

### Task 2: Remove outliers
df_train = df_train.drop("TOTAL")  # "TOTAL" just sums all observations

### Task 3: Create new feature(s)
df_train["ratio_receipt_messages"] = df_train["from_poi_to_this_person"]/df_train["to_messages"]
df_train["ratio_sent_messages"] = df_train["from_this_person_to_poi"]/df_train["from_messages"]
df_train["ratio_shared_receipt"] = df_train["shared_receipt_with_poi"]/df_train["to_messages"]
df_train = df_train.replace(np.nan, 0)  # Division by zero NaNs
features_list = extract_df_features(df_train) # refresh features_list

### Store to my_dataset for easy export below.
my_dataset = df_train.to_dict(orient="index")

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

# K Score of the New Features
k_values = SelectKBest(k="all").fit(features, labels).scores_
k_features = features_list
k_features.pop(features_list.index("poi"))
# Plot the Scores
ind = range(1,len(k_features)+1)
plt.figure(i)
plt.grid(True, zorder=0)
plt.bar(ind, k_values, zorder=3)
plt.title("Features' k Scores")
plt.xlabel("Feature")
plt.xticks(ind, tuple(k_features), rotation="vertical")
plt.ylabel("k Score")
i += 1

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Decision Tree Classifier
print_log_message("Training and validating baseline model...")
estimators = [
    ("scale", MinMaxScaler()),
    ("clf", tree.DecisionTreeClassifier(
        random_state=42, class_weight="balanced"
        ))
    ]
clf_tree_base = Pipeline(estimators)
# Testing with all features
features_list = extract_df_features(df_train) # refresh features_list
test_classifier(clf_tree_base, my_dataset, features_list)
print_log_message("Done.")

# Test the Decision Tree removing highly sparse features
features_removed = ["loan_advances", "director_fees"]
features_list = extract_df_features(df_train)
test_classifier(clf_tree_base, my_dataset, features_list)
print_log_message("Done.")

# Test the Decision Tree varying K best values
features_removed = [
    "loan_advances", "director_fees",
    "total_stock_value"  # Highly correlated with "exercised_stock_options"
    ]
message = "baseline model"
results_tree_fs1 = test_k_values(
    df_train, clf_tree_base, message, my_dataset, features_removed
    )

plt.figure(i)
plt.plot(
    results_tree_fs1["k_list"], results_tree_fs1["precision_list"], "-ro",
    results_tree_fs1["k_list"], results_tree_fs1["recall_list"], '-bo',
    results_tree_fs1["k_list"], results_tree_fs1["f1_list"], '-go'
    )
plt.title("Decision Tree Performance for Different K Best Values")
plt.xlabel("K value")
plt.ylabel("Score")
plt.xticks(PLOTS_XTICKS)
plt.yticks(PLOTS_YTICKS)
plt.legend(("Precision", "Recall", "F1"))
plt.grid(True)
i += 1

# Test the Decision Tree varying K best values using all features
message = "baseline model with all features"
results_tree_all = test_k_values(df_train, clf_tree_base, message, my_dataset)

# Test the Decision Tree varying K best values excluding sparse features
features_removed = [
    "loan_advances", "director_fees",
    "total_stock_value", "deferred_income", "deferral_payments"
    ]
message = "baseline model excluding sparse features"
results_tree_fs2 = test_k_values(
    df_train, clf_tree_base, message, my_dataset, features_removed
    )

plt.figure(i)
plt.plot(
    results_tree_all["k_list"], results_tree_all["f1_list"], "-ro",
    results_tree_fs1["k_list"], results_tree_fs1["f1_list"], "-go",
    results_tree_fs2["k_list"], results_tree_fs2["f1_list"], '-bo'
    )
plt.title("Decision Tree Performance Varying Feature List with K Best")
plt.xlabel("K value")
plt.ylabel("Score")
plt.xticks(PLOTS_XTICKS)
plt.yticks(PLOTS_YTICKS)
plt.legend((
    "All features used", "Baseline model features", "Sparse features excluded"
    ))
plt.grid(True)
i += 1

# Linear SVC varying K best values
estimators = [
    ("scale", MinMaxScaler()),
    ("clf", LinearSVC(random_state=42, class_weight="balanced"))
    ]
clf_linear = Pipeline(estimators)

# Consider all features
message = "Linear SVC using all features"
results_linear_all = test_k_values(df_train, clf_linear, message, my_dataset)

# Exclude sparse features and the correlated feature "total_stock_value"
message = "Linear SVC using feature set 1"
features_removed = ["loan_advances", "director_fees", "total_stock_value"]
results_linear_fs1 = test_k_values(
    df_train, clf_linear, message, my_dataset, features_removed
    )

# Exclude all sparse features
message = "Linear SVC excluding highly sparse or correlated features"
features_removed = [
    "loan_advances", "director_fees",
    "total_stock_value", "deferred_income", "deferral_payments"
    ]
results_linear_fs2 = test_k_values(
    df_train, clf_linear, message, my_dataset, features_removed
    )

plt.figure(i)
plt.plot(
    results_linear_all["k_list"], results_linear_all["f1_list"], "-ro",
    results_linear_fs1["k_list"], results_linear_fs1["f1_list"], "-go",
    results_linear_fs2["k_list"], results_linear_fs2["f1_list"], '-bo'
    )
plt.title("Linear SVC Performance Varying Feature List with K Best")
plt.xlabel("K value")
plt.ylabel("Score")
plt.xticks(PLOTS_XTICKS)
plt.yticks(PLOTS_YTICKS)
plt.legend((
    "All features used", "Feature set 1", "Sparse features excluded"
    ))
plt.grid(True)
i += 1

# Random Forest varying K best values
estimators = [
    ("scale", MinMaxScaler()),
    ("clf", RandomForestClassifier(random_state=42, class_weight="balanced"))
    ]
clf_rforest = Pipeline(estimators)

message = "Random Forest with all features"
results_rforest_all = test_k_values(df_train, clf_rforest, message, my_dataset)

message = "Random Forest with features set 1"
features_removed = ["loan_advances", "director_fees", "total_stock_value"]
results_rforest_fs1 = test_k_values(
    df_train, clf_rforest, message, my_dataset, features_removed
    )

message = "Random Forest excluding all sparse features"
features_removed = [
    "loan_advances", "director_fees",
    "total_stock_value", "deferred_income", "deferral_payments"
    ]
results_rforest_fs2 = test_k_values(
    df_train, clf_rforest, "Random Forest", my_dataset, features_removed
    )

plt.figure(i)
plt.plot(
    results_rforest_all["k_list"], results_rforest_all["f1_list"], "-ro",
    results_rforest_fs1["k_list"], results_rforest_fs1["f1_list"], "-go",
    results_rforest_fs2["k_list"], results_rforest_fs2["f1_list"], '-bo'
    )
plt.title("Random Forest Performance Varying Feature List with K Best")
plt.xlabel("K value")
plt.ylabel("Score")
plt.xticks(PLOTS_XTICKS)
plt.yticks(PLOTS_YTICKS)
plt.legend((
    "All features used", "Feature set 1", "Sparse features excluded"
    ))
plt.grid(True)
i += 1

# SVC varying K best values
estimators = [
    ("scale", MinMaxScaler()),
    ("clf", SVC(class_weight="balanced", random_state=42))
    ]
clf_SVC = Pipeline(estimators)

message = "SVC with all features"
results_svc_all = test_k_values(df_train, clf_SVC, message, my_dataset)

features_removed = ["loan_advances", "director_fees", "total_stock_value"]
message = "SVC with feature set 1"
results_svc_fs1 = test_k_values(
    df_train, clf_SVC, message, my_dataset, features_removed
    )

message = "SVC excluding sparse features"
features_removed = [
    "loan_advances", "director_fees",
    "total_stock_value", "deferred_income", "deferral_payments"
    ]
results_svc_fs2 = test_k_values(
    df_train, clf_SVC, message, my_dataset, features_removed)

plt.figure(i)
plt.plot(
    results_svc_all["k_list"], results_svc_all["f1_list"], "-ro",
    results_svc_fs1["k_list"], results_svc_fs1["f1_list"], "-go",
    results_svc_fs2["k_list"], results_svc_fs2["f1_list"], '-bo'
    )
plt.title("SVC Performance Varying Feature List with K Best")
plt.xlabel("K value")
plt.ylabel("Score")
plt.xticks(PLOTS_XTICKS)
plt.yticks(PLOTS_YTICKS)
plt.legend((
    "All features used", "Feature set 1", "Sparse features excluded"
    ))
plt.grid(True)
i += 1

# F1 score comparison between the models' performance
plt.figure(i)
plt.plot(
    results_tree_all["k_list"], results_tree_all["f1_list"], "-ro",
    results_linear_all["k_list"], results_linear_all["f1_list"], "-go",
    results_rforest_all["k_list"], results_rforest_all["f1_list"], '-bo',
    results_svc_all["k_list"], results_svc_all["f1_list"], '-ko'
    )
plt.title("Models' F1 Score Using all Features as the K Best Input")
plt.xlabel("K value")
plt.ylabel("Score")
plt.xticks(PLOTS_XTICKS)
plt.yticks(PLOTS_YTICKS)
plt.legend((
    "Decition tree", "Linear SVC", "Random Forest", "SVC"
    ))
plt.grid(True)
i += 1

plt.figure(i)
plt.plot(
    results_tree_fs1["k_list"], results_tree_fs1["f1_list"], "-ro",
    results_linear_fs1["k_list"], results_linear_fs1["f1_list"], "-go",
    results_rforest_fs1["k_list"], results_rforest_fs1["f1_list"], '-bo',
    results_svc_fs1["k_list"], results_svc_fs1["f1_list"], '-ko'
    )
plt.title("Models' F1 Score Using Feature Set 1 as the K Best Input")
plt.xlabel("K value")
plt.ylabel("Score")
plt.xticks(PLOTS_XTICKS)
plt.yticks(PLOTS_YTICKS)
plt.legend((
    "Decition tree", "Linear SVC", "Random Forest", "SVC"
    ))
plt.grid(True)
i += 1

plt.figure(i)
plt.plot(
    results_tree_fs2["k_list"], results_tree_fs2["f1_list"], "-ro",
    results_linear_fs2["k_list"], results_linear_fs2["f1_list"], "-go",
    results_rforest_fs2["k_list"], results_rforest_fs2["f1_list"], '-bo',
    results_svc_fs2["k_list"], results_svc_fs2["f1_list"], '-ko'
    )
plt.title("Models' F1 Score Using Feature Set 2 as the K Best Input")
plt.xlabel("K value")
plt.ylabel("Score")
plt.xticks(PLOTS_XTICKS)
plt.yticks(PLOTS_YTICKS)
plt.legend((
    "Decition tree", "Linear SVC", "Random Forest", "SVC"
    ))
plt.grid(True)
i += 1
### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Grid Search on the Decision Tree Classifier
clf = tree.DecisionTreeClassifier(random_state=42)
param_grid = dict(reduce_dim=[None, PCA(1), PCA(2), PCA(5), PCA(7), PCA(10)],
                  clf__max_depth=[None, 1, 5, 10, 100],
                  clf__min_samples_split=[2, 10, 50, 100],
                  clf__min_samples_leaf=[1, 10, 50, 100],
                  clf__min_impurity_split=[0.0, 0.5, 5.0, 10.0, 50.0, 100.0])

# All features
print_log_message("Grid Search on Decision Tree all Features...")
features_list = extract_df_features(df_train)

metrics_tree_optimized_all, clf_tree_optmized_all = tune_hyper_parameters(
    clf, param_grid, my_dataset, features_list, "grid search"
    )

# Feature set 2
print_log_message("Grid Search on Decision Tree Feature Set 2...")

features_removed = [
    "loan_advances", "director_fees",
    "total_stock_value", "deferred_income", "deferral_payments"
    ]
features_list = extract_df_features(df_train, features_removed)

metrics_tree_optimized_fs2, clf_tree_optmized_fs2 = tune_hyper_parameters(
    clf, param_grid, my_dataset, features_list, "grid search"
    )

# K features with best result (Feature Set 3)
print_log_message("Grid Search on Decision Tree K best features...")

features_list = results_tree_fs2["best_features"]

param_grid["reduce_dim"] = [None]
metrics_tree_optimized_fs3, clf_tree_optmized_fs3 = tune_hyper_parameters(
    clf, param_grid, my_dataset, features_list, "grid search"
    )

# Bar plot comparing the f1 scores
f1_fs1 = metrics_tree_optimized_all[2]  # f1 score
f1_fs2 = metrics_tree_optimized_fs2[2]
f1_fs3 = metrics_tree_optimized_fs3[2]
ind = range(1,4)
plt.figure(i)
pfs1, pfs2, pfs3 = plt.bar(ind, [f1_fs1, f1_fs2, f1_fs3])
pfs1.set_facecolor('r')
pfs2.set_facecolor('g')
pfs3.set_facecolor('b')
plt.title("Decision Tree Perfomance after Optimization")
plt.xlabel("Input space for the optimization algorithm")
plt.ylabel("F1 score")
plt.xticks(ind, ("All fetures", "Feature Set 2", "Feature Set 3"))
plt.yticks(PLOTS_YTICKS)
i += 1

# Grid Search on the Linear SVM Classifier
clf = LinearSVC(dual=False, random_state=42)
param_grid = dict(reduce_dim=[None, PCA(1), PCA(2), PCA(5), PCA(10)],
                  clf__penalty=["l1", "l2"],
                  clf__C=[0.1, 1, 10, 100, 1000],
                  clf__class_weight=[None, "balanced"])

# All features
print_log_message("Grid Search on Linear SVC all Features...")
features_list = extract_df_features(df_train)

metrics_linear_optimized_all, clf_linear_optmized_all = tune_hyper_parameters(
    clf, param_grid, my_dataset, features_list, "grid search"
    )

# Feature set 2
print_log_message("Grid Search on Linear SVC Feature Set 2...")

features_removed = [
    "loan_advances", "director_fees",
    "total_stock_value", "deferred_income", "deferral_payments"
    ]
features_list = extract_df_features(df_train, features_removed)

metrics_linear_optimized_fs2, clf_linear_optmized_fs2 = tune_hyper_parameters(
    clf, param_grid, my_dataset, features_list, "grid search"
    )

# K features with best result (Feature Set 3)
print_log_message("Grid Search on Linear SVC K best features...")

features_list = results_linear_fs2["best_features"]

param_grid["reduce_dim"] = [None]
metrics_linear_optimized_fs3, clf_linear_optmized_fs3 = tune_hyper_parameters(
    clf, param_grid, my_dataset, features_list, "grid search"
    )

# Bar plot comparing the f1 scores
f1_fs1 = metrics_linear_optimized_all[2]  # f1 score
f1_fs2 = metrics_linear_optimized_fs2[2]
f1_fs3 = metrics_linear_optimized_fs3[2]
ind = range(1,4)
plt.figure(i)
pfs1, pfs2, pfs3 = plt.bar(ind, [f1_fs1, f1_fs2, f1_fs3])
pfs1.set_facecolor('r')
pfs2.set_facecolor('g')
pfs3.set_facecolor('b')
plt.title("Linear SVC Perfomance after Optimization")
plt.xlabel("Input space for the optimization algorithm")
plt.ylabel("F1 score")
plt.xticks(ind, ("All fetures", "Feature Set 2", "Feature Set 3"))
plt.yticks(PLOTS_YTICKS)
i += 1

# Randomized Search on the Random Forest Classifier
clf = RandomForestClassifier(random_state=42)
param_distr = dict(reduce_dim=[None, PCA(1), PCA(5), PCA(10)],
                  clf__n_estimators=sp_randint(5,100),
                  clf__bootstrap= [True, False],
                  clf__class_weight=[None, "balanced"],
                  clf__max_depth=[None, 1, 5, 10, 100],
                  clf__min_samples_split=sp_randint(2, 100),
                  clf__min_samples_leaf=sp_randint(1, 100),
                  clf__min_impurity_split=[0.0, 0.5, 5.0, 10.0, 50.0, 100.0])

# All features
print_log_message("Grid Search on Random Forest all Features...")
features_list = extract_df_features(df_train)

metrics_rforest_optimized_all, clf_rforest_optmized_all = tune_hyper_parameters(
    clf, param_distr, my_dataset, features_list, "randomized search"
    )

# Feature set 2
print_log_message("Grid Search on Random Forest Feature Set 2...")

features_removed = [
    "loan_advances", "director_fees",
    "total_stock_value", "deferred_income", "deferral_payments"
    ]
features_list = extract_df_features(df_train, features_removed)

metrics_rforest_optimized_fs2, clf_rforest_optmized_fs2 = tune_hyper_parameters(
    clf, param_distr, my_dataset, features_list, "randomized search"
    )

# K features with best result (Feature Set 3)
print_log_message("Grid Search on Random Forest K best features...")

features_list = results_rforest_fs2["best_features"]

param_distr["reduce_dim"] = [None]
metrics_rforest_optimized_fs3, clf_rforest_optmized_fs3 = tune_hyper_parameters(
    clf, param_distr, my_dataset, features_list, "randomized search"
    )

# Bar plot comparing the f1 scores
f1_fs1 = metrics_rforest_optimized_all[2]  # f1 score
f1_fs2 = metrics_rforest_optimized_fs2[2]
f1_fs3 = metrics_rforest_optimized_fs3[2]
ind = range(1,4)
plt.figure(i)
pfs1, pfs2, pfs3 = plt.bar(ind, [f1_fs1, f1_fs2, f1_fs3])
pfs1.set_facecolor('r')
pfs2.set_facecolor('g')
pfs3.set_facecolor('b')
plt.title("Random Forest Perfomance after Optimization")
plt.xlabel("Input space for the optimization algorithm")
plt.ylabel("F1 score")
plt.xticks(ind, ("All fetures", "Feature Set 2", "Feature Set 3"))
plt.yticks(PLOTS_YTICKS)
i += 1

# Randomized Search on the SVM Classifier
clf = SVC(random_state=42)
param_distr = dict(reduce_dim=[None, PCA(1), PCA(5), PCA(10)],
                  clf__class_weight=[None, "balanced"],
                  clf__C=expon(),
                  clf__kernel=["linear", "poly", "rbf", "sigmoid"],
                  clf__gamma=expon(),
                  clf__coef0=expon())

# All features
print_log_message("Grid Search on SVC all Features...")
features_list = extract_df_features(df_train)

metrics_svc_optimized_all, clf_svc_optmized_all = tune_hyper_parameters(
    clf, param_distr, my_dataset, features_list, "randomized search"
    )

# Feature set 2
print_log_message("Grid Search on SVC Set 2...")

features_removed = [
    "loan_advances", "director_fees",
    "total_stock_value", "deferred_income", "deferral_payments"
    ]
features_list_fs2 = extract_df_features(df_train, features_removed)

metrics_svc_optimized_fs2, clf_svc_optmized_fs2 = tune_hyper_parameters(
    clf, param_distr, my_dataset, features_list, "randomized search"
    )

# K features with best result (Feature Set 3)
print_log_message("Grid Search on SVC K best features...")

features_list = results_svc_fs2["best_features"]

param_distr["reduce_dim"] = [None]
metrics_svc_optimized_fs3, clf_svc_optmized_fs3 = tune_hyper_parameters(
    clf, param_distr, my_dataset, features_list, "randomized search"
    )

# Bar plot comparing the f1 scores
f1_fs1 = metrics_svc_optimized_all[2]  # f1 score
f1_fs2 = metrics_svc_optimized_fs2[2]
f1_fs3 = metrics_svc_optimized_fs3[2]
ind = range(1,4)
plt.figure(i)
pfs1, pfs2, pfs3 = plt.bar(ind, [f1_fs1, f1_fs2, f1_fs3])
pfs1.set_facecolor('r')
pfs2.set_facecolor('g')
pfs3.set_facecolor('b')
plt.title("SVC Perfomance after Optimization")
plt.xlabel("Input space for the optimization algorithm")
plt.ylabel("F1 score")
plt.xticks(ind, ("All fetures", "Feature Set 2", "Feature Set 3"))
plt.yticks(PLOTS_YTICKS)
i += 1

# # Example starting point. Try investigating other evaluation techniques!
# from sklearn.cross_validation import train_test_split
# features_train, features_test, labels_train, labels_test = \
#     train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
# SVC with feature set 2 was the best classifier obtained
dump_classifier_and_data(clf_svc_optmized_fs2, my_dataset, features_list_fs2)

plt.show()
