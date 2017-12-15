import sys
from time import gmtime, strftime

from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline

sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from tester_custom  import test_classifier

def print_log_message(message):
    showtime = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    print "[", showtime, "]:", message

def extract_df_features(df, features_removed=None):
    # Extract all features from DataFrame.
    features_list = df.columns.values.tolist()

    # Remove specified features.
    if features_removed:
        features_list = [
            feature for feature in features_list if feature not in features_removed
            ]
    # poi at first position.
    features_list.insert(0, features_list.pop(features_list.index("poi")))
    return features_list

def test_k_values(df_train, clf, clf_name, my_dataset, features_removed=None):
    # Initialize lists
    k_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    best_f1 = 0
    # Extract features.
    features_list = extract_df_features(df_train, features_removed)

    # Testing for a variety of K best values
    for k in range(1, len(features_list)):
        message = "Training and validating " + clf_name + \
            " and K best = " + str(k) + "..."
        print_log_message(message)

        data = featureFormat(my_dataset, features_list, sort_keys = True)
        labels, features = targetFeatureSplit(data)
        features_selected_idx = SelectKBest(k=k).fit(features, labels).get_support(indices=True)
        features_selected = [features_list[i+1] for i in features_selected_idx]
        features_selected.insert(0, "poi")

        message = "Selected features: " + str(features_selected)
        print_log_message(message)

        # Test the Classifier
        precision, recall, f1 = test_classifier(
            clf, my_dataset, features_selected
            )

        # Assembly the output lists
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
        k_list.append(k)

        if f1 > best_f1:
            best_features = features_selected
            best_f1 = f1

        print_log_message("Done.")

        result = dict(
            k_list=k_list, precision_list=precision_list,
            recall_list=recall_list, f1_list=f1_list,
            best_features=best_features
            )
    return result

def tune_hyper_parameters(
    clf, search_params, my_dataset, features_list, search_strategy
    ):
    estimators = [
        ("normalize", MinMaxScaler()),
        ("reduce_dim", PCA(random_state=42)),
        ("clf", clf)
    ]
    pipe = Pipeline(estimators)
    if search_strategy == "grid search":
        searcher = GridSearchCV(pipe, param_grid=search_params, scoring="f1")
    elif search_strategy == "randomized search":
        searcher = RandomizedSearchCV(
            pipe, param_distributions=search_params,
            scoring="f1", n_jobs=-1, n_iter=1000
            )
    else:
        print "Invalid argument"
        return None

    data = featureFormat(my_dataset, features_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    searcher.fit(features, labels)
    best_clf = searcher.best_estimator_

    message = "Classifier parameters after optimization: " + \
        str(best_clf.get_params())
    print_log_message(message)

    metrics = test_classifier(best_clf, my_dataset, features_list)
    print_log_message("Done.")
    return metrics, best_clf
