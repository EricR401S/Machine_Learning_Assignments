import pandas as pd
import numpy as np
import pickle
import time

################################
# Load the data
################################
data = pickle.load(open(r"C:\Users\Eric-DQGM\Desktop\a5_q1.pkl", "rb"))

y_train = data["y_train"]
X_train_original = data["X_train"]  # Original dataset
X_train_ohe = data["X_train_ohe"]  # One-hot-encoded dataset

X_test_original = data["X_test"]
X_test_ohe = data["X_test_ohe"]

################################
# Produce submission
################################


def create_submission(confidence_scores, save_path):
    """Creates an output file of submissions for Kaggle

    Parameters
    ----------
    confidence_scores : list or numpy array
        Confidence scores (from predict_proba methods from classifiers) or
        binary predictions (only recommended in cases when predict_proba is
        not available)
    save_path : string
        File path for where to save the submission file.

    Example:
    create_submission(my_confidence_scores, './data/submission.csv')

    """
    import pandas as pd

    submission = pd.DataFrame({"score": confidence_scores})
    submission.to_csv(save_path, index_label="id")


# implement xgboost
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import log_loss
from sklearn.metrics import brier_score_loss
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import LeaveOneOut

# split the training data into training and validation sets
X_train_hr, X_val, y_train_hr, y_val = train_test_split(
    X_train_ohe, y_train, test_size=0.2, random_state=42
)

gpu_dict = {
    "objective": "binary:logistic",
    "tree_method": "gpu_hist",
    "predictor": "gpu_predictor",
    "n_estimators": 1000,
    "max_depth": 5,
    "learning_rate": 0.1,
    "min_child_weight": 1,
    "gamma": 0.0,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "scale_pos_weight": 1,
    "seed": 42,
    # "eval_metric": "error", #auc
    "eval_metric": "auc",
}
# define the grid
param_grid = {
    "max_depth": np.random.choice(
        [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    ),
    "learning_rate": np.random.choice([0.005, 0.01, 0.1, 0.02, 0.03, 0.2]),
    "n_estimators": np.random.choice(
        [
            600,
            700,
            800,
            900,
            1000,
            1100,
            1200,
            1300,
            1400,
            1500,
            1600,
        ]
    ),
    "min_child_weight": np.random.choice([1, 2, 3, 4, 5, 6]),
    "gamma": np.random.choice([0.0, 0.1, 0.15, 0.2, 0.3, 0.4]),
    "subsample": np.random.choice([0.6, 0.7, 0.8, 0.9, 1.0]),
    "colsample_bytree": np.random.choice([0.6, 0.7, 0.8, 0.9, 1.0]),
    "reg_alpha": np.random.choice([0, 0.25, 0.5, 0.75, 1]),
    "reg_lambda": np.random.choice([0, 0.25, 0.5, 0.75, 1]),
    "scale_pos_weight": np.random.choice([1, 2, 3, 4]),  # 1-5
    "early_stopping_rounds": np.random.choice(
        [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500]
    ),
    # "num_boost_round": np.random.choice(
    #     [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 5000, 10000]
    # ),
}

# define the model
model = xgb.XGBClassifier(**gpu_dict)

my_aucs = {}  # auc
my_params = {}
my_scores = {}
my_grid = {}
my_probas = {}
run = 100
best_auc_so_far = 0
start = time.time()
print(f"Starting grid search...narrowed version, for {run} iterations")
print(f"Starting grid search...for {run} iterations")
for i in range(run):
    print(f"Starting iteration {i} of {run}")
    # define the grid
    # param_grid = {
    #     "max_depth": np.random.choice([3, 4, 5, 6, 7, 8, 9, 10]),
    #     "learning_rate": np.random.choice([0.001, 0.01, 0.1, 0.2, 0.3, 1]),
    #     "n_estimators": np.random.choice(
    #         [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    #     ),
    #     "min_child_weight": np.random.choice([1, 2, 3, 4, 5]),
    #     "gamma": np.random.choice([0.0, 0.1, 0.2, 0.3, 0.4]),
    #     "subsample": np.random.choice([0.6, 0.7, 0.8, 0.9, 1.0]),
    #     "colsample_bytree": np.random.choice([0.6, 0.7, 0.8, 0.9, 1.0]),
    #     "reg_alpha": np.random.choice([0, 0.25, 0.5, 0.75, 1]),
    #     "reg_lambda": np.random.choice([0, 0.25, 0.5, 0.75, 1]),
    #     "scale_pos_weight": np.random.choice([1, 2, 3, 4, 5]),
    #     "early_stopping_rounds": np.random.choice(
    #         [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 500, 1000]
    #     ),
    #     "num_boost_round": np.random.choice(
    #         [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 5000, 10000]
    #     ),
    #     "seed": int(np.random.choice(np.linspace(1, 100000, 100000))),
    # }

    # define the modelparam_grid = {

    param_grid = {
        "max_depth": np.random.choice(
            [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 30, 40, 50]  # cut below 10
        ),
        "learning_rate": np.random.choice(
            [0.01, 0.02, 0.03, 0.04, 0.05]
        ),  # 0.01 was a contender
        "n_estimators": np.random.choice(
            [
                1200,
                1300,
                1400,
                1500,
                1600,
                1700,
                1800,
                1900,
                2000,
            ]  # cut from 500, 900
        ),
        "min_child_weight": np.random.choice([1, 2, 3]),  # did not exceed 3
        "gamma": np.random.choice(np.linspace(0, 0.5, 25)),  # between 0 and 0.5
        "subsample": np.random.choice(np.linspace(0.83, 1, 20)),  # cut below 0.8
        "colsample_bytree": np.random.choice(np.linspace(0.6, 1, 20)),  # start 0.4
        "reg_alpha": np.random.choice([0, 0.25, 0.5, 0.75, 1]),
        "reg_lambda": np.random.choice([0, 0.25, 0.5, 0.75, 1]),
        "scale_pos_weight": np.random.choice([1, 2, 3, 4]),
        # "early_stopping_rounds": np.random.choice(
        #     [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 500, 1000]
        # ),
        "early_stopping_rounds": np.random.choice(
            [
                0,
                10,
                20,
                30,
                40,
                50,
                60,
                70,
                80,
                90,
                100,
                200,
                300,
                400,
                500,
                600,
                # 700,
                # 800,
                # 900, sometimes score goes down
                # 1000,
            ]
        ),
        "num_boost_round": np.random.choice(
            [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 5000, 10000]
        ),
        "seed": int(np.random.choice(np.linspace(1, 100000, 100000))),
    }
    print(f"Early stopping rounds : {param_grid['early_stopping_rounds']}")
    # define the model
    model = xgb.XGBClassifier(**gpu_dict)

    # # define the search
    # search = RandomizedSearchCV(model, param_grid, n_iter=10, n_jobs=-1, cv=3, random_state=42)

    # # perform the search
    # results = search.fit(X_train_ohe, y_train)

    # # summarize
    # print('Best Score: %s' % results.best_score_)

    model.set_params(**param_grid)

    # model.fit(X_train_hr, y_train_hr, eval_metric="error", eval_set=[(X_val, y_val)])

    model.fit(X_train_hr, y_train_hr, eval_set=[(X_val, y_val)])

    score = model.score(X_val, y_val)
    # auc_score = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
    # probas = model.predict_proba(X_val)[:, 1]
    my_probas[i] = model.predict_proba(X_val)[:, 1]
    auc_score = roc_auc_score(y_val, my_probas[i])

    params = model.get_params()

    my_aucs[i] = auc_score
    my_params[i] = params
    my_scores[i] = score
    my_grid[i] = param_grid

    print(f"Run : {i} out of {run}")
    if auc_score > best_auc_so_far:
        best_auc_so_far = auc_score
        print(f"New best AUC: {best_auc_so_far}")

# # get the best score and key
# best_key = max(my_scores, key=my_scores.get)
# best_score = max(my_scores.values())
# print("Best Score: %s" % best_score)
# print("Best AUC: %s" % my_aucs[best_key])
# print("Best Key: %s" % best_key)

# # final model
# final_model = xgb.XGBClassifier(**my_params[best_key])
# no_es = {"early_stopping_rounds": 0}
# final_model.set_params(**no_es)
# final_model.fit(X_train_ohe, y_train, verbose=True)

# # make predictions for test data
# y_pred = final_model.predict_proba(X_test_ohe)

# # create submission file
# create_submission(y_pred[:, 1], r"C:\Users\Eric-DQGM\Desktop\submissionxgrs.csv")


# get my top 5 scores and keys
top_5_keys_auc = sorted(my_aucs, key=my_aucs.get, reverse=True)[:5]
top_5_keys_accuracy = sorted(my_scores, key=my_scores.get, reverse=True)[:5]
top_5_scores_accuracy = sorted(my_scores.values(), reverse=True)[:5]
print("Top 5 Scores: %s" % top_5_scores_accuracy)
print("Top 5 AUC: %s" % [my_aucs[i] for i in top_5_keys_auc])


def final_model_submission(my_params, file_name, key, X_train_ohe, y_train, X_test_ohe):
    # final model

    final_model = xgb.XGBClassifier(**my_params[key])
    no_es = {"early_stopping_rounds": 0}
    final_model.set_params(**no_es)
    final_model.fit(X_train_ohe, y_train, verbose=True)

    # make predictions for test data
    y_pred = final_model.predict_proba(X_test_ohe)
    create_submission(y_pred[:, 1], file_name)


# write results of top_5_keys_auc to a csv file

import csv

with open(r"C:\Users\Eric-DQGM\Desktop\top_5_keys_auc.csv", "w") as f:
    writer = csv.writer(f)
    # writer.writerow([top_5_keys_auc])
    for ind in top_5_keys_auc:
        writer.writerow([ind, ":", my_aucs[ind]])
        writer.writerow([ind, ":", my_grid[ind]])
    for key, value in my_aucs.items():
        writer.writerow([key, value])

# write results of top_5_keys_accuracy to a csv file

with open(r"C:\Users\Eric-DQGM\Desktop\top_5_keys_accuracy.csv", "w") as f:
    writer = csv.writer(f)
    # writer.writerow([top_5_keys_auc])
    for ind in top_5_keys_accuracy:
        writer.writerow([ind, ":", my_scores[ind]])
        writer.writerow([ind, ":", my_grid[ind]])
    for key, value in my_scores.items():
        writer.writerow([key, value])

# write results of top_5_scores_accuracy to a csv file

with open(r"C:\Users\Eric-DQGM\Desktop\top_5_scores_accuracy.csv", "w") as f:
    writer = csv.writer(f)
    # writer.writerow([top_5_keys_auc])

    for key, value in my_scores.items():
        writer.writerow([key, value])

# write results of my_params to a csv file

with open(r"C:\Users\Eric-DQGM\Desktop\my_params.csv", "w") as f:
    writer = csv.writer(f)
    # writer.writerow([top_5_keys_auc])

    for key, value in my_params.items():
        writer.writerow([key, value])

# write results of my_grid to a csv file

with open(r"C:\Users\Eric-DQGM\Desktop\my_grid.csv", "w") as f:
    writer = csv.writer(f)
    # writer.writerow([top_5_keys_auc])

    for key, value in my_grid.items():
        writer.writerow([key, value])


final_model_submission(
    my_params,
    r"C:\Users\Eric-DQGM\Desktop\submissionxgrs1kAAUC.csv",
    top_5_keys_auc[0],
    X_train_ohe,
    y_train,
    X_test_ohe,
)

final_model_submission(
    my_params,
    r"C:\Users\Eric-DQGM\Desktop\submissionxgrs1kBAUC.csv",
    top_5_keys_auc[1],
    X_train_ohe,
    y_train,
    X_test_ohe,
)

final_model_submission(
    my_params,
    r"C:\Users\Eric-DQGM\Desktop\submissionxgrs1kCAUC.csv",
    top_5_keys_auc[2],
    X_train_ohe,
    y_train,
    X_test_ohe,
)

final_model_submission(
    my_params,
    r"C:\Users\Eric-DQGM\Desktop\submissionxgrs1kDAUC.csv",
    top_5_keys_auc[3],
    X_train_ohe,
    y_train,
    X_test_ohe,
)

final_model_submission(
    my_params,
    r"C:\Users\Eric-DQGM\Desktop\submissionxgrs1kEAUC.csv",
    top_5_keys_auc[4],
    X_train_ohe,
    y_train,
    X_test_ohe,
)

print("Done with AUC submissions")
# creating the submissions files for the top 5 accuracy scores

final_model_submission(
    my_params,
    r"C:\Users\Eric-DQGM\Desktop\submissionxgrs1kAAccuracy.csv",
    top_5_keys_accuracy[0],
    X_train_ohe,
    y_train,
    X_test_ohe,
)

final_model_submission(
    my_params,
    r"C:\Users\Eric-DQGM\Desktop\submissionxgrs1kBAccuracy.csv",
    top_5_keys_accuracy[1],
    X_train_ohe,
    y_train,
    X_test_ohe,
)

final_model_submission(
    my_params,
    r"C:\Users\Eric-DQGM\Desktop\submissionxgrs1kCAccuracy.csv",
    top_5_keys_accuracy[2],
    X_train_ohe,
    y_train,
    X_test_ohe,
)

final_model_submission(
    my_params,
    r"C:\Users\Eric-DQGM\Desktop\submissionxgrs1kDAccuracy.csv",
    top_5_keys_accuracy[3],
    X_train_ohe,
    y_train,
    X_test_ohe,
)

final_model_submission(
    my_params,
    r"C:\Users\Eric-DQGM\Desktop\submissionxgrs1kEAccuracy.csv",
    top_5_keys_accuracy[4],
    X_train_ohe,
    y_train,
    X_test_ohe,
)

end = time.time()

print("Top 5 Scores: %s" % top_5_scores_accuracy)
print("Top 5 AUC: %s" % [my_aucs[i] for i in top_5_keys_auc])
print("Done")
print("Done with auc version")
print("Time elapsed: %s" % (end - start))
