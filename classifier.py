import pandas as pd
import numpy as np
import os

from sklearn.datasets import make_classification

from sklearn.model_selection import cross_validate, StratifiedKFold # cross_val_score

from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score #, confusion_matrix

from sklearn.naive_bayes import GaussianNB #, CategoricalNB
# from sklearn.gaussian_process import GaussianProcessClassifier
# from sklearn.gaussian_process.kernels import RBF
from sklearn.neighbors import KNeighborsClassifier #, RadiusNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, NuSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
# ,BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier, StackingClassifier, VotingClassifier, HistGradientBoostingClassifier 
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import GridSearchCV

import pickle

import warnings
warnings.filterwarnings('ignore')

def classifiers_best(
    X, y, dict_basemodel=None
    , random_state = 0
    , n_splits = 5
    , return_train_score=False
    , n_jobs = None
):
    
    if dict_basemodel == None:
        dict_basemodel = {
            'GaussianNB' : GaussianNB()
            , 'KNeighborsClassifier' : KNeighborsClassifier(
                n_neighbors = 5
                , weights = 'uniform'
                , algorithm = 'auto'
            )
            , 'LogisticRegression' : LogisticRegression(
                random_state = random_state
            )
            , 'LinearSVC' : LinearSVC(
                penalty = 'l2'
                , loss = 'squared_hinge'
                , dual = False
            )
            , 'NuSVC' : NuSVC(
                nu = 0.5
                , kernel = 'rbf'
                , gamma = 'scale'
            )
            , 'SVC' : SVC(
                C = 1.0
                , kernel = 'rbf'
                , gamma = 'scale'
            )
            , 'DecisionTreeClassifier' : DecisionTreeClassifier(
                criterion = 'gini'
                , splitter = 'best'
            )
            , 'AdaBoostClassifier' : AdaBoostClassifier(
                n_estimators = 50
                , learning_rate = 1
                , algorithm = 'SAMME.R'
                , random_state = random_state
            )
            , 'RandomForestClassifier' : RandomForestClassifier(
                n_estimators = 100
                , criterion = 'gini'
            )
            , 'QuadraticDiscriminantAnalysis' : QuadraticDiscriminantAnalysis()
            # , 'MLPClassifier' : MLPClassifier(
            #     hidden_layer_sizes = (100,)
            #     , activation = 'relu'
            #     , solver = 'adam'
            #     , learning_rate = 'constant'
            #     , random_state = random_state
            # )
        }
        
    cv = StratifiedKFold(n_splits = n_splits, random_state = random_state, shuffle = True)

    scoring = {
        'accuracy':make_scorer(accuracy_score) 
        , 'precision':make_scorer(precision_score)
        , 'recall':make_scorer(recall_score) 
        , 'f1_score':make_scorer(f1_score)
              }
    
    models_scores_table = []
    for model_key in dict_basemodel:
        model = dict_basemodel[model_key]
        dict_cv = cross_validate(model, X, y, cv = cv, scoring = scoring, return_train_score = return_train_score, n_jobs = n_jobs)
        dict_cv = {k : round(np.mean(dict_cv[k]),4) for k in dict_cv}
        dict_cv['model_key'] = model_key
        models_scores_table.append(dict_cv)
    models_scores_table = pd.DataFrame(models_scores_table).fillna(0).set_index('model_key')
    sum_row = models_scores_table[[x for x in models_scores_table.columns if 'time' in x]].idxmin(axis=0).to_dict()
    sum_row.update(models_scores_table[[x for x in models_scores_table.columns if 'time' not in x]].idxmax(axis=0).to_dict())
    sum_row = pd.DataFrame([sum_row],index=['Best'])
    
    return sum_row, models_scores_table

def GridSearchCV_auto(
    model, X, y
    , parameters = None
    , scoring = make_scorer(accuracy_score)
    , n_splits = 5
    , return_train_score = False
    , random_state = 0
    , n_jobs = None
):
    if parameters == None :
        if str(model) == str(GaussianNB()) :
            parameters = {}
            return pd.DataFrame([{'params':parameters}])
        elif str(model) == str(KNeighborsClassifier()) :
            parameters = {
                'n_neighbors' : [4,6]
                , 'weights' : ('uniform','distance')
                , 'algorithm' : ('auto','ball_tree','kd_tree','brute')
                , 'leaf_size' : [30,32]
            }
        elif str(model) == str(LogisticRegression()) :
            parameters = {
                'penalty' : ('none','l1','l2','elasticnet')
                , 'solver' : ('newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga')
                , 'random_state' : [0]
                , 'max_iter' : [100,200]
            }
        elif str(model) == str(LinearSVC()):
            parameters = {
                'penalty' : ('l1', 'l2')
                , 'loss' : ('hinge', 'squared_hinge')
                , 'random_state' : [0]
                , 'max_iter' : [1000,1100]
            }
        elif str(model) == str(NuSVC()):
            parameters = {
                'nu' : [0.4,0.6]
                , 'kernel' : ('linear', 'poly', 'rbf', 'sigmoid')
                , 'gamma' : ('scale', 'auto')
                , 'random_state' : [0]
            }
        elif str(model) == str(SVC()):
            parameters = {
                'kernel' : ('linear', 'poly', 'rbf', 'sigmoid')
                , 'gamma' : ('scale', 'auto')
                , 'random_state' : [0]
            }
        elif str(model) == str(DecisionTreeClassifier()):
            parameters = {
                'criterion' : ('gini', 'entropy')
                , 'splitter' : ('best', 'random')
                , 'max_features' : ('auto', 'sqrt', 'log2')
                , 'random_state' : [0]
            }
        elif str(model) == str(AdaBoostClassifier()):
            parameters = {
                'n_estimators' : [50,60]
                , 'algorithm' : ('SAMME', 'SAMME.R')
                , 'random_state' : [0]
            }
        elif str(model) == str(RandomForestClassifier()):
            parameters = {
                'n_estimators' : [100,200]
                , 'criterion' : ('gini', 'entropy')
                , 'max_features' : ('auto', 'sqrt', 'log2')
                , 'random_state' : [0]
            }
        elif str(model) == str(QuadraticDiscriminantAnalysis()) :
            parameters = {}
            return pd.DataFrame([{'params':parameters}])
        elif str(model) == str(MLPClassifier()):
            parameters = {
                'hidden_layer_sizes' : ((100,))
                , 'activation' : ('identity', 'logistic', 'tanh', 'relu')
                , 'solver' : ('lbfgs', 'sgd', 'adam')
                , 'learning_rate' : ('constant', 'invscaling', 'adaptive')
                , 'random_state' : [0]
                , 'early_stopping' : [True]
            }
        else:
            return 'please pass parameters'
    
    cv = StratifiedKFold(n_splits = n_splits, random_state = random_state, shuffle = True)
    
    model = GridSearchCV(model, parameters, cv = cv, scoring = scoring, return_train_score = return_train_score, n_jobs = n_jobs)
    model.fit(X,y)
    
    cv_results_ = pd.DataFrame(model.cv_results_)
    if return_train_score :
        cv_results_['mean_diff_score'] = abs(cv_results_['mean_train_score'] - cv_results_['mean_test_score'])
        cv_results_ = cv_results_.sort_values(['rank_test_score','mean_diff_score','mean_score_time'],ascending=True).reset_index(drop=True)
    else:
        cv_results_ = cv_results_.sort_values(['rank_test_score','mean_score_time'],ascending=True).reset_index(drop=True)
        
    return cv_results_

def fit_and_save(model, params, local_dir='.'):
    if str(model) == str(GaussianNB()) :
        model = GaussianNB(**params)
    elif str(model) == str(KNeighborsClassifier()) :
        model = KNeighborsClassifier(**params)
    elif str(model) == str(LogisticRegression()) :
        model = LogisticRegression(**params)
    elif str(model) == str(LinearSVC()) :
        model = LinearSVC(**params)
    elif str(model) == str(NuSVC()) :
        model = NuSVC(**params)
    elif str(model) == str(SVC()) :
        model = SVC(**params)
    elif str(model) == str(DecisionTreeClassifier()) :
        model = DecisionTreeClassifier(**params)
    elif str(model) == str(AdaBoostClassifier()) :
        model = AdaBoostClassifier(**params)
    elif str(model) == str(RandomForestClassifier()) :
        model = RandomForestClassifier(**params)
    elif str(model) == str(QuadraticDiscriminantAnalysis()) :
        model = QuadraticDiscriminantAnalysis(**params)
    elif str(model) == str(MLPClassifier()) :
        model = MLPClassifier(**params)
    else:
        model = 'model not support'

    model.fit(X,y)
    filename = str(model)+'.pickle'
    pickle.dump(model, open(filename, 'wb'))
  
    return
