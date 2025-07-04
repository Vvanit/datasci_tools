import numpy as np
import pandas as pd
import scipy.stats as stats

def Pointbiserial(X,y):
    y = np.where(y == np.unique(y)[0], 1, 0)
    biserial_res = stats.pointbiserialr(X,y)[1]
    return ('Point Biserial',biserial_res)

def Pointbiserial2(X,y):
    X = np.where(X == np.unique(X)[0], 1, 0)
    biserial_res = stats.pointbiserialr(X,y)[1]
    return ('Point Biserial',biserial_res)

def chi_test(X,y):
    chiRes = stats.chi2_contingency(pd.crosstab(X,y).values)
    return ('Chi-square',chiRes[1])

def anova_test(X,y):
    dic_para = {}
    str_com = "anova = stats.f_oneway("
    group = np.unique(X)
    for x in range(len(group)):
        dic_para["data%s" %x] = y[X == group[x]]
        str_com += "dic_para['data%s']" %x
        if x != len(group)-1:
            str_com += ", "
        else:
            str_com += ")"
    loc = {}
    exec(str_com,{'stats': stats, 'dic_para': dic_para},loc)
    return ('ANOVA',loc['anova'][1])

def pearsonr(X,y):
    personr_res = stats.pearsonr(X,y)[0]
    return ('Personr',personr_res)

def independent_stat(X,y):
    result = []
    for x in range(X.shape[1]):
        if ((type(X[:,x][0]) == np.int64) or (type(X[:,x][0]) == int)) and (type(y[0]) == str) and len(np.unique(y)) == 2:
            biserial_res = Pointbiserial(X[:,x],y)
            result.append(biserial_res)
        elif type(X[:,x][0]) == str and len(np.unique(X[:,x])) == 2 and ((type(y[0]) == np.int64) or (type(y[0]) == int)):
            biserial_res2 = Pointbiserial2(X[:,x],y)
            result.append(biserial_res2)
        elif type(X[:,x][0]) == str and type(y[0]) == str:
            chi_res = chi_test(X[:,x],y)
            result.append(chi_res)
        elif type(X[:,x][0]) == str and ((type(y[0]) == np.int64) or (type(y[0]) == int)):
            anova_res = anova_test(X[:,x],y)
            result.append(anova_res)
        elif ((type(X[:,x][0]) == np.int64) or (type(X[:,x][0]) == int)) and type(y[0]) == str:
            anova_res2 = anova_test(y,X[:,x])
            result.append(anova_res2)
        else:
            personr_res = pearsonr(X[:,x],y)
            result.append(personr_res)
    return result

def independent_test(X,y, cor_bar = 0.55, sig_bar = 0.05):
    list_ = []
    result = independent_stat(X,y)
    for x in range(len(result)):
        stat_name = result[x][0]
        stat_value = result[x][1]
        if stat_name == 'Personr':
            if stat_value > cor_bar:
                stat_result = True
            else: 
                stat_result = False
        else:
            if stat_value < sig_bar:
                stat_result = True
            else: 
                stat_result = False
        list_.append(stat_result)
    return list_
