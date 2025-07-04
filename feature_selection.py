import pandas as pd
import numpy as np

# feature selection
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler
import imblearn
from imblearn.over_sampling import SMOTE,RandomOverSampler
# feature decomposition
from sklearn.decomposition import PCA, TruncatedSVD, FastICA, FactorAnalysis, NMF
from sklearn.random_projection import GaussianRandomProjection,SparseRandomProjection
from sklearn.manifold import TSNE
from utils.classifier_tools import classifiers_best


#======Check catagorical field=========

def check_categorical_column(df,class_col,threshold=0.25):
    num_list = []
    cat_list = []
    date_list = []
    df = df.drop(columns=[class_col])
    column_list = df.columns.tolist()
    for c in column_list:
        n = df[c].unique()
        t = df[c].dtypes.name
        if len(n)!=len(df):
            if t == 'int64' or t=='float64':
                if np.diff(np.diff(np.sort(n))).sum() ==0 or len(n)==2:
                    cat_list.append(c)
                else:
                    num_list.append(c)
            if t == 'object' and (len(n)/len(df))<=threshold:
                try:
                    df[c].apply(pd.to_datetime)
                    date_list.append(c)
                except ValueError:
                    cat_list.append(c)
    return num_list, cat_list, date_list


#======One_hot_feature=========

#=====Main function=====
def one_hot_feature(df,u_id,x,class_col,method_dict,freq_mode):
    
    #method_dict = {'limit_x_most':['C1'],'freq_encoding':['C2','C3'],'target_encoding':['C4']}
    #x for most x feature
    #class_col = instance class
    
    new_col_list = []
    
    for m in method_dict:
        if m == 'limit_x_most':
            df,new_col = limit_x_most(df,u_id,x,method_dict[m])
        if m == 'freq_encoding':
            df,new_col = freq_encoding(df,u_id,method_dict[m],freq_mode=freq_mode)
        if m == 'target_encoding':
            df,new_col = target_encoding(df,class_col,method_dict[m])
        new_col_list+=new_col
    return df,new_col_list

#===== Sub function =====

def limit_x_most(df,u_id,x,col_list): #u_id = unique id for counting
    start_col = set(df.columns)
    for c in col_list:
        # cal_freq
        c_df = df.groupby([c]).agg(count_n = (u_id,'count')).reset_index()
        c_df = c_df.sort_values('count_n',ascending=False)
        c_df['pct']=c_df['count_n']/c_df['count_n'].sum()
        most_x_cat = c_df[c].tolist()[:x]
        most_x = []
        for v in df[c]:
            if v in most_x_cat:
                most_x.append(v)
            else:
                most_x.append('others')
        df['most_x'] = most_x
        df_one_hot = pd.get_dummies(df.most_x, prefix=c)
        df = df.merge(df_one_hot,left_index=True, right_index=True)
        df = df.drop(columns=[c,'most_x'])
    new_col = set(df.columns).difference(start_col)
    return df,new_col

def freq_encoding(df,u_id,col_list,freq_mode='pct'): #u_id = unique id for counting
    start_col = set(df.columns)
    for c in col_list:
        c_df = df.groupby([c]).agg(freq_enc_n = (u_id,'count')).reset_index()
        # c_df = c_df.sort_values('count_n',ascending=False)
        c_df['freq_enc_pct']=c_df['freq_enc_n']/c_df['freq_enc_n'].sum()
        c_df = c_df.rename(columns={'freq_enc_n':c+'_freq','freq_enc_pct': c+'_freq_pct'})
        df = df.merge(c_df,left_on=c,right_on=c)
        if freq_mode == 'pct':
            drop_list = [c,c+'_freq']
        else:
            drop_list = [c,c+'_freq_pct']
        df = df.drop(columns=drop_list)
    new_col = set(df.columns).difference(start_col)
    return df,new_col

def target_encoding(df,class_col,col_list):
    start_col = set(df.columns)
    for c in col_list:
        c_df = df.groupby([c]).agg(sum_t = (class_col,'sum'),count_t =(c,'count')).reset_index()
        c_df['mean_target'] = c_df['sum_t']/c_df['count_t']
        c_df = c_df.rename(columns={'mean_target':c+'_mean_target'})
        df = df.merge(c_df,left_on=c,right_on=c)
        df = df.drop(columns=[c,'sum_t','count_t'])
    new_col = set(df.columns).difference(start_col)
    return df,new_col

#=====Feature selection method=====

#=====Main function=====
def feature_selection(df,uid, class_col=None, cat_list=[] ,date_list=[],mode='n_feature', n_feature=10,method_list=None):
    
    # uid = unique id field
    # class_col = instance class
    # n_feature = n of feature expected
    # cat_list = category column list
    # date_list = date column list
    # =====sample===== 
    # res = feature_selection(df,'c_id', 'claim_flag', ['gender','marital_status','label','province'])
    
    if method_list == None:
        method_list = ['InformationGain','Chisquare','Correlation','VarianceThreshold','MAD','DispersionRatio']
    
    # prep data split to int and str
    df = df.set_index(uid)
    x_num = df.loc[:, ~df.columns.isin(cat_list+date_list+[class_col])] #cut class + date column
    x_cat = df[cat_list]
    if class_col!=None:
        y = df[[class_col]]
    else:
        y = None
    for i in range(len(cat_list)):
        x_cat[cat_list[i]] = x_cat[cat_list[i]].map({v:i for i,v in enumerate(x_cat[cat_list[i]].value_counts().index)})
        
    dict_method = {
        'InformationGain' : info_gain(x_num, y, n_feature,mode)
        , 'Chisquare' :chi_test(x_cat, y, n_feature,cat_list) 
        , 'Correlation' : cor_coeff(x_num,y,n_feature,mode)
        , 'VarianceThreshold' : var_thres(x_num)     
        , 'MAD' : mad(x_num,n_feature,mode)
        , 'DispersionRatio' : dis_ratio(x_num,n_feature,mode)
          }
        
    list_res = []
    list_chi = []
    for model_key in method_list:
        if model_key != 'Chisquare':
            model = dict_method[model_key]
            list_res += model
        else:
            model = dict_method[model_key]
            list_chi += model
  
    res = pd.Series(list_res).value_counts()
    return list(dict(res))[0:n_feature]+list_chi

#===== Sub function =====

def info_gain(x,y,n_feature,mode):
    res = []
    if y != None:
        mi_scores = mutual_info_classif(x, y, discrete_features=True)
        mi_scores = pd.Series(mi_scores, name="MI Scores", index=x.columns)
        mi_scores = mi_scores.sort_values(ascending=False)
        if mode != 'n_feature':
            res = list(dict(mi_scores[mi_scores > mi_scores.mean()]).keys())
        else:
            res = list(dict(mi_scores).keys())[0:n_feature]
    return res

def chi_test(x,y,n_feature,cat_list,chi_sig=0.05):
    res = []
    if y != None:
        if cat_list != None:
            best = SelectKBest(chi2,k=x.shape[1])
            best = best.fit(x,y)
            df_score = pd.DataFrame(best.pvalues_,columns=['p_values'])
            df_score['chi2_values'] = best.scores_
            df_score['columns'] = cat_list
            chi_score = df_score.sort_values(by='p_values')
            chi_score = chi_score[chi_score['p_values'] <= chi_sig]
            if len(chi_score) <= n_feature:
                n_feature = len(chi_score)
            res = list(chi_score['columns'][0:n_feature])
    return res

def cor_coeff(x,y,n_feature,mode,feature_thr=0.75,y_thr=0.1):
    select_list = []
    if y != None:
        col_y = list(y.columns)[0]
        data = pd.merge(x, y, left_index=True, right_index=True)
        cor = data.corr()
        cor = cor.drop(col_y, axis=0)
        cor = abs(cor).sort_values(by=col_y, ascending=False, axis = 0)
        if mode != 'n_feature':
            cor = cor[cor[col_y]>=y_thr]
        black_list = []
        for r in cor.index:
            if r not in black_list:
                select_list.append(r)
            black_list+=list(cor.loc[r][(cor.loc[r]>=feature_thr)&(cor.loc[r]<1)].index)
            if len(select_list)==n_feature and mode == 'n_feature':
                break
    elif mode!='n_feature':
        corr_mtx = x.corr()
        # corr_mtx
        feature_group={}
        black_list = set()
        # corr_mtx.columns
        for c in corr_mtx.columns:
            if c not in black_list:
                f = corr_mtx[c][abs(corr_mtx[c])>=0.95].index.tolist()
                new = set(f)-black_list
                feature_group[c] = list(new)
                black_list=black_list.union(new)
        select_list=list(feature_group.keys())
    return select_list

def var_thres(x,var_thr=0.5):
    var_thres=VarianceThreshold(var_thr)
    t = var_thres.fit_transform(x)
    new_cols = var_thres.get_support()
    var_res = x.iloc[:,new_cols]
    return list(var_res.columns)

def mad(x,n_feature,mode):
    mean_abs = np.sum(np.abs(x - np.mean(x, axis = 0)), axis = 0)/x.shape[0]
    mean_abs = mean_abs.sort_values(ascending=False)
    if mode != 'n_feature':
        res = list(dict(mean_abs[mean_abs > mean_abs.mean()]).keys())
    else:
        res = list(dict(mean_abs).keys())[0:n_feature]
    return res

def dis_ratio(x,n_feature,mode,thr = 1):
    x_new = x+1
    am = np.mean(x_new, axis = 0)
    gm = np.power(np.prod(x_new, axis = 0),1/x_new.shape[0])
    disp_ratio = am/gm
    disp_ratio = disp_ratio.sort_values(ascending=False)
    if mode != 'n_feature':
        res = list(dict(disp_ratio).keys())[0:n_feature]
    else:
        res= list(dict(disp_ratio[disp_ratio > thr]).keys())
    return res
#=====================================================
def min_max_scale(X):
    scaler = MinMaxScaler()
    x_scale = scaler.fit_transform(X)
    return x_scale,scaler

#=====sampling method=====

def apply_SMOTE(X,y):
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)
    return X_res, y_res

def apply_upsampling(X,y):
    ros = RandomOverSampler(random_state=42)
    X_res, y_res = ros.fit_resample(X, y)
    return X_res, y_res
#=====================================
def pick_component(criteria_list,component_list,threshold = 0.95,min_component=3):
    total = 0
    for i in range(len(criteria_list)):
        total+=criteria_list[i]
        if total>=threshold:
            break
    i= i+1
    if i<=min_component:
        i = min_component
    return i

#=====get decomposition feature======

#=====main function=====
def get_decomposition_feature(x,technique_list=None,max_components=10,random_state=42):
    
    # x = dataframe
    # technique_list = ['svd','ica']
    
    decomp_dict = {}
    
    if technique_list == None:
        technique_list = ['pca','svd','ica','nmf','gr','sr']
    
    #Original
    decomp_dict['org']={'x':x.to_numpy(),'n':len(x.columns)}
    x_scale,scaler = min_max_scale(x)
    
    pca_components = min(len(x),len(x.columns))-1
    svd_components = min(max_components,len(x.columns))
    
    technuque_dict = {'pca':pca(x_scale,pca_components),
                 'svd':svd(x_scale,svd_components, random_state),
                 'ica':ica(x_scale,max_components, random_state),
                 'nmf':nmf(x_scale,max_components,random_state),
                 'gr':gr(x_scale,max_components),
                 'sr':sr(x_scale,max_components),
                }
    for t in technique_list:
        decomp_dict[t]=technuque_dict[t]
        
    return decomp_dict 

# fit model by decomposition feature
def fit_model_decomposition(decomp_dict,y,sampling='upsampling'):    
    res_df = pd.DataFrame()
    for technique in decomp_dict.keys():
        if sampling != 'upsampling':
            X_res, y_res = apply_SMOTE(decomp_dict[technique]['x'], y)
        else:
            X_res, y_res = apply_upsampling(decomp_dict[technique]['x'], y)
        sum_row, models_scores_table = classifiers_best(X_res,y_res)
        models_scores_table['decomp_technique']=technique
        models_scores_table['sampling']=sampling
        res_df = res_df.append(models_scores_table.reset_index(), ignore_index=True)
    
    return res_df

#=====decomposition method=====

#PCA
def pca(x,max_components,svd_solver='arpack'):
    pca = PCA(n_components=max_components, svd_solver='arpack')
    pca = pca.fit(x)
    variance_ratio=pca.explained_variance_ratio_
    x_decomp = pca.transform(x)
    pick_comp = pick_component(variance_ratio,x_decomp)
    x_decomp = x_decomp[:,:pick_comp]
    return {'x':x_decomp,'variance':variance_ratio,'n':pick_comp,'model':pca}

#SVD
def svd(x,max_components, random_state):
    svd = TruncatedSVD(n_components=max_components, random_state=random_state)
    svd = svd.fit(x)
    variance_ratio=svd.explained_variance_ratio_
    x_decomp = svd.transform(x)
    pick_comp = pick_component(variance_ratio,x_decomp)
    return {'x':x_decomp,'variance':variance_ratio,'n':pick_comp,'model':svd}

#FAST ICA
def ica(x,max_components, random_state):
    ica = FastICA(n_components=max_components, random_state=random_state)
    x_decomp = ica.fit_transform(x)
    return {'x':x_decomp,'n':max_components,'model':ica}

#FactorAnalysis
def fa(x,max_components):
    Fa = FactorAnalysis(n_components = max_components)
    x_decomp = Fa.fit_transform(x)
    return {'x':x_decomp,'n':max_components,'model':Fa}

#NonNegative Matrix Factorization
def nmf(x,max_components,random_state):
    Nmf = NMF(n_components=max_components, random_state=random_state)
    x_decomp = Nmf.fit_transform(x)
    return {'x':x_decomp,'n':max_components,'model':Nmf}

#Gaussian Random Projection
def gr(x,max_components,eps=0.1):
    GR = GaussianRandomProjection(n_components = max_components, eps=eps)
    x_decomp = GR.fit_transform(x)
    return {'x':x_decomp,'n':max_components,'model':GR}

#Sparse Random Projection
def sr(x,max_components,eps=0.1):
    SR = SparseRandomProjection(n_components = max_components, eps=eps)
    x_decomp = SR.fit_transform(x)
    return {'x':x_decomp,'n':max_components,'model':SR}
