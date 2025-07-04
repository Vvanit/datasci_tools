## for notebook path
import sys
sys.path.insert(1,'../')

from utils.database import pdReadSQL
from utils.aws.s3 import uploadFileToS3,downloadFileFromS3,push_json_s3,get_json_s3,read_tablefile
from dateutil.relativedelta import relativedelta
import pandas as pd
import numpy as np
import re
import joblib
import json
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import datetime as dt
from datetime import datetime, timedelta
import calendar
from dateutil.relativedelta import relativedelta

import tensorflow as tf

# from tensorflow import keras
# from keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import classification_report,confusion_matrix
import matplotlib.pyplot as plt

tf.get_logger().setLevel('ERROR')

def prep_data(order,product_tx,bucket,cat_list_key,month=24,retrain=False):
    
    # get data
    today = datetime.today()
    res = calendar.monthrange(today.year, today.month)
    day = res[1]
    last_day = f"{today.year}-{today.month}-{day}"
    con_date = datetime.strptime(last_day, '%Y-%m-%d') + timedelta(days=1) - relativedelta(months= month+1)

    item_info = product_tx[product_tx['order_date'] >= con_date]
    item_info = item_info[(item_info['product_sales'] >= 1000) & (item_info['category_lev_2'].notna())]
    item_info = item_info.groupby(['order_id','category_lev_2','lead_id']).agg(order_date = ('order_date','max'), qty = ('quantity','sum'),
                                                                   sales = ('product_sales','sum')).reset_index()
    item_info['yyyymm'] = item_info['order_date'].dt.strftime('%Y%m').astype(int)
    item_info = item_info.rename(columns = {'category_lev_2':'cat_lv2'})

    header = order[order['order_date'] >= con_date]
    header['yyyymm'] = header['order_date'].dt.strftime('%Y%m').astype(int)
    header = header[['order_id','order_date','yyyymm','lead_id']]
    header = header.sort_values(by = ['lead_id', 'order_date','order_id'])
    header = header.sort_values(by = ['lead_id', 'order_date','order_id'])
    header['prev_order_id'] = header.groupby(['lead_id'])['order_id'].shift(1)
    header['prev_order_id'] = np.where(header['order_id'] == header['prev_order_id'], np.nan, header['prev_order_id'])
    header['repurchase_day'] = header.groupby(['lead_id'])['order_date'].shift(1)
    header['repurchase_day'] = (header['order_date'] - header['repurchase_day']).dt.days
    
    fix_feature=['repurchase_day','tx_cnt','avg_tx']
    retrain = retrain
    feature_column = fix_feature.copy()
    class_column = []
    # item_info = pdReadSQL(vault_path,sql_item_info.format(month))
    #     data_start_day = item_info['order_date'].max().replace(day=1)+ relativedelta(months=-25)
    last_yyyymm = int((item_info['order_date'].max().replace(day=1)+ relativedelta(months=-1)).strftime('%Y%m'))
    #     item_info=item_info[item_info['order_date']>=data_start_day]
    total_sales = item_info['sales'].sum()
    total_qty = item_info['qty'].sum()
    selected_cat = item_info.groupby(['cat_lv2']).agg(qty=('qty','sum'),sales=('sales','sum'),tx_cnt=('order_id','nunique'))
    selected_cat['qty_ratio'] = (selected_cat['qty']/total_qty)*100
    selected_cat['sales_ratio'] = (selected_cat['sales']/total_sales)*100
    selected_cat = selected_cat[selected_cat['tx_cnt']>=100]
    selected_cat = selected_cat[(selected_cat['qty_ratio']>=1)&(selected_cat['sales_ratio']>=1)].reset_index()
    qty_coverage = selected_cat['qty_ratio'].sum()
    sale_coverage = selected_cat['sales_ratio'].sum()
    top_tx_cat = selected_cat.sort_values(by=['tx_cnt'],ascending=False)['cat_lv2'][:3].to_list()

    #     if sale_coverage < 80:
    #         retrain = True

    sales_txt = 'sales coverage {}%, Qty coverage {}%'.format('%.2f'%(sale_coverage),'%.2f'%(qty_coverage))
    cat_list = selected_cat['cat_lv2'].tolist()
    cat_list.sort()

    try:
        ref_cat_list = get_json_s3(bucket,cat_list_key)
    except:
    #     print('ref_cat_dict not avaliable')
        ref_cat_list = None

    if cat_list != ref_cat_list:
        retrain = True
        push_json_s3(cat_list,bucket,cat_list_key)
        ref_cat_list = cat_list

    item_info = item_info[item_info['cat_lv2'].isin(cat_list)]
    item_info['tx_cnt']=item_info['order_id']
    agg = {'sales':'sum','tx_cnt':'nunique'}
    cus_sales = item_info[['order_id','lead_id','sales','tx_cnt']].groupby(['lead_id','order_id']).agg(agg).groupby(level=0).cumsum().reset_index()
    item_info = item_info.drop(columns=['tx_cnt'])
    cus_sales['avg_tx']= cus_sales['sales']/cus_sales['tx_cnt']
    cus_sales =cus_sales.rename(columns={'order_id':'prev_order_id'})

    header = pd.merge(header,item_info[['order_id','cat_lv2']],how='left',left_on='order_id',right_on='order_id')

    last_info = header.copy()
    last_info = last_info.rename(columns={'cat_lv2':'prev_cat'})
    last_info = last_info[last_info['prev_cat'].notnull()]
    # last_info['repurchase_day'] = (dt.date.today()-last_info['order_date'].dt.date).dt.days
    last_info['repurchase_day'] = (pd.Timestamp('today') - last_info['order_date']).dt.days
    last_info = pd.get_dummies(data = last_info,columns=['prev_cat']).groupby(['lead_id','order_id']).max().reset_index()

    header=header[header['prev_order_id'].notnull()]
    header=header.rename(columns={"cat_lv2": "pred_cat"})
    header = header[header['pred_cat'].notnull()]
    header = pd.merge(header,item_info.rename(columns={'order_id': "prev_order_id"})[['prev_order_id','cat_lv2']],how='left',left_on='prev_order_id',right_on='prev_order_id')
    header = header.rename(columns={"cat_lv2": "prev_cat"})
    header = header[header['prev_cat'].notnull()]
    cus = header.rename(columns={'prev_cat':'cus_cat'})
    key_column = list(set(cus.columns)-set(fix_feature))
    cus = pd.get_dummies(data = cus,columns=['cus_cat'])
    purchased_cat = list(set(cus.columns)-set(fix_feature)-set(key_column))
    feature_column += list(set(cus.columns)-set(fix_feature)-set(key_column))
    feature_column.sort()
    cus = cus.groupby(['order_id','lead_id']).max().reset_index()
    cus = cus.drop(columns=['order_id','order_date','yyyymm','pred_cat','repurchase_day']).groupby(['lead_id','prev_order_id']).sum().groupby(level=0).cumsum().reset_index()
    cus = pd.merge(cus,cus_sales,how='left',left_on=['lead_id','prev_order_id'],right_on=['lead_id','prev_order_id'])

    X_last = last_info.sort_values(by=['order_date']).groupby(['lead_id']).agg(transaction_No=('order_id','last')).reset_index()
    X_last = pd.merge(X_last,last_info,how='left',left_on=['lead_id','transaction_No'],right_on=['lead_id','order_id'])
    X_last = pd.merge(X_last,cus.loc[:, cus.columns != 'sales'],how='left',left_on=['lead_id','prev_order_id'],right_on=['lead_id','prev_order_id'])
    X_last = X_last.dropna(subset=['tx_cnt','avg_tx'])

    key_column = list(set(header.columns)-set(fix_feature))
    header = pd.get_dummies(data = header,columns=['prev_cat'])
    feature_column += list(set(header.columns)-set(fix_feature)-set(key_column))
    feature_column.sort()
    header = header.groupby(['order_id','lead_id','pred_cat']).max().reset_index()
    header = pd.merge(header,cus,how='left',left_on=['lead_id','prev_order_id'],right_on=['lead_id','prev_order_id'])

    X_last = X_last.set_index('lead_id')
    X_last = X_last[feature_column]


    key_column = list(set(header.columns)-set(fix_feature))
    header = pd.get_dummies(data = header,columns=['pred_cat'])
    class_column += list(set(header.columns)-set(fix_feature)-set(key_column))
    class_column.sort()
    prep_df = header

    return prep_df, last_yyyymm, retrain,class_column,feature_column,cat_list,top_tx_cat,purchased_cat,X_last,sales_txt

def train_test_prep(prep_df,last_yyyymm,class_column,feature_column,X_last,retrain=False,bucket='growthai-proplugin'
                     ,scaler_key='preprocess_data/Up_Sell_Cross_Sell_v2/upsell_crosssell_scaler.joblib'):
    
    data_dict={}

    Y_train = prep_df[prep_df['yyyymm']<last_yyyymm][class_column].to_numpy()
    X_train = prep_df[prep_df['yyyymm']<last_yyyymm][feature_column].to_numpy()
#     print(Y_train.shape,X_train.shape)

    Y_val = prep_df[prep_df['yyyymm']==last_yyyymm][class_column].to_numpy()
    X_val = prep_df[prep_df['yyyymm']==last_yyyymm][feature_column].to_numpy()
    
    X_last =X_last.to_numpy()
    
    if retrain==True:
        sc = MinMaxScaler()
        X_train = sc.fit_transform(X_train)
        data_dict['X_val'] = sc.transform(X_val)
        X_last = sc.transform(X_last)
        
        joblib.dump(sc,'upsell_crosssell_scaler.joblib')
        uploadFileToS3(bucket=bucket, key=scaler_key, localPath='upsell_crosssell_scaler.joblib')
    else:      
        downloadFileFromS3(bucket=bucket, key=scaler_key, localPath='upsell_crosssell_scaler.joblib')
        sc = joblib.load('upsell_crosssell_scaler.joblib')
        
        X_train = sc.transform(X_train)
        data_dict['X_val'] = sc.transform(X_val)
        X_last = sc.transform(X_last)
    
    X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.2)
    data_dict['X_train'] = X_train
    data_dict['X_test'] = X_test
    data_dict['Y_train'] = Y_train
    data_dict['Y_test'] = Y_test
    data_dict['Y_val'] = Y_val
    data_dict['X_last'] = X_last
    return data_dict

def retrain_model(data_dict,model_key='preprocess_data/Up_Sell_Cross_Sell_v2/upsell_crosssell_ann.h5',bucket='growthai-proplugin',show_plot=False):
    X_train=data_dict['X_train']
    X_test=data_dict['X_test']
    y_train=data_dict['Y_train']
    y_test=data_dict['Y_test']
    
#     es = EarlyStopping(monitor='val_loss', mode='min', patience=10)
#     mc = ModelCheckpoint('upsell_crosssell_ann.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

    model = tf.keras.Sequential()
    model.add(Dense(units=64, activation='relu', input_dim=X_train.shape[1]))
#     model.add(Dropout(0.3))
    model.add(Dense(units=128, activation='relu'))
#     model.add(Dropout(0.3))
    model.add(Dense(units=256, kernel_initializer='uniform', activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(units=512, kernel_initializer='uniform', activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(units=256, kernel_initializer='uniform', activation='relu'))
#     model.add(Dropout(0.2))
    model.add(Dense(units=128, kernel_initializer='uniform', activation='relu'))
#     model.add(Dropout(0.3))
    model.add(Dense(units=64, activation='relu'))
#     model.add(Dropout(0.3))
    model.add(Dense(y_train.shape[1], activation='softmax'))
    model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])
    #     model.summary()

#     history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=20, epochs=100,callbacks=[es,mc])
#     history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=50, epochs=200,callbacks=[mc])
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=50, epochs=200)
    
    model.save('upsell_crosssell_ann.h5')
    
    if show_plot == True:
        # plot training history
        plt.plot(history.history['accuracy'], label='train')
        plt.plot(history.history['val_accuracy'], label='test')
        plt.legend()
        plt.show()

#     model_key = 'preprocess_data/Up_Sell_Cross_Sell/upsell_crosssell_ann.h5'
    uploadFileToS3(bucket=bucket, key=model_key, localPath='upsell_crosssell_ann.h5')
    return model

def get_model(bucket='growthai-proplugin',model_key='preprocess_data/Up_Sell_Cross_Sell_v2/upsell_crosssell_ann.h5'):
    downloadFileFromS3(bucket=bucket, key=model_key, localPath='upsell_crosssell_ann.h5')
    model = tf.keras.models.load_model('upsell_crosssell_ann.h5')
    return model

def top_index(cat_list,top_tx_cat):
    top_tx_index=[]
    for c in top_tx_cat:
        top_tx_index.append(cat_list.index(c))
    return top_tx_index

def hybrid_pred(top_tx_index,y,mode='Hybrid'):
    y_hybrid = []
    if mode == 'Hybrid':
        y = np.argsort(y)[::-1].tolist()
        for i in range(len(top_tx_index)):
            if top_tx_index[i] not in y_hybrid and top_tx_index!=[]:
                y_hybrid.append(top_tx_index[i])
            if y[i] not in y_hybrid and y!=[]:
                y_hybrid.append(y[i])
    if mode == 'Model':
        y_hybrid = y
    if mode == 'Stat':
        y_hybrid = top_tx_index
    return y_hybrid

def top3_pred(y_pred,Y_val,cat_list,top_tx_cat,mode='Hybrid'):
    y_top_list = []
    top_tx_index = top_index(cat_list,top_tx_cat)
    for i in range(len(y_pred)):
        rec_list = hybrid_pred(top_tx_index,y_pred[i],mode=mode)[:6]
        if np.argmax(Y_val[i]) in rec_list:
            y_top_list.append(list(Y_val[i]))
        else:
            y_top_list.append(list(y_pred[i]))      
    return y_top_list

def model_validation(model,X_val,Y_val,cat_list=[],top_tx_cat=[],top3_accuracy = True,mode='Hybrid'):
    y_pred = model.predict(X_val)
    if top3_accuracy:
        y_pred = top3_pred(y_pred,Y_val,cat_list,top_tx_cat,mode=mode)        
    pred_list=[]
    act_list = []
    for pred in range(len(y_pred)):
        pred_list.append(np.argmax(y_pred[pred]))
        act_list.append(np.argmax(Y_val[pred]))
    res = classification_report(act_list,pred_list,output_dict=True)['accuracy']*100
    res_txt = '%.3f'%(classification_report(act_list,pred_list,output_dict=True)['accuracy']*100)+'%'
    return res,res_txt

def upsell_crosssell_result(X_last,y_pred,purchased_cat,cat_list):
    Upsell_list = []
    CrossSell_list = []
    cus_list = X_last.index.to_list()
    existing = X_last[purchased_cat].to_numpy()
    for i in range(len(y_pred)):
        sub_u = []
        sub_c = []
        for c in np.argsort(y_pred[i])[::-1].tolist(): #sort desc by softmax value
            if existing[i][c]==1:
                sub_u.append(cat_list[c])
            else:
                sub_c.append(cat_list[c])
        Upsell_list.append('|'.join(sub_u[:3]))
        CrossSell_list.append('|'.join(sub_c[:3]))
    recom_df = pd.DataFrame({'lead_id': cus_list, 'sgmt_recom_upsell': Upsell_list, 'sgmt_recom_cross_sell': CrossSell_list})
    return recom_df

def get_upsell_cross_sell_recom(order,product_tx,bucket,cat_list_key,scaler_key,model_key):
    
    prep_df, last_yyyymm, retrain,class_column,feature_column,cat_list,top_tx_cat,purchased_cat,X_last,sales_txt = prep_data(order,product_tx,bucket,cat_list_key)
    model = get_model(bucket=bucket,model_key=model_key)
    if retrain == False:
        data_dict = train_test_prep(prep_df,last_yyyymm,class_column,feature_column,X_last,bucket=bucket,scaler_key=scaler_key)
        res_hybrid,txt_hybrid = model_validation(model,data_dict['X_val'],data_dict['Y_val'],cat_list=cat_list,top_tx_cat=top_tx_cat,top3_accuracy = True)
    else:
        res_hybrid = 0

    if res_hybrid<51 or retrain == True:
        # print('retraining model')
        data_dict = train_test_prep(prep_df,last_yyyymm,class_column,feature_column,X_last,retrain=retrain,bucket=bucket,scaler_key=scaler_key)
        model = retrain_model(data_dict,model_key=model_key,bucket=bucket)
        res_hybrid,txt_hybrid = model_validation(model,data_dict['X_val'],data_dict['Y_val'],cat_list=cat_list,top_tx_cat=top_tx_cat,top3_accuracy = True,mode='Hybrid')
    res_stat,txt_stat = model_validation(model,data_dict['X_val'],data_dict['Y_val'],cat_list=cat_list,top_tx_cat=top_tx_cat,top3_accuracy = True,mode='Stat')
    res_model,txt_model = model_validation(model,data_dict['X_val'],data_dict['Y_val'],cat_list=cat_list,top3_accuracy = True,mode='Model')
    model_stat = 'UpSell CrossSell Hybrid Model accuracy {} /Stat {} /ANN {}'.format(txt_hybrid,txt_stat,txt_model)
        
    y_pred = model.predict(data_dict['X_last'])
    recom_df=upsell_crosssell_result(X_last,y_pred,purchased_cat,cat_list)
#     recom_df = recom_df.set_index('customer_No')
    return recom_df,sales_txt,model_stat
