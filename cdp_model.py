# Request section to import another module
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
#

import pandas as pd
import numpy as np
import datetime as dt
import statistics
import math
from itertools import repeat
from collections import Counter

def extract_json(df,json_col_name,selected_key=[]):
    if selected_key !=[]:
        for k in selected_key:
            df[k] = df[json_col_name].apply(lambda x: x[k] if x not in [None,'NULL'] else np.nan)
    return df

###### Standard model ######

def cut_outlier(in_list,cut_point=None):
    if cut_point!=None:
        in_list = [i for i in in_list if i > cut_point]
    q1, q3= np.nanpercentile(in_list,[25,75])
    iqr = q3 - q1
    lower_bound = q1 -(1.5 * iqr) 
    upper_bound = q3 +(1.5 * iqr) 
    raw = np.array(in_list)
    arr = raw[(raw>=lower_bound)&(raw<=upper_bound)]
    criteria= np.nanpercentile(arr,[20,40,60,80])
    mn = min(arr)
    mx = max(arr)
    if raw[raw>upper_bound].size > 0:
        u = min(raw[raw>upper_bound])
    else: 
        u = np.nan
    if raw[raw<lower_bound].size > 0: 
        l= max(raw[raw<lower_bound])
    else: 
        l = np.nan
    stat = [mn,mx,l,u]
    return arr,stat,criteria

def list_to_score(a,stat,criteria,threshold):
    score_list = []
    for i in [1,2,3]:
        if a >= criteria[i-1] and a <= criteria[i]:
            score_list.append(i+1)

    if a< criteria[0]:
        score_list = [1]
        
    if a> criteria[3]:
        score_list = [5]

    s = int(statistics.mean(score_list))

    if stat[2] is np.nan  and stat[3] is not np.nan:
        if ((stat[3]-stat[1])/stat[3])<threshold:
            s = min(score_list)

    if stat[2] is not np.nan and stat[3]is np.nan:
        if ((stat[2]-stat[0])/stat[2])> threshold:
            s = max(score_list)

    return s

def segment(r_score,fxm_score):
    if  r_score >= 4 and fxm_score >= 3:
        Segment = "Champion"
    elif  2 <= r_score <= 4 and fxm_score >= 3:
        Segment = "Loyal Customers"
    elif  r_score <= 2 and fxm_score >= 4 :
        Segment = "Should not Lose"
    elif  r_score <= 2 and 2 <= fxm_score <= 4:
        Segment = "At Risk"
    elif  2 <= r_score <= 3 and 2 <= fxm_score <= 3:
        Segment = "Need Attention"
    elif  2 <= r_score <= 3 and fxm_score <=2:
        Segment = "About to Sleep"
    elif r_score >= 3 and 1 <= fxm_score <=3:
        Segment = "Potential Loyalists"
    elif 3 <= r_score <= 4 and fxm_score <= 1:
        Segment = "Promising"
    elif 4 <= r_score <= 5 and fxm_score <= 1:
        Segment = "New Customers"
    else:
        Segment = "Lost"
    return Segment

def get_rfm(user_data,cut_point_dict,threshold=0.2):
    """sgmt_day_from_last_bill --> datediff beetween today and last_order_date
    sgmt_frequency --> count transaction / count month
    sgmt_basket_size --> total sales / count transaction"""
    
    rfm_data = user_data[['lead_id','sgmt_day_from_last_bill','sgmt_frequency','sgmt_basket_size']]
    data = (rfm_data['sgmt_day_from_last_bill']*-1).to_list() # convert value cuz day_from_last_bill more is not good
    arr,stat,criteria = cut_outlier(data,cut_point=cut_point_dict['r'])
    rfm_data['r_score'] = list(map(list_to_score,data,repeat(stat),repeat(criteria),repeat(threshold)))

    data = rfm_data['sgmt_frequency'].to_list()
    arr,stat,criteria = cut_outlier(data)
    rfm_data['f_score'] = list(map(list_to_score,data,repeat(stat),repeat(criteria),repeat(threshold)))

    data = rfm_data['sgmt_basket_size'].to_list()
    arr,stat,criteria = cut_outlier(data)
    rfm_data['m_score'] = list(map(list_to_score,data,repeat(stat),repeat(criteria),repeat(threshold)))
    rfm_data['fxm_score'] = (rfm_data['f_score']*rfm_data['m_score'])/5
        
    rfm_data['segment'] = list(map(segment,rfm_data['r_score'].to_list(),rfm_data['fxm_score'].to_list()))
    return rfm_data

def cal_opportunity(s):
    s = cut_outlier(s)[0]
    per_80 = np.percentile(s, 80)
    avg = np.mean(s[s > per_80])
    if math.isnan(avg):
        avg = np.mean(s)
    return avg

def getModel_opportunity(ClvRfm):
    sgmt_opportunity = ClvRfm.groupby('sgmt_rfm').agg({'sgmt_clv':[cal_opportunity]})
    sgmt_opportunity.columns = ['sgmt_opportunity']
    sgmt_opportunity = sgmt_opportunity.reset_index()
    sgmt_opportunity = ClvRfm.merge(sgmt_opportunity,on=['sgmt_rfm'],how='left')
    sgmt_opportunity['sgmt_opportunity'] = np.where(sgmt_opportunity['sgmt_clv'] > sgmt_opportunity['sgmt_opportunity'], sgmt_opportunity['sgmt_clv'], sgmt_opportunity['sgmt_opportunity'])
    sgmt_opportunity['sgmt_opportunity'] = sgmt_opportunity['sgmt_opportunity'].round(2)
    sgmt_opportunity = sgmt_opportunity[['lead_id','sgmt_opportunity']]
    return sgmt_opportunity

def get_clc(order):
    # get_cus_first_date
    get_cus_first_date = order.copy()
#     get_cus_first_date = extract_json(get_cus_first_date,'customer_hierarchy',selected_key=['cus_startdate_date'])
    get_cus_first_date = get_cus_first_date.groupby('lead_id').agg(cus_created_date = ('cus_startdate', 'min')).reset_index()

    # get_cus_last_tx
    date_interval = dt.datetime.today().date() - dt.timedelta(days=90)
    date_interval = dt.datetime.combine(date_interval, dt.datetime.min.time())
    order['order_date'] = pd.to_datetime(order['order_date'])
    order_last_tx = order[order['order_date'] >= date_interval]
    get_cus_last_tx = order_last_tx.groupby('lead_id').agg(last_tx_time_stamp = ('order_date', 'max')).reset_index()

    # get_order_sales
    get_order_sales = order.groupby('order_id').agg(sales = ('ttl_net_sales', 'sum')).reset_index()
    # get_order_sales = order[['order_id', 'ttl_net_sales']]

    # order_seq
    order_seq = order.copy()
    order_seq = order_seq.sort_values(by = ['lead_id', 'order_date'])
    order_seq['prev_order_id'] = order_seq.groupby(['lead_id'])['order_id'].shift(1)
    order_seq['prev_order_date'] = order_seq.groupby(['lead_id'])['order_date'].shift(1)
    order_seq = order_seq[['order_id','order_date','lead_id','prev_order_id','prev_order_date']]

    # clc
    get_cus_date = get_cus_first_date.merge(get_cus_last_tx, how = 'left', on = 'lead_id')
    clc = get_cus_date.merge(order_seq[order_seq['lead_id'].notna()], how = 'left', left_on = ['lead_id','last_tx_time_stamp'], right_on = ['lead_id','order_date'])                                                                               
    clc = clc.merge(get_order_sales, how = 'left', on = 'order_id').rename(columns = {'sales': 'cur_sales'})
    clc = clc.merge(get_order_sales, how = 'left', left_on = 'prev_order_id', right_on = 'order_id').rename(columns = {'order_id_x': 'order_id','sales': 'prev_sales'})
    clc['time_stamp_ymd'] = clc['order_date'].dt.strftime('%Y-%m-%d')
    clc['cus_created_date'] = pd.to_datetime(clc['cus_created_date']).dt.strftime('%Y-%m-%d')
    clc['sgmt_customer_lifecycle'] = np.where((clc['cur_sales'] >= clc['prev_sales']) & (clc['prev_sales'].notna()), 'increase',
                            np.where((clc['cur_sales'] < clc['prev_sales']) & (clc['prev_sales'].notna()), 'decrease',
                            np.where(clc['cus_created_date'] < clc['time_stamp_ymd'], 'repurchase',
                            np.where(clc['cus_created_date'] == clc['time_stamp_ymd'], 'first_purchase', 'no transaction'))))
    clc = clc.groupby('lead_id').agg(sgmt_customer_lifecycle=('sgmt_customer_lifecycle', lambda x: x.value_counts().index[0])).reset_index()
    return clc

def fav_store(df):
    fav_store = df.groupby(['lead_id','store_name']).agg(store_order=('order_id','nunique'),store_sales=('ttl_net_sales','sum')).reset_index()
    max_order = fav_store.groupby(['lead_id']).agg(max_order=('store_order','max'),max_sales=('store_sales','max')).reset_index()
    cus = df.groupby(['lead_id']).agg(ttl_order=('order_id','nunique'),store_count=('store_name','nunique')).reset_index()
    fav_store = fav_store.merge(cus,how='left',left_on=['lead_id'],right_on=['lead_id'])
    fav_store = fav_store.merge(max_order,how='left',left_on=['lead_id'],right_on=['lead_id'])
    fav_store['order_ratio'] = fav_store['store_order']/fav_store['ttl_order']
    fav_store['store_count'].fillna(0,inplace=True)
    fav_store['criteria'] = 1/fav_store['store_count']
    fav_store = fav_store[(fav_store['criteria']<=fav_store['order_ratio']) & (fav_store['store_order']==fav_store['max_order'])]
#     fav_store = fav_store[(fav_store['criteria']<=fav_store['order_ratio']) & (fav_store['store_order']==fav_store['max_order']) &(fav_store['store_sales']==fav_store['max_sales'])]
    fav_store = fav_store.groupby(['lead_id']).agg(sgmt_fav_store = ('store_name',lambda x:'|'.join(x.unique()))).reset_index()
    return fav_store

def get_customer_drive(product_tx,product_discount,new_threshold=0.2,discount_threshold=0.4):
    # prep data
    # product_info
    product_info = product_tx.groupby('product_id').agg(new_start = ('order_date', 'min')).reset_index()
    product_info['new_end'] = product_info['new_start'] + dt.timedelta(days=30)

    # pro
    pro = product_tx.merge(product_info, how = 'left', on = 'product_id')
    pro = pro[pro['lead_id'].notna()]
    pro['new_product'] = np.where((pro['order_date'] >= pro['new_start']) & (pro['order_date'] <= pro['new_end']), pro['quantity'], 0)
    pro = pro.groupby('lead_id').agg(quantity = ('quantity','sum'), new_product = ('new_product', 'sum')).reset_index()

    # cus_disc
    dis = product_discount.copy()
    dis['flag_discount'] = 1
    pro_tx = product_tx.copy()
    pro_tx['flag_discount'] = 0
    dis = pd.concat([pro_tx,dis])
    dis = dis.groupby('order_id').agg(lead_id = ('lead_id','min'), flag_discount = ('flag_discount','max')).reset_index()
    cus_disc = dis.groupby('lead_id').agg(count_tx = ('order_id', 'nunique'), count_discount_tx = ('flag_discount','sum')).reset_index()
    drive = pro.merge(cus_disc, how = 'left', on = 'lead_id')
    
    # cal drive
    drive = drive.fillna(0)
    drive[['quantity', 'new_product', 'count_tx', 'count_discount_tx']] = drive[['quantity', 'new_product', 'count_tx', 'count_discount_tx']].astype(float)
    
    drive['flag_new'] = np.where((drive['new_product']/drive['quantity'])>=new_threshold,"New Product",'')
    drive['flag_promo'] = np.where((drive['count_discount_tx']/drive['count_tx'])>=discount_threshold,'Promotion','')
    drive['sgmt_drive_type'] = np.where((drive['flag_new'] != '') & (drive['flag_promo'] != ''), drive['flag_new'] + '|' + drive['flag_promo'], 
         np.where(drive['flag_new'] == '', drive['flag_promo'], drive['flag_new']))
    drive['sgmt_drive_type'] = drive['sgmt_drive_type'].replace('', np.nan)
    return drive[['lead_id', 'sgmt_drive_type']]

# def standard_model(vault_path,bucket,key,filename):
    
#     multi_layer_data = read_tablefile(bucket,key+filename)
#     multi_layer_data = extract_json(multi_layer_data,'customer_hierarchy',['source'])
#     multi_layer_data = multi_layer_data[multi_layer_data['source']=='member']
#     order = multi_layer_data[multi_layer_data['layer'] == 'order'].reset_index(drop = True)
#     order_tx = multi_layer_data[multi_layer_data['layer'] == 'order_tx'].reset_index(drop = True)
#     product_tx = multi_layer_data[multi_layer_data['layer'] == 'product_tx'].reset_index(drop = True)
#     product_discount = multi_layer_data[multi_layer_data['layer'] == 'product_discount'].reset_index(drop = True)
    
#     order['year_month'] = pd.to_datetime(order['order_date'], format='%Y-%m-%d').dt.strftime("%Y/%m")
  
#     #id aggregation
#     id_agg = order.groupby('lead_id').agg(last_order_dt = ('order_date', max),
#                                                                          sgmt_clv = ('ttl_net_sales', sum),
#                                                                          ttl_product_discount = ('ttl_product_discount',sum),
#                                                                          ttl_order_discount = ('ttl_order_discount',sum),
#                                                                          count_tx = ('order_id', 'nunique'),
#                                                                          count_month = ('year_month', 'nunique')
#                                                                         ).reset_index()
    
#     snapshot = pdReadSQL(vault_path,"""select lead_id,clv,count_order,count_month,latest_order_dt from customer_snapshot c left join lead_mapping m on c.customer_code = m.customer_code""",optimize=False)
#     snapshot = snapshot.rename(columns={'clv':'sgmt_clv','count_order':'count_tx','latest_order_dt':'last_order_dt'})
#     # r_cut_point = (dt.date.today() - pd.to_datetime(snapshot['last_order_dt']).dt.date).astype('timedelta64[D]').astype(int).min()*-1
#     r_cut_point =(dt.date.today() - pd.to_datetime(snapshot['last_order_dt']).dt.date).min().days*-1
#     cut_point_dict = {}
#     cut_point_dict['r'] = r_cut_point
    
    
#     id_agg = pd.concat([id_agg,snapshot]).reset_index(drop=True)
#     id_agg['last_order_dt'] = pd.to_datetime(id_agg['last_order_dt'])
#     id_agg = id_agg.groupby(['lead_id']).agg(last_order_dt = ('last_order_dt', max),
#                                                                          sgmt_clv = ('sgmt_clv', sum),
#                                                                          ttl_product_discount = ('ttl_product_discount',sum),
#                                                                          ttl_order_discount = ('ttl_order_discount',sum),
#                                                                          count_tx = ('count_tx', sum),
#                                                                          count_month = ('count_month', 'sum')
#                                                                         ).reset_index()
    
#     # id_agg['sgmt_day_from_last_bill'] = (dt.date.today() - pd.to_datetime(id_agg['last_order_dt']).dt.date).astype('timedelta64[D]').astype(int)
#     id_agg['sgmt_day_from_last_bill'] =pd.to_timedelta(dt.date.today() - pd.to_datetime(id_agg['last_order_dt']).dt.date).dt.days
#     id_agg['sgmt_frequency'] = id_agg['count_tx']/id_agg['count_month']
#     id_agg['sgmt_basket_size'] = id_agg['sgmt_clv']/id_agg['count_tx']
#     id_agg['total_discount'] = id_agg['ttl_product_discount']+id_agg['ttl_order_discount']
    
#     # spending_tier
#     data = id_agg['sgmt_basket_size'].tolist()
#     data = cut_outlier(data)[0]
#     m_cut,h_cut = np.percentile(data,[40,80])
#     id_agg['sgmt_spending_tier'] = np.where(id_agg['sgmt_basket_size']>=h_cut,"high",np.where(id_agg['sgmt_basket_size']>=m_cut,"mid","low"))

#     # rfm
#     id_agg['sgmt_rfm'] = get_rfm(id_agg,cut_point_dict)['segment']
#     id_agg = id_agg[['lead_id', 'sgmt_clv', 'sgmt_rfm','sgmt_basket_size','sgmt_frequency',
#                             'last_order_dt','total_discount']]
# #     id_agg = id_agg.drop(columns=['last_order_dt','ttl_order_discount','count_tx'])

#     # opportunity
#     opty = getModel_opportunity(id_agg[['lead_id','sgmt_clv','sgmt_rfm']])
#     id_agg = id_agg.merge(opty,how='left', on = 'lead_id')

#     # favorite_store
#     store = fav_store(order)
#     id_agg = id_agg.merge(store, how = 'left', on = 'lead_id')

#     # clc
#     clc = get_clc(order)
#     id_agg = id_agg.merge(clc,how='left',on = 'lead_id')

#     # drive type
#     drive = get_customer_drive(product_tx,product_discount)
#     id_agg = id_agg.merge(drive, how = 'left', on = 'lead_id')
    
# #     if lead_mode:
# #         id_agg = id_agg.rename(columns={'customer_uid':'lead_id'})
    
#     return id_agg 
