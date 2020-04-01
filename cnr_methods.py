import numpy as np 
import pandas as pd 

def get_preprocessed_data():
    x_train = pd.read_csv('X_train.csv')
    x_test = pd.read_csv('X_test.csv')

    nwps23 = ['NWP2_00h_D-2_U', 'NWP2_00h_D-2_V',
       'NWP2_12h_D-2_U', 'NWP2_12h_D-2_V', 'NWP2_00h_D-1_U',
       'NWP2_00h_D-1_V', 'NWP2_12h_D-1_U', 'NWP2_12h_D-1_V',
       'NWP2_00h_D_U', 'NWP2_00h_D_V', 'NWP2_12h_D_U', 'NWP2_12h_D_V',
       'NWP3_00h_D-2_U', 'NWP3_00h_D-2_V', 'NWP3_00h_D-2_T',
       'NWP3_06h_D-2_U', 'NWP3_06h_D-2_V', 'NWP3_06h_D-2_T',
       'NWP3_12h_D-2_U', 'NWP3_12h_D-2_V', 'NWP3_12h_D-2_T',
       'NWP3_18h_D-2_U', 'NWP3_18h_D-2_V', 'NWP3_18h_D-2_T',
       'NWP3_00h_D-1_U', 'NWP3_00h_D-1_V', 'NWP3_00h_D-1_T',
       'NWP3_06h_D-1_U', 'NWP3_06h_D-1_V', 'NWP3_06h_D-1_T',
       'NWP3_12h_D-1_U', 'NWP3_12h_D-1_V', 'NWP3_12h_D-1_T',
       'NWP3_18h_D-1_U', 'NWP3_18h_D-1_V', 'NWP3_18h_D-1_T',
       'NWP3_00h_D_U', 'NWP3_00h_D_V', 'NWP3_00h_D_T', 'NWP3_06h_D_U',
       'NWP3_06h_D_V', 'NWP3_06h_D_T', 'NWP3_12h_D_U', 'NWP3_12h_D_V',
       'NWP3_12h_D_T', 'NWP3_18h_D_U', 'NWP3_18h_D_V', 'NWP3_18h_D_T']

    x_train[nwps23] = x_train[nwps23].interpolate(method = 'spline',order=3)
    x_test[nwps23] = x_train[nwps23].interpolate(method = 'spline',order=3)

    x_train['Set'] = 'Train'
    x_test['Set'] = 'Test'
    full_data = pd.concat([x_train,x_test])

    U_100m = []
    V_100m = []
    U_10m = []
    V_10m = []
    T = []
    CLCT = []

    for column in full_data.columns:
        if (column.__contains__('U')) and (column.__contains__('NWP1') or column.__contains__('NWP2') or column.__contains__('NWP3')):
            U_100m.append(column)
        elif (column.__contains__('V')) and (column.__contains__('NWP1') or column.__contains__('NWP2') or column.__contains__('NWP3')):
            V_100m.append(column)
        elif (column.__contains__('U')) and (column.__contains__('NWP4')):
            U_10m.append(column)
        elif (column.__contains__('V')) and (column.__contains__('NWP4')):
            V_10m.append(column)
        elif (column.__contains__('_T')):
            T.append(column)
        elif (column.__contains__('CLCT')):
            CLCT.append(column)

    full_data['U_100m'] = full_data[U_100m].mean(axis=1)
    full_data['V_100m'] = full_data[V_100m].mean(axis=1)
    full_data['U_10m'] = full_data[U_10m].mean(axis=1)
    full_data['V_10m'] = full_data[V_10m].mean(axis=1)
    full_data['T'] = full_data[T].mean(axis=1)
    full_data['CLCT'] = full_data[CLCT].mean(axis=1)

    return full_data

def get_simplified_data():
    simple_data = get_preprocessed_data()
    simple_data = simple_data[['ID','Time','WF','U_100m','V_100m','U_10m','V_10m','T','CLCT','Set']]
    return simple_data