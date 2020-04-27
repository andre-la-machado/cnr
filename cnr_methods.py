import numpy as np
import cupy as cp
import pandas as pd 
import datetime
import xgboost as xgb

# Função de Processamento dos Dados
def get_preprocessed_data():
    x_train = pd.read_csv('Data/X_train.csv')
    x_test = pd.read_csv('Data/X_test.csv')

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

    full_data['U_100m'] = full_data[U_100m].median(axis=1)
    full_data['V_100m'] = full_data[V_100m].median(axis=1)
    full_data['U_10m'] = full_data[U_10m].median(axis=1)
    full_data['V_10m'] = full_data[V_10m].median(axis=1)
    full_data['T'] = full_data[T].median(axis=1)
    full_data['CLCT'] = full_data[CLCT].median(axis=1)

    full_data['CLCT'] = full_data['CLCT'].apply(lambda x: 0 if x < 0 else x)

    return full_data

# Função de Geração dos Dados Simplificados
def get_simplified_data():
    simple_data = get_preprocessed_data()
    simple_data = simple_data[['ID','Time','WF','U_100m','V_100m','U_10m','V_10m','T','CLCT','Set']]
    return simple_data

# Função de Transformação dos Dados (Estabilização de Variância)
def transform_data(df):
    for column in df.columns:
        df[column] = np.diff(df[column],prepend=df[column].iloc[0])
    return df

# Função de Transformação Reversa dos Dados 
def revert_data(df):
    for column in df.columns:
        df[column] = np.cumsum(df[column])
    return df

# Função de Cálculo da Métrica da Competição
def metric_cnr(preds,dtrain):
    labels = dtrain.get_label()
    cape_cnr = 100*np.sum(np.abs(preds-labels))/np.sum(labels)
    return 'CAPE', cape_cnr

# Funções para Implementação do LOFO em GPU

def lofo_df(df,y,features,feature_out):
    if feature_out is not None:
        df = df.drop(feature_out,axis=1)
    gpu_matrix = cp.asarray(df[[feature for feature in features if feature != feature_out]])
    gpu_matrix = xgb.DMatrix(gpu_matrix,label=y)
    return gpu_matrix

def lofo_score(X,y,features,feature_out,model):
    feature_df = lofo_df(X,y,features,feature_out=feature_out)
    model_params = model.get_xgb_params()
    lofo_score = xgb.cv(params=model_params,dtrain=feature_df)
    return lofo_score.iloc[:, [2]].mean()

def LOFO_GPU_Importance(X,y,features,model):
    base_score = lofo_score(X,y,features,None,model)
    scores = np.empty(0)
    i = 0
    for feature_out in features:
        i = i + 1
        start_time = datetime.datetime.now()
        feature_score = lofo_score(X,y,features,feature_out,model)
        scores = np.append(scores,base_score-feature_score)
        end_time = datetime.datetime.now()
        delta = end_time - start_time
        print('{}/{} {}.{} s/it'.format(i,len(features),delta.seconds,delta.microseconds))
    importance_df = pd.DataFrame()
    importance_df["feature"] = features
    importance_df["score"] = scores
    return importance_df.sort_values(by='score',ascending=True)