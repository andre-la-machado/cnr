import numpy as np
import cupy as cp
import pandas as pd 
import datetime
import xgboost as xgb
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from sklearn.model_selection import TimeSeriesSplit, train_test_split

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

    full_data['U_100m'] = full_data[U_100m].mean(axis=1)
    full_data['V_100m'] = full_data[V_100m].mean(axis=1)
    full_data['U_10m'] = full_data[U_10m].mean(axis=1)
    full_data['V_10m'] = full_data[V_10m].mean(axis=1)
    full_data['T'] = full_data[T].mean(axis=1)
    full_data['CLCT'] = full_data[CLCT].mean(axis=1)

    full_data['CLCT'] = full_data['CLCT'].apply(lambda x: 0 if x < 0 else x)

    return full_data

# Função de Geração dos Dados Simplificados
def get_simplified_data():
    X = get_preprocessed_data()
    X = X[['ID','Time','WF','U_100m','V_100m','U_10m','V_10m','T','CLCT','Set']]
    X['Time'] = pd.to_datetime(X['Time'],dayfirst=True)
    X = X.set_index('Time')

    y = pd.read_csv('Data/Y_train.csv')
    y = y.set_index('ID')
    return X,y

# Função de Transformação dos Dados (Estabilização de Variância)
def transform_data(df):
    for column in df.columns:
        df[column] = np.diff(df[column],prepend=df[column].iloc[0])
    return df

# Função de Transformação Reversa dos Dados 
def revert_data(preds):
    reverted_data = np.cumsum(preds)
    return reverted_data

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

def lofo_objective(X,y,features,feature_out,param):
    #Internal Parameters
    k_fold_splits = 5
    num_boost_round = 100
    early_stopping_rounds = 10

    # Define Time Split Cross Validation
    tscv = TimeSeriesSplit(n_splits=k_fold_splits)

    test_scores = np.empty(0)
    for train_index, test_index in tscv.split(X):
        # Get the Data of the Split
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Separating Training Set of Split on Train and Validation Subsets
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.143, shuffle=False)

        dtrain = lofo_df(X_train,y_train['Production'],features,feature_out)
        dval = lofo_df(X_val,y_val['Production'],features,feature_out)
        dtest = lofo_df(X_test,y_test['Production'],features,feature_out)

        # Train the Model
        watchlist = [(dtrain,'train'),(dval,'eval')]
        bst = xgb.train(param, dtrain, num_boost_round=num_boost_round, evals=watchlist, feval=metric_cnr,early_stopping_rounds=early_stopping_rounds,verbose_eval=False)
        preds = bst.predict(dtest,ntree_limit=bst.best_ntree_limit)
        score = metric_cnr(preds,dtest)
        test_scores = np.append(test_scores,score[1])

    return test_scores.mean()

def LOFO_GPU_Importance(X,y,features,param):
    X = transform_data(X[features])
    base_score = lofo_objective(X,y,features,None,param)
    scores = np.empty(0)
    i = 0
    for feature_out in features:
        i = i + 1
        start_time = datetime.datetime.now()
        feature_score = lofo_objective(X,y,features,feature_out,param)
        scores = np.append(scores,base_score-feature_score)
        end_time = datetime.datetime.now()
        delta = end_time - start_time
        print('{}/{} {}.{} s/it'.format(i,len(features),delta.seconds,delta.microseconds))
    importance_df = pd.DataFrame()
    importance_df["feature"] = features
    importance_df["score"] = scores
    return importance_df.sort_values(by='score',ascending=True)

    # Funções para Filtragem de Variáveis

def get_selected_features(n_features):
    selected_features = pd.read_csv(r'C:\Users\andre_\OneDrive\Documentos\Feature Selection\Importance_WF1.csv')
    if n_features == 0:
        selected_features = list(selected_features['feature'].values)
    else:
        selected_features = list(selected_features[:n_features]['feature'].values)

    base_features = ['ID','Time','WF','U_100m','V_100m','U_10m','V_10m','T','CLCT','Set']

    selected_features = base_features + selected_features

    # Dropping Duplicates
    features = [] 
    [features.append(x) for x in selected_features if x not in features]
    feature_data = pd.read_csv(r'C:\Users\andre_\OneDrive\Documentos\Feature Selection\Selected_Features_Data.csv')
    feature_data = feature_data[features]
    return feature_data

