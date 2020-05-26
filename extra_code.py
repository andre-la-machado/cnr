# This File contains some functions created to make a more deeper filtering on Tsfresh on the Feature Engineering Notebook. Maybe they will be used again in the future.

# Tsfresh Feature Creation

data = data[['ID','WF','U_100m','V_100m','U_10m','V_10m','T','CLCT','Set']]

tsfresh_data = pd.DataFrame()
for variable in ['U_100m','V_100m','U_10m','V_10m','T','CLCT']: 
    df_shift, y = make_forecasting_frame(data[variable],kind=variable,max_timeshift=20,rolling_direction=1)
    X = extract_features(df_shift, column_id="id", column_sort="time", column_value="value", impute_function=impute,show_warnings=False,n_jobs=3)
    X['Feature'] = variable
    tsfresh_data = tsfresh_data.append(X)

tsfresh_data = tsfresh_data.pivot(columns='Feature')

tsfresh_data.columns = tsfresh_data.columns.map('{0[0]}|{0[1]}'.format)

tsfresh_data = tsfresh_data.loc[:, tsfresh_data.apply(pd.Series.nunique) != 1]

# Tsfresh Feature Selection
U_100m = []
U_10m = []
V_100m = []
V_10m = []
T = []
CLCT = []


for column in tsfresh_data.columns:
    if column.__contains__('|U_100m'):
        U_100m.append(column)
    elif column.__contains__('|U_10m'):
        U_10m.append(column)
    elif column.__contains__('|V_100m'):
        V_100m.append(column)
    elif column.__contains__('|V_10m'):
        V_10m.append(column)
    elif (column.__contains__('|T')):
        T.append(column)
    elif (column.__contains__('|CLCT')):
        CLCT.append(column)


tsfresh_features_filtered = []
for feature in [U_100m,U_10m,V_100m,V_10m,T,CLCT]:
    relevance_table = calculate_relevance_table(tsfresh_data[feature],y,n_jobs=3)
    tsfresh_features_filtered.append(relevance_table[relevance_table['relevant']==True].sort_values('p_value')['feature'].values[:10]) 
tsfresh_features_filtered = np.concatenate(tsfresh_features_filtered)


# LOFO Feature Selection Code

from lofo import LOFOImportance, Dataset, plot_importance
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import make_scorer, mean_absolute_error

final_features = feature_data.merge(tsfresh_data,left_on=feature_data.index,right_on=tsfresh_data.index,how='left')

final_features = final_features.merge(y_train,on='ID',how='left')

final_features = final_features.rename({'key_0':'Date'},axis=1)

final_features = final_features.fillna(0)

cv = TimeSeriesSplit()

scorer = make_scorer(mean_absolute_error, greater_is_better=False)

features = final_features.drop(['Date','ID','WF','Set'],axis=1).columns.values

dataset = Dataset(df=final_features, target="Production", features=features)

lofo_imp = LOFOImportance(dataset, cv=cv, scoring=scorer)

importance_df = lofo_imp.get_importance()

plot_importance(importance_df, figsize=(12, 20))

