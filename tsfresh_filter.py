# This File contains some functions created to make a more deeper filtering on Tsfresh on the Feature Engineering Notebook. Maybe they will be used again in the future.

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