# %%
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
sns.set_style('darkgrid')
from cnr_methods import get_selected_features, transform_data, revert_data, get_simplified_data
import random

from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.preprocessing import MinMaxScaler


# %%
import tensorflow as tf
print(tf.test.is_built_with_cuda())  # Sanity check: GPU available to tf or not
print(tf.test.is_built_with_gpu_support())
print(tf.config.list_physical_devices('GPU'))
print(tf.__version__)  # Check if __version__>="2.0.0"
print(tf.keras.__version__)


# ## Read Data
# Here, the data used correspond to the results of the Feature Engineering and Selection Step. For simplicity, during Hyperparameter Optimization, only Wind Farm 3 Training Data is used.

# %%
full_data = pd.read_csv(r"C:\Users\andre_\OneDrive\Documentos\Feature Selection\Selected_Features_Data.csv")

#full_data = full_data.rename({'Unnamed: 0' : 'Time'},axis=1)
full_data = full_data.set_index('Time')

full_label = pd.read_csv(r'C:\Users\andre_\OneDrive\Documentos\GitHub\cnr\Data\Y_train.csv')


# %%
full_data = full_data[['ID', 'WF', 'U_100m', 'V_100m', 'U_10m', 'V_10m', 'T', 'CLCT', 'Set',
       'Wind Speed 100m', 'Wind Direction 100m', 'Wind Speed 10m',
       'Wind Direction 10m','Month_Number']]


# %%
X = full_data[full_data['Set']=='Train']

WF = 'WF3'
X = X[X['WF']==WF]
y = full_label[full_label['ID'].isin(X['ID'])]


# %%
X.head()


# ## Scaling Data

# For a better performance of the Network, here functions are created to scale data between [-1,1] using MinMaxScaler. For the Direction Data, presented in degrees, Sin and Cos are calculated, which naturally have values in this same scale.

# %%
def preprocessing_X(X):

  scaler = MinMaxScaler(feature_range=(-1,1))

  X_saved_columns = X[['ID','WF','Set','Month_Number']]
  X_saved_columns = X_saved_columns.reset_index().drop('Time',axis=1)
  X = X.drop(['ID','WF','Set','Month_Number'],axis=1)

  # Fill NaN's
  #X = X.fillna(method="ffill", axis=1) # ZOH
  #X = X.fillna(0)

  # Scaling Data
  directions = X[['Wind Direction 100m', 'Wind Direction 10m']]
  directions["Sin_Wind Direction 100m"] = np.sin(X['Wind Direction 100m']*(np.pi/180))
  directions["Cos_Wind Direction 100m"] = np.cos(X['Wind Direction 100m']*(np.pi/180))
  directions["Sin_Wind Direction 10m"] = np.sin(X['Wind Direction 10m']*(np.pi/180))
  directions["Cos_Wind Direction 10m"] = np.cos(X['Wind Direction 10m']*(np.pi/180))
  directions = directions.drop(['Wind Direction 100m', 'Wind Direction 10m'],axis=1)
  directions = directions.reset_index().drop('Time',axis=1)

  X = X.drop(['Wind Direction 100m', 'Wind Direction 10m'],axis=1)
  X_columns = X.columns

  X = scaler.fit_transform(X)
  X = pd.DataFrame(X,columns=X_columns)
  X = pd.concat([X,directions],axis=1)
  X = pd.concat([X,X_saved_columns],axis=1)

  X = X.fillna(-2)

  return X


# %%
def preprocessing_y(y):

  scaler = MinMaxScaler(feature_range=(-1,1))

  #y = y.fillna(method="ffill", axis=1) # ZOH
  #y = y.fillna(0)
  y = y.drop('ID',axis=1)
  y = scaler.fit_transform(y)
  y = pd.DataFrame(y)
  y = y.fillna(-2)

  return y,scaler


# %%
y.shape


# ## Subsets Creation

# Here, the Data is converted to a group of subsets where each subset has n_steps of past data.

# %%
n_steps = 60


# %%
# split a multivariate sequence into samples
def split_sequences(X, y = None, n_steps = 1):
	sample_X, sample_y = list(), list()
	for i in range(len(X)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the dataset
		if end_ix > len(X):
			break
		# gather input and output parts of the pattern
		seq_x = X[i:end_ix, :]
		sample_X.append(seq_x)
		if y is not None:
			seq_y = y[end_ix-1,-1]
			sample_y.append(seq_y)
	return np.array(sample_X), np.array(sample_y)


# %%
def shift_save(df,n_steps):
  empty = pd.DataFrame(np.zeros((n_steps-1,df.shape[1])),columns=df.columns)
  df = pd.concat([empty,df])
  return df


# %%
sample_X,sample_y = split_sequences(X.values,y.values,n_steps)


# %%
n_features = sample_X.shape[2] - 2


# ## Model

# Here, a function to create the Model usin Keras is defined.

# %%
def LSTM_Model(input_shape, batch_size=1):
  # Numerical branch

  input_layer = tf.keras.Input(shape = input_shape,batch_size = batch_size)

  hidden_1 = tf.keras.layers.LSTM(units=128,return_sequences=True,stateful=True)(input_layer)
  hidden_1 = tf.keras.layers.Dropout(0.4)(hidden_1)

  hidden_2 = tf.keras.layers.LSTM(units=87,return_sequences=True,stateful=True)(hidden_1)
  hidden_2 = tf.keras.layers.Dropout(0.4)(hidden_2)

  hidden_3 = tf.keras.layers.LSTM(units=57,stateful=True)(hidden_2)
  hidden_3 = tf.keras.layers.Dropout(0.4)(hidden_3)

  # Output
  #outputs = tf.keras.layers.PReLU()(hidden_2)
  #outputs = tf.keras.layers.Dropout(rate=0.2)(hidden_2)
  outputs = tf.keras.layers.Dense(units=1)(hidden_3)

  model = tf.keras.Model(inputs=input_layer, outputs=outputs)

  return model


# %%
input_shape = (n_steps,n_features)


# %%
model = LSTM_Model(input_shape)


# %%
model.summary()


# ## Validation

# Here, a Validation Process is done using the SKlearn's TimeSeriesSplit with 5 folds. The results of RMSE for each Split on Train, Validation and Test sets are plotted in the end.

# %%
random.seed(317)
tf.random.set_seed(317)

patience = 3
epochs = 10
k_fold_splits = 5
total_it = 50
monitor = "root_mean_squared_error"
batch_size = 1


# %%
# Define Time Split Cross Validation
tscv = TimeSeriesSplit(n_splits=k_fold_splits)

# Separating Data from Hold Out Set

X_cv, _, y_cv, _ = train_test_split(X, y, test_size=0.125, shuffle=False)

train_scores = np.empty(0)
val_scores = np.empty(0)
test_scores = np.empty(0)
for train_index, test_index in tscv.split(X_cv):

    #train_index = train_index[-n_rows:]
    #test_index = test_index[-n_rows:]

    # Get the Data of the Split
    X_train, X_test = X_cv.iloc[train_index], X_cv.iloc[test_index]
    y_train, y_test = y_cv.iloc[train_index], y_cv.iloc[test_index]

    # Separating Training Set of Split on Train and Validation Subsets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.143, shuffle=False)

    # Preprocessing Data
    X_train = preprocessing_X(X_train)
    X_val = preprocessing_X(X_val)
    X_test = preprocessing_X(X_test)

    X_train = X_train.drop(['ID','WF','Set','Month_Number'],axis=1)
    X_val = X_val.drop(['ID','WF','Set','Month_Number'],axis=1)
    X_test = X_test.drop(['ID','WF','Set','Month_Number'],axis=1)

    y_train,_ = preprocessing_y(y_train)
    y_val,_ = preprocessing_y(y_val)
    y_test,_ = preprocessing_y(y_test)

    # Reshape Data
    X_train, y_train = split_sequences(X_train.values,y_train.values,n_steps)
    X_val, y_val = split_sequences(X_val.values,y_val.values,n_steps)
    X_test, y_test = split_sequences(X_test.values,y_test.values,n_steps)

    # Create Model
    model = LSTM_Model(input_shape,batch_size=batch_size)

    # Callbacks
    callbacks_list = [tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=patience, verbose=0, min_delta=1e-8)]

    # Train the Model
    model.compile(optimizer='adam', loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError()])
    history = model.fit(x = X_train, y = y_train, batch_size = batch_size, epochs = epochs, validation_data = (X_val, y_val), callbacks=callbacks_list,shuffle=False)

    # Train and Validation Score
    train_score = np.array(history.history['root_mean_squared_error']).mean()
    val_score = np.array(history.history['val_root_mean_squared_error']).mean()

    # Test Score
    preds = model.predict(X_test,batch_size = batch_size,callbacks=callbacks_list)
    preds = tf.cast(preds, tf.float32)
    y_test = tf.cast(y_test, tf.float32)

    m = tf.keras.metrics.RootMeanSquaredError()
    m.update_state(y_test,preds)
    test_score = m.result().numpy()

    train_scores = np.append(train_scores,train_score)
    val_scores = np.append(val_scores,val_score)
    test_scores = np.append(test_scores,test_score)


# %%
plt.figure(figsize=(10,8))
plt.plot(range(len(train_scores)),train_scores,label='Train Scores')
plt.plot(range(len(val_scores)),val_scores,label='Validation Scores')
plt.plot(range(len(test_scores)),test_scores,label='Test Scores')
plt.legend()


# ### Hold Out Score

# Here, the same model is trained on the last n months of the Validation and tested on a Holdout Set never seen before.

# %%
n_months = 4


# %%
X_train, X_holdout, y_train, y_holdout = train_test_split(X, y, test_size=0.125, shuffle=False)


# %%
X_train = X_train[X_train['Month_Number'].isin(X_train['Month_Number'].unique()[-n_months:])]
y_train  = y_train.iloc[-X_train.shape[0]:]


# %%
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.143, shuffle=False)


# %%
X_train = preprocessing_X(X_train)
X_val = preprocessing_X(X_val)
X_holdout = preprocessing_X(X_holdout)

X_train = X_train.drop(['ID','WF','Set','Month_Number'],axis=1)
X_val = X_val.drop(['ID','WF','Set','Month_Number'],axis=1)
X_holdout = X_holdout.drop(['ID','WF','Set','Month_Number'],axis=1)

y_train,scaler = preprocessing_y(y_train)
y_val,_ = preprocessing_y(y_val)
y_holdout,_ = preprocessing_y(y_holdout)


# %%
X_train, y_train = split_sequences(X_train.values,y_train.values,n_steps)
X_val, y_val = split_sequences(X_val.values,y_val.values,n_steps)
X_holdout, y_holdout = split_sequences(X_holdout.values,y_holdout.values,n_steps)


# %%
model = LSTM_Model(input_shape)


# %%
callbacks_list = [tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=patience, verbose=0, min_delta=1e-8)]
model.compile(optimizer='adam', loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError()])
model.fit(x = X_train, y = y_train, batch_size = batch_size, epochs = epochs, validation_data = (X_val, y_val), callbacks=callbacks_list)


# %%
preds = model.predict(X_holdout,batch_size = batch_size,callbacks=callbacks_list)


# Here, Predicitions and True Labels of the Holdout Set are scaled back to measure the CAPE Error, and be compared on a Chart.

# %%
preds = scaler.inverse_transform(preds)
y_holdout = scaler.inverse_transform(pd.DataFrame(y_holdout))


# %%
def metric_cnr(preds,labels):
    cape_cnr = 100*np.sum(np.abs(preds-labels))/np.sum(labels)
    return 'CAPE', cape_cnr


# %%
metric_cnr(preds,y_holdout)


# %%
plt.figure(figsize=(15,8))
preds_len = np.arange(len(y_holdout))
plt.plot(preds_len,y_holdout,label='True Values')
plt.plot(preds_len,preds,'r--',label='Predicts')
plt.legend()


# ## Submission Generation

# Here some different models for Submit Generation are tried. The first One works with a Expanded Window Training Set.

# %%
def WF_submit_gen(full_data,full_label,WF,input_shape,batch_size):
  # Get Data of Wind Farm
  X_WF = full_data[full_data['WF']==WF]

  full_X_train = X_WF[X_WF['Set']=='Train']
  full_X_test = X_WF[X_WF['Set']=='Test']
  full_y_train = full_label[full_label['ID'].isin(full_X_train['ID'])]

  # Months Loop
  WF_preds = pd.DataFrame(columns=['Production'])
  predicted_X_test = pd.DataFrame(columns=full_X_train.columns)
  predicted_X_test.index.names = ['Time']
  for month in full_X_test['Month_Number'].unique():

    print("Farm {} - Month {}".format(WF,month))

    # Special Condition for Month 10
    if month == 9:
      X_test = full_X_test[full_X_test['Month_Number']==month]
      X_test_10 = full_X_test[full_X_test['Month_Number']==10]
      X_test_pure = pd.concat([X_test,X_test_10])
    else:
      X_test_pure = full_X_test[full_X_test['Month_Number']==month]

    if month == 10:
      continue

    # Append Data Already Predicted
    X_train = pd.concat([full_X_train,predicted_X_test])
    y_train = pd.concat([full_y_train,WF_preds])

    # Split Train Data on Train and Validation Set
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, shuffle=False)

    # Preprocessing Data
    X_train = preprocessing_X(X_train)
    X_val = preprocessing_X(X_val)
    X_test = preprocessing_X(X_test_pure)

    # Shift-Save on X_test
    X_test = shift_save(X_test,n_steps)

    X_train = X_train.drop(['ID','WF','Set','Month_Number'],axis=1)
    X_val = X_val.drop(['ID','WF','Set','Month_Number'],axis=1)
    X_test = X_test.drop(['ID','WF','Set','Month_Number'],axis=1)

    y_train,scaler_y = preprocessing_y(y_train)
    y_val,_ = preprocessing_y(y_val)

    # Subset Creation 

    X_train, y_train = split_sequences(X_train.values,y_train.values,n_steps)
    X_val, y_val = split_sequences(X_val.values,y_val.values,n_steps)
    X_test, _ = split_sequences(X_test.values,None,n_steps)

    # Model Creation and Training
    model = LSTM_Model(input_shape,batch_size=batch_size)
    callbacks_list = [tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=patience, verbose=0, min_delta=1e-8)]
    model.compile(optimizer='adam', loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError()])
    model.fit(x = X_train, y = y_train, batch_size = batch_size, epochs = epochs, validation_data = (X_val, y_val), callbacks=callbacks_list,shuffle=False)

    # Prediction Generation
    pred = model.predict(X_test,batch_size = batch_size,callbacks=callbacks_list)
    pred = pred.reshape(pred.shape[0])
    pred = pd.DataFrame(pred,columns=['Production'])

    # Save Predictions on Final Array
    predicted_X_test = pd.concat([predicted_X_test,X_test_pure])

    pred = scaler_y.inverse_transform(pred)
    pred = pd.DataFrame(pred,columns=['Production'])
    WF_preds = pd.concat([WF_preds,pred])

  #WF_preds = scaler_y.inverse_transform(pd.DataFrame(WF_preds))
  #WF_preds = pd.DataFrame(WF_preds,columns=['Production'])

  return WF_preds


# Here a Direct Submission Model is created. In other words, All the training data is used to forecast all the Test Data at once.

# %%
def WF_submit_gen_3(full_data,full_label,WF,input_shape,batch_size):
  # Get Data of Wind Farm
  X_WF = full_data[full_data['WF']==WF]

  # Scale X Data
  X_WF = preprocessing_X(X_WF)

  full_X_train = X_WF[X_WF['Set']=='Train']
  full_X_test = X_WF[X_WF['Set']=='Test']
  full_y_train = full_label[full_label['ID'].isin(full_X_train['ID'])]

  # Scale y_train
  full_y_train,scaler_y = preprocessing_y(full_y_train)

  # Subsets Generation
  full_X_train = full_X_train.drop(['ID','WF','Set'],axis=1)
  full_X_train, full_y_train = split_sequences(full_X_train.values,full_y_train.values,n_steps)

  full_X_test_split = full_X_test.drop(['ID','WF','Set'],axis=1)
  full_X_test_split = shift_save(full_X_test_split,n_steps)
  full_X_test_split, _ = split_sequences(full_X_test_split.values,None,n_steps)

  # Split Train Data on Train and Validation Set
  X_train, X_val, y_train, y_val = train_test_split(full_X_train, full_y_train, test_size=0.3, shuffle=False)

  # Model Creation and Training
  model = LSTM_Model(input_shape,batch_size=batch_size)
  callbacks_list = [tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=patience, verbose=0, min_delta=1e-8)]
  model.compile(optimizer='adam', loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError()])
  model.fit(x = X_train, y = y_train, batch_size = batch_size, epochs = epochs, validation_data = (X_val, y_val), callbacks=callbacks_list,shuffle=False)

  # Prediction Generation
  pred = model.predict(full_X_test_split,batch_size = batch_size,callbacks=callbacks_list)
  pred = pred.reshape(pred.shape[0])

  WF_preds = scaler_y.inverse_transform(pd.DataFrame(pred))
  WF_preds = pd.DataFrame(WF_preds,columns=['Production'])

  return WF_preds


# Here, a model is created with Rolling Window on Training Data, instead of Expanded Window. This model is less expensive in processing.

# %%
def WF_submit_gen_4(full_data,full_label,WF,input_shape,batch_size,n_months):
  # Get Data of Wind Farm
  X_WF = full_data[full_data['WF']==WF]

  full_X_train = X_WF[X_WF['Set']=='Train']
  full_X_test = X_WF[X_WF['Set']=='Test']
  full_y_train = full_label[full_label['ID'].isin(full_X_train['ID'])]

  # Months Loop
  WF_preds = pd.DataFrame(columns=['Production'])
  predicted_X_test = pd.DataFrame(columns=full_X_train.columns)
  predicted_X_test.index.names = ['Time']
  for month in full_X_test['Month_Number'].unique():

    print("Farm {} - Month {}".format(WF,month))

    # Special Condition for Month 10
    if month == 9:
      X_test = full_X_test[full_X_test['Month_Number']==month]
      X_test_10 = full_X_test[full_X_test['Month_Number']==10]
      X_test_pure = pd.concat([X_test,X_test_10])
    else:
      X_test_pure = full_X_test[full_X_test['Month_Number']==month]

    if month == 10:
      continue

    # Append Data Already Predicted
    X_train = pd.concat([full_X_train,predicted_X_test])
    y_train = pd.concat([full_y_train,WF_preds])
    
    X_train = X_train[X_train['Month_Number'].isin(X_train['Month_Number'].unique()[-n_months:])]
    y_train  = y_train.iloc[-X_train.shape[0]:]

    # Split Train Data on Train and Validation Set
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, shuffle=False)

    # Preprocessing Data
    X_train = preprocessing_X(X_train)
    X_val = preprocessing_X(X_val)
    X_test = preprocessing_X(X_test_pure)

    # Shift-Save on X_test
    X_test = shift_save(X_test,n_steps)

    X_train = X_train.drop(['ID','WF','Set','Month_Number'],axis=1)
    X_val = X_val.drop(['ID','WF','Set','Month_Number'],axis=1)
    X_test = X_test.drop(['ID','WF','Set','Month_Number'],axis=1)

    y_train,scaler_y = preprocessing_y(y_train)
    y_val,_ = preprocessing_y(y_val)

    # Subset Creation 

    X_train, y_train = split_sequences(X_train.values,y_train.values,n_steps)
    X_val, y_val = split_sequences(X_val.values,y_val.values,n_steps)
    X_test, _ = split_sequences(X_test.values,None,n_steps)

    # Model Creation and Training
    model = LSTM_Model(input_shape,batch_size=batch_size)
    callbacks_list = [tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=patience, verbose=0, min_delta=1e-8)]
    model.compile(optimizer='adam', loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError()])
    model.fit(x = X_train, y = y_train, batch_size = batch_size, epochs = epochs, validation_data = (X_val, y_val), callbacks=callbacks_list,shuffle=False)

    # Prediction Generation
    pred = model.predict(X_test,batch_size = batch_size,callbacks=callbacks_list)
    pred = pred.reshape(pred.shape[0])
    pred = pd.DataFrame(pred,columns=['Production'])

    # Save Predictions on Final Array
    predicted_X_test = pd.concat([predicted_X_test,X_test_pure])

    pred = scaler_y.inverse_transform(pred)
    pred = pd.DataFrame(pred,columns=['Production'])
    WF_preds = pd.concat([WF_preds,pred])

  return WF_preds


# Here, the Submissio Generation Modelo is applied to each Wind Farm.

# %%
final_preds = pd.DataFrame()
for WF in full_data['WF'].unique():
  WF_preds = WF_submit_gen_4(full_data,full_label,WF,input_shape,batch_size,n_months)
  final_preds = final_preds.append(WF_preds)


# Finally the Submit File is created, and a Final Plot of the Submission is done.

# %%
final_preds = final_preds.reset_index().drop('index',axis=1)
final_preds['ID'] = pd.read_csv(r'C:\Users\andre_\OneDrive\Documentos\GitHub\cnr\Data\random_submission_example.csv')['ID']
final_preds = final_preds.set_index('ID')


# %%
final_preds


# %%
plt.plot(final_preds.index,final_preds['Production'])


# %%
final_preds.to_csv('Submission_LSTM.csv')


# %%



