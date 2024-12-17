from pandas import read_csv
import numpy as np
import pandas as pd
import csv
import window_slasher_fun

CONST_NUM_CITIES = 6
CONST_NUM_FEATS = 11
CONST_SKIP_HOURS = 2

CONST_day_ticks = 24
CONST_norm_factor = 0.5
#number of days based on which the forecast is made
CONST_m = 3
#number of days to forecast
CONST_n = 2

CONST_window_day_size = CONST_n+CONST_m
CONST_window_tick_size = CONST_window_day_size * CONST_day_ticks

# load data
df = pd.DataFrame(read_csv('./proc_data/concat_clean_data_new.csv', sep=",", header=None))
# delete first 11 hours because they come from incomplete day
df = df[12:-1]
# take every CONST_SKIP_HOURS hour( every CONST_SKIP_HOURS observation), to reduce the amount of data
df = df.reset_index(drop=True)
df = df.apply(pd.to_numeric, errors='coerce')
print(df.shape)
dim = df.shape

# split data in ratio
X, Y = window_slasher_fun.split_train_test_data(df, 0.80)
X = X.reset_index(drop=True)
Y = Y.reset_index(drop=True)

# params to normalize data
m = np.mean(X, 0)
max = np.max(X, 0)
min = np.min(X, 0)

# normalizing data to [lower, upper]
lower = -1
upper = 1
for i in df.columns:
    X[i] = (X[i]-min[i])*(upper - lower)/(max[i] - min[i]) + lower
    Y[i] = (Y[i]-min[i])*(upper - lower)/(max[i] - min[i]) + lower
print(X)
print(Y)

# determine which features are present in X
x_bools = [True for _ in range(dim[1])]

# determine which features are present in Y to predict
# 0 - temperature
# 3 - wind speed
y_feats_num = [0, 3]
y_bools = [False for _ in range(dim[1])]
for ind in y_feats_num:
    for i in range(CONST_NUM_CITIES):
        y_bools[ind + i*CONST_NUM_FEATS] = True

m = m[x_bools]
max = max[x_bools]
min = min[x_bools]

dim_train = X.shape
dim_test = Y.shape
print(dim_train)
print(dim_test)

num_of_windows_train = int((dim_train[0] - CONST_window_tick_size)/CONST_day_ticks)
num_of_windows_test = int((dim_test[0] - CONST_window_tick_size)/CONST_day_ticks)

windows_train = np.array([np.array(X[:][i*CONST_day_ticks:(i*CONST_day_ticks + CONST_window_tick_size)]) for i in range(num_of_windows_train)])
windows_test = np.array([np.array(Y[:][i*CONST_day_ticks:(i*CONST_day_ticks + CONST_window_tick_size)]) for i in range(num_of_windows_test)])

X_windows_train, Y_windows_train = window_slasher_fun.split_windows_X_Y(windows_train, x_bools, y_bools, CONST_m, CONST_n, CONST_day_ticks, CONST_window_tick_size)
X_windows_test, Y_windows_test =  window_slasher_fun.split_windows_X_Y(windows_test, x_bools, y_bools, CONST_m, CONST_n, CONST_day_ticks, CONST_window_tick_size)

print(X_windows_train.shape)
print(X_windows_test.shape)

num_of_feat_Y = 2

temp_mask = [False for _ in range(CONST_day_ticks*CONST_NUM_CITIES*CONST_n*num_of_feat_Y)]
wind_mask = [False for _ in range(CONST_day_ticks*CONST_NUM_CITIES*CONST_n*num_of_feat_Y)]
#print(np.array(temp_mask).shape)
#print(np.array(wind_mask).shape)


for j in range(CONST_n):
    for i in range(CONST_day_ticks):
        temp_mask[j*CONST_day_ticks*num_of_feat_Y*CONST_NUM_CITIES 
                  + num_of_feat_Y*CONST_NUM_CITIES*i] = True

for j in range(CONST_n):
    for i in range(CONST_day_ticks):
        wind_mask[j*CONST_day_ticks*num_of_feat_Y*CONST_NUM_CITIES 
                  + num_of_feat_Y*CONST_NUM_CITIES*i + 1] = True

#print(temp_mask)
#print(wind_mask)
#print(np.array(temp_mask).shape)
#print(np.array(wind_mask).shape)

Y_temp_train = Y_windows_train.T[temp_mask].T
Y_temp_test = Y_windows_test.T[temp_mask].T

Y_wind_train = Y_windows_train.T[wind_mask].T
Y_wind_test = Y_windows_test.T[wind_mask].T

Y_windows_train = [None for _ in range(Y_temp_train.shape[0])]
#np.array((Y_temp_train.shape[0], CONST_n*num_of_feat_Y))
Y_windows_test = [None for _ in range(Y_temp_test.shape[0])]
#np.array((Y_temp_test.shape[0], CONST_n*num_of_feat_Y))

# get mean temperature and max wind speed
for i in range(Y_temp_train.shape[0]):
    Y_windows_train[i] = np.array([np.mean(Y_temp_train[i][0:CONST_day_ticks]), 
                                    np.max(Y_wind_train[i][0:CONST_day_ticks]), 
                                    np.mean(Y_temp_train[i][CONST_day_ticks:-1]), 
                                    np.max(Y_temp_train[i][CONST_day_ticks:-1])])
    
for i in range(Y_temp_test.shape[0]):
    Y_windows_test[i] = np.array([np.mean(Y_temp_test[i][0:CONST_day_ticks]), 
                                    np.max(Y_wind_test[i][0:CONST_day_ticks]), 
                                    np.mean(Y_temp_test[i][CONST_day_ticks:-1]), 
                                    np.max(Y_temp_test[i][CONST_day_ticks:-1])])

X_windows_train = np.array(X_windows_train)
X_windows_test = np.array(X_windows_test)
Y_windows_train = pd.DataFrame(Y_windows_train)
Y_windows_test = pd.DataFrame(Y_windows_test)

'''
print(Y_temp_train.shape)
print(Y_wind_test.shape)
print(Y_temp_train.shape)
print(Y_wind_test.shape)
'''
skip_hours = [False for _ in range(X_windows_train.shape[1])]
for i in range(int(CONST_m*CONST_day_ticks/CONST_SKIP_HOURS)):
    # + 1 because there is the time variable indicating the week of the year 
    for j in range((CONST_NUM_FEATS*CONST_NUM_CITIES + 1)):
        skip_hours[i*(CONST_NUM_CITIES*CONST_NUM_FEATS + 1)*CONST_SKIP_HOURS + j] = True

X_windows_train = X_windows_train.T[skip_hours].T
X_windows_test = X_windows_test.T[skip_hours].T

print(X_windows_train.shape)
print(X_windows_test.shape)
print(Y_windows_train.shape)
print(Y_windows_test.shape)


path = './clean_norm_data/new_feats/'
window_slasher_fun.save_array_to_csv(X_windows_train, path + 'X_train.csv')
window_slasher_fun.save_array_to_csv(Y_windows_train, path + 'Y_train.csv')

window_slasher_fun.save_array_to_csv(m, path + 'mean.csv')
window_slasher_fun.save_array_to_csv(min, path + 'min.csv')
window_slasher_fun.save_array_to_csv(max, path + 'max.csv')
try:
    with open(path + 'norm_factor.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([CONST_norm_factor])
    print(f"Float value saved to {path + 'norm_factor.csv'}")
except Exception as e:
    print(f"An error occurred: {e}")

window_slasher_fun.save_array_to_csv(X_windows_test, path + 'X_test.csv')
window_slasher_fun.save_array_to_csv(Y_windows_test, path + 'Y_test.csv')