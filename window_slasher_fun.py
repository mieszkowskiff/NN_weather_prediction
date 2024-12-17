import numpy as np
import pandas as pd

def convert_date_time(df, day_ticks):
    dims = df.shape
    day_zero = 0

    for i in range(dims[0]):
        df['day'][i] = day_zero
        if (i % day_ticks == day_ticks-1):
            if (day_zero < 4):
                day_zero += 1
            else:
                day_zero = 0

    for i in range(dims[0]):
        df['time'][i] = i % day_ticks

    return df

def split_train_test_data(df, ratio):
    dim = df.shape
    return df[:][0:int(ratio*dim[0])], df[:][int(ratio*dim[0]):dim[0]]

def split_windows_X_Y(windows, x_bools, y_bools, m, n, day_ticks, window_tick_size):
    dims = windows.shape
    
    x_mask = np.array([x_bools for _ in range(m*day_ticks)]).reshape(-1)
    y_mask = np.array([y_bools for _ in range(n*day_ticks)]).reshape(-1)

    X_windows = np.array( [ windows[i][0:(m*day_ticks)] for i in range(dims[0]) ] )
    Y_windows = np.array( [ windows[i][(m*day_ticks):window_tick_size] for i in range(dims[0]) ] )

    X_windows = X_windows.reshape(dims[0], -1)
    Y_windows = Y_windows.reshape(dims[0], -1)

    X_windows = np.array( [ X_windows[i][x_mask] for i in range(dims[0]) ] )
    Y_windows = np.array( [ Y_windows[i][y_mask] for i in range(dims[0]) ] )

    return X_windows, Y_windows

def save_array_to_csv(array, filename):
    try:
        np.savetxt(filename, array, delimiter=',', fmt='%.6f')
        print(f"Array saved to {filename}")
    except Exception as e:
        print(f"An error occurred: {e}")
