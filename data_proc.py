import pandas as pd
import math
import numpy as np

def disp_df(data, features):
    for i in range(len(features)):
        print('##### ' + features[i].upper() + ' #####')
        print(data[i].head())
        print(data[i].shape)
        print()

def disp_arr(data, features, cities, start = 0, end = 7):
    for i in range(len(features)):
        print('##### ' + features[i].upper() + ' #####')
        print(cities)
        print(data[i][start:end])
        print(data[i].shape)
        print()

# cities chosen for test
cities = ["Indianapolis", "Saint Louis", "Chicago", "Detroit", "Pittsburgh", "Nashville"]

# all available weather describing features
ALL_FEATURES = ["temperature", "pressure", 
                "wind_direction", "humidity", 
                "weather_description", "wind_speed",
                "city_attributes"]

# chosen weather features 
features = ["temperature", "pressure", 
                 "wind_speed", "wind_direction", "humidity"]

# reading data for features
# list of dataframes
data = []
data_path = './data/'

for feat in features:
    tmp_data = pd.read_csv(data_path + feat + '.csv')
    datetime = tmp_data["datetime"]
    
    tmp_data = tmp_data[cities]
    tmp_data["datetime"] = datetime

    # drop first row cause its always NaN
    tmp_data = tmp_data.drop(0, axis = 0)
    
    data.append(tmp_data)

f = len(features)
c = len(cities)


if False:
    for i in range(f):
        data[i] = data[i][0:5000]

arr = np.array([data[i].to_numpy() for i in range(f)])

print("Checking for NaN values in data...")
print("Replacing all NaN with new values...")

for j in range(c): 
    print(cities[j])
    for i in range(f):
        print(features[i])
        indices = data[i][cities[j]][data[i].isna()[cities[j]] == True].index
        check = data[i].isna()[cities[j]].astype(int).sum()
        counter = 0 
        for ind in indices:
            val = 0
            iter = 0
            while math.isnan(data[i][cities[j]][ind - iter]):
                iter += 1
            val = data[i][cities[j]][ind - iter]
            arr[i][ind - 1][j] = val
            counter += 1
        
        tmp_data = pd.DataFrame(arr[i])
        print('NaN objects found: ', check) 
        print('Objects replaced with a new value: ', counter) 
        if check != 0 and tmp_data.isna()[j].astype(int).sum() == 0:
            print('All ' + str(check) + ' NaN objects simulated.')
        print()

# new value is written to arr, not dataframe, so getting rid of NaNs does not work perfectly yet. The loop
# with writing the new value to the array is called too many times

print("Checking if datetime column is continous...")

time1 = pd.to_datetime(pd.DataFrame(arr[0].T[-1][1:(arr.shape[1])])[0])
time2 = pd.to_datetime(pd.DataFrame( arr[0].T[-1][0:(arr.shape[1] - 1)] )[0])

time_discontinuity = 0
for i in range(len(time1)):
    if (pd.Timedelta( time1[i] - time2[i]) != pd.Timedelta("1 hour")):
        time_discontinuity += 1
        print("Row: ", i)

if (time_discontinuity == 0):
    print("##### Datetime is continous #####")
    print()
else:
    print("##### Discontinuity found #####")
    print()



# endcode win_direction
print("Encoding wind direction...")
wind_dir_index = features.index("wind_direction")

def encode_wind_direction(angle):
    direction_encoded = [0 for _ in range(8)]
    counter = 0
    while (angle > (22.5 + counter * 45)): 
        counter += 1      
    if counter == 8:
        direction_encoded[0] = 1
    else:
        direction_encoded[counter] = 1
    return direction_encoded

def encode_wind_direction2(angle):
    direction_encoded = [0 for _ in range(2)]
    
    direction_encoded[0] = np.cos(angle * np.pi/180)
    direction_encoded[1] = np.sin(angle * np.pi/180)

    return direction_encoded

for i in range(c):
    for j in range(arr.shape[1]):
        arr[wind_dir_index][j][i] = np.array(encode_wind_direction2(arr[wind_dir_index][j][i]))

print("Encoding into wind directions categories finished...")

#disp_arr(arr, features, cities)

# encoding datetime
# weeks
weeks = pd.to_datetime(pd.DataFrame(arr[0].T[-1])[0]).dt.isocalendar().week
#print(weeks)
# days
days = pd.to_datetime(pd.DataFrame(arr[0].T[-1])[0]).dt.day_of_year
# time
time = pd.to_datetime(pd.DataFrame(arr[0].T[-1])[0]).dt.hour

cities_weather = [None for _ in range(c)]

print("Expanding encodings into seperate columns. This takes couple of minutes...")

for i in range(c):
    tmp_df = pd.DataFrame([arr[j].T[i] for j in range(f)]).transpose()
    #print(tmp_df)

    tmp_arr = [None for _ in range(arr.shape[1])]
    for j in range(arr.shape[1]):
        tmp_arr[j] = np.array(tmp_df[2])[j] 
    
    tmp_df2 = pd.DataFrame(tmp_arr)
    #print(tmp_df2)

    tmp_arr = [None for _ in range(arr.shape[1])]
    for j in range(arr.shape[1]):
        tmp_arr[j] = np.array(tmp_df[4])[j] 

    tmp_df3 = pd.DataFrame(tmp_arr)
    #print(tmp_df3)

    del tmp_df[2]
    del tmp_df[4]

    for j in range(2):
        tmp_df['dir' + str(j)] = tmp_df2[j]

    #print(tmp_df)
    cities_weather[i] = tmp_df

cities_weather = np.array(cities_weather)
print("### Before contcatenation ###")
print(cities_weather.shape)

cities_weather = np.concatenate(cities_weather, axis = 1)
print("### After contcatenation ###")
print(cities_weather)
print(cities_weather.shape)

print("### Adding a column of weeks ###")
print(cities_weather)
print(cities_weather.shape)


df = pd.DataFrame(cities_weather)
col_names = []
for city in cities:
    for feature in features:
        if feature == "wind_direction":
            col_names.append(city + '_wind_dir0')
            col_names.append(city + '_wind_dir1')
        else:
            col_names.append(city + '_' + feature)

print(df.head())
df.columns = col_names
df.to_csv('./proc_data/cities_weather.csv', index = False)