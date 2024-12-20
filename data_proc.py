#basic libraries
import pandas as pd
import math
import numpy as np
import copy

#functions definitions
import data_proc_fun

# all available weather describing features
ALL_FEATURES = ["temperature", "pressure", 
                "wind_direction", "humidity", 
                "weather_description", "wind_speed",
                "city_attributes"]

ALL_CITIES = ['Vancouver', 'Portland', 'San Francisco', 'Seattle',
            'Los Angeles', 'San Diego', 'Las Vegas', 'Phoenix', 'Albuquerque',
            'Denver', 'San Antonio', 'Dallas', 'Houston', 'Kansas City',
            'Minneapolis', 'Saint Louis', 'Chicago', 'Nashville', 'Indianapolis',
            'Atlanta', 'Detroit', 'Jacksonville', 'Charlotte', 'Miami',
            'Pittsburgh', 'Toronto', 'Philadelphia', 'New York', 'Montreal',
            'Boston', 
            'Beersheba', 'Tel Aviv District',
            'Eilat', 'Haifa',
            'Nahariyya', 'Jerusalem']

# choose cities
cities = ["Indianapolis", "Saint Louis", "Chicago", "Detroit", "Pittsburgh"]

file_path = './data/city_attributes.csv'
#data_proc_fun.disp_map_chosen_cities(cities, file_path)

# chosen weather features 
features = [
    "temperature", 
    "pressure", 
    "wind_direction", 
    "humidity", 
    "wind_speed"
    ]

# reading data for features
# list of dataframes
data = []
data_path = './data/'

for feat in features:
    tmp_data = pd.read_csv(data_path + feat + '.csv')
    datetime = tmp_data["datetime"]

    tmp_data = tmp_data[cities]
    #tmp_data["datetime"] = datetime

    # drop first row cause its always NaN
    tmp_data = tmp_data.drop(0, axis = 0)
    
    data.append(tmp_data)

f = len(features)
c = len(cities)


if True:
    for i in range(f):
        data[i] = data[i][0:5000]

arr = np.array([data[i].to_numpy() for i in range(f)])

print("Checking for NaN values in data...")
print("Replacing all NaN with new values...")

NaN_counter = 0
for j in range(c): 
    #print(cities[j])
    for i in range(f):
        #print(features[i])
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
        # print('NaN objects found: ', check) 
        # print('Objects replaced with a new value: ', counter) 
        NaN_counter += tmp_data.isna()[j].astype(int).sum()
if NaN_counter == 0:
    print('##### All NaN objects simulated. #####')
else:
    print("##### There was an Error while removing NaNs. #####")
print()

# new value is written to arr, not dataframe, so getting rid of NaNs does not work perfectly yet. The loop
# with writing the new value to the array is called more times than necessary, but it works
# Recursion could be used for removing NaNs

print("Checking if datetime column is continous...")

# This check takes only 0-th weather feature, because all features shares the same datetime
# These 2 dataframes, store the same information, BUT time1 starts from 1, so its shifted 1 hour forward
# with respect to time2. This allows for an easy check. If the datetime column is continous
time_arr = datetime.to_numpy()  
time1 = pd.to_datetime(pd.DataFrame(time_arr[1:(arr.shape[1])])[0])
time2 = pd.to_datetime(pd.DataFrame(time_arr[0:(arr.shape[1] - 1)] )[0])

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

# Checking if datetime starts at 00:00, if not deleting the incomplete day
print("Checking if datetime starts at 00:00, if not, deleting the incomplete day...")
# getting the sequence of the following hours in data 
time_hour = pd.to_datetime(pd.DataFrame( time_arr[0:(arr.shape[1])] )[0]).dt.hour
check = 0
tmp_list = [None for _ in range(f)]
while time_hour[check] != 0:
    check += 1 
for i in range(f):
    tmp_list[i] = arr[i][(check - 1):arr.shape[1]]
arr = np.array(tmp_list)
print("##### Only full days present in data #####")
print()

# if weather description present in chosen features, perform encoding
if 'weather_description' in features:
    print("Encoding weather_description into categories...")
    weather_description, weather_categories = data_proc_fun.weather_description_encoding(arr, features, c)
    weather_description = weather_description.T
    print("##### Encoding into weather categories finished. #####")
    print()

# if wind direction present in chosen features, perform encoding
if "wind_direction" in features:
    print("Encoding wind direction...")
    wind_direcion = data_proc_fun.encode_wind_direction(arr, features, c)
    wind_direcion = wind_direcion.T
    print("##### Wind directions encoding finished. #####")
    print()

# weeks
weeks = pd.to_datetime(pd.DataFrame( time_arr[0:(arr.shape[1])] )[0]).dt.isocalendar().week
#print(weeks)
# days
days = pd.to_datetime(pd.DataFrame( time_arr[0:(arr.shape[1])] )[0]).dt.day_of_year

days = days / 365
days = days * 2 * math.pi
days_dir1 = np.sin(days)
days_dir2 = np.cos(days)
# time
hour = pd.to_datetime(pd.DataFrame( time_arr[0:(arr.shape[1])] )[0]).dt.hour
#print(time)

cities_weather = [None for _ in range(c)]
arr = arr.T
for i in range(c):
    cities_weather[i] = pd.DataFrame(arr[i])
#print(cities_weather[0].shape)
#print(cities_weather[0].head)
#cities_weather = pd.DataFrame(cities_weather)
#print(cities_weather[0].shape)

if "wind_direction" in features:
    # Note: In case wind speed = 0, shouldnt wind direction be encoded like [0, 0], currently
    # wind speed = 0 is always followed by angle = 360, so it gets encoding like [1, 0], which means
    # no wind in north direction. Meaning behind it for a human is clear, but wouldnt NN benefit from 
    # getting [0, 0] to entirely ignore wind information in this case?  
    print("Expanding wind direction encodings into seperate columns. This takes a minute...")
    wind_index = features.index("wind_direction")
    for i in range(c):
        #del cities_weather[i][wind_index]
        cities_weather[i] = cities_weather[i].drop(wind_index, axis = 1)
        cities_weather[i] = cities_weather[i].T.reset_index(drop = True).T
        #print(cities_weather[i].head)
        #cities_weather[i].columns = [i for i in range(cities_weather[i].shape[1])]

        # wind direction is encoded into north/south and east/west categories, 
        # thats why the loop is in range(2)
        for j in range(2):
            cities_weather[i]['dir' + str(j + 1)] = wind_direcion[j][i]
    
    # update features because additional columns were inserted
    # new wind direction columns, as new columns, are always adjusted at the end
    # of the dataframe, old wind direction column was deleted  
    features.remove("wind_direction")
    features.append("dir1")
    features.append("dir2")

    print("##### Expanding wind direction encodings finished. #####")
    print()

#print(cities_weather[0][:][0:15])
#print(cities_weather[0][:][20:27])
#print(cities_weather.shape)

if "weather_description" in features:
    print("Expanding weather description encodings into seperate columns. This takes a minute...")
    weather_index = features.index("weather_description")
    for i in range(c):
        cities_weather[i] = cities_weather[i].drop(weather_index, axis = 1)
        cities_weather[i] = cities_weather[i].T.reset_index(drop = True).T

        # weather description is encoded into clouds, rain, snow, thunderstorm and atmospheric
        # categories, thats why the loop is in range(5), 5 categories
        for j in range(len(weather_categories)):
            cities_weather[i][weather_categories[j]] = weather_description[j][i]
    
    # update features because additional columns were inserted
    # new wind direction columns, as new columns, are always adjusted at the end
    # of the dataframe, old wind direction column was deleted  
    features.remove("weather_description")
    for category in weather_categories:
        features.append(category)     

    print("##### Expanding weather description encodings finished. #####")
    print()

f = len(features)
for i in range(c):
    #short_city_feat = copy.deepcopy(short_features)
    #short_city_feat = [cities[i] + feat for feat in short_features]
    cities_weather[i] = cities_weather[i].T.reset_index(drop = True).T
    cities_weather[i].columns = [cities[i][:3] + '_' + feat[:4] for feat in features]

print("Saving processed data to files...")
path = "./proc_data/"
for i in range(1, c):
    cities_weather[0] = cities_weather[0].join(cities_weather[i])


cities_weather[0]["days_dir1"] = days_dir1
cities_weather[0]["days_dir2"] = days_dir2
cities_weather[0]["hour"] = hour


cities_weather[0].to_csv(path + 'concat_clean_data.csv', sep="\t", index = False, float_format='%.3f')

print("##### All data successfully saved to files. #####")
print()