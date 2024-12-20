import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt

def encode_angle(angle):
    direction_encoded = [0 for _ in range(2)]
    
    direction_encoded[0] = np.sin(angle * np.pi/180)
    direction_encoded[1] = np.cos(angle * np.pi/180)

    return direction_encoded

def encode_wind_direction(arr, features, c):
    wind_dir_index = features.index("wind_direction")

    wind_direction = np.zeros((arr.shape[1], c, 2))

    for i in range(c):
        for j in range(arr.shape[1]):
            wind_direction[j][i] = np.array(encode_angle(arr[wind_dir_index][j][i]))

    return wind_direction

def disp_map_chosen_cities(chosen_cities, file_path):
    # Read data from file
    path = file_path
    data = pd.read_csv(path)

    # Extract latitude and longitude
    latitudes = data['Latitude']
    longitudes = data['Longitude']
    city_names = data['City']
    mask = [False for _ in range(len(city_names))]
    for i in range(len(city_names)):
        if city_names[i] in chosen_cities:
            mask[i] = True

    chosen_latitudes = latitudes[mask]
    chosen_longitudes = longitudes[mask]
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.scatter(longitudes, latitudes, c='blue', marker='o', edgecolor='black', alpha=0.7)
    plt.scatter(chosen_longitudes, chosen_latitudes, c='red', marker='o', edgecolor='black', alpha=0.7)

    # Annotate each city
    for i, city in enumerate(city_names):
        plt.text(longitudes[i] + 0.2, latitudes[i] + 0.2, city, fontsize=8)

    # Add labels and title
    plt.title('Cities on Map')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True)

    # Show the map
    plt.show()

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

def weather_description_encoding(arr, features, c):
    #print("##### Unique strings in weather_description #####")
    #print()
    string_index = features.index("weather_description")

    strings = set()

    for i in range(c):
        strings = strings.union(set(arr[string_index].T[i]))
        
    #print(strings)
    #print()
    #print("WHAT IS THIS SHIT BRO. NO CAP")
    #print()

    weather_categories = ["clouds", "rain", "snow", "thunderstorm", "atmospheric"]

    # encoding weather description with respect to weather_categories, "sky is clear" ---> [1, 0, 0, 0, 0]
    # "very heavy rain" ---> [0, 4, 0, 0, 0]

    # Define categories with weighted encoding
    categories = {
        "clear_clouds": {
            "sky is clear": 1,
            "few clouds": 2,
            "scattered clouds": 2,
            "broken clouds": 3,
            "overcast clouds": 3
        },
        "rain_drizzle": {
            "light rain": 1,
            "moderate rain": 2,
            "heavy intensity rain": 3,
            "very heavy rain": 4,
            "freezing rain": 3,
            "drizzle": 1,
            "light intensity drizzle": 1,
            "heavy intensity drizzle": 2,
            "shower drizzle": 1,
            "light intensity drizzle rain": 1,
            "light intensity shower rain": 1,
            "shower rain": 2,
            "proximity shower rain": 1
        },
        "snow_sleet": {
            "light snow": 1,
            "snow": 2,
            "heavy snow": 3,
            "light rain and snow": 2,
            "sleet": 2,
            "light shower sleet": 1,
            "light shower snow": 1,
            "shower snow": 2,
            "heavy shower snow": 3
        },
        "thunderstorms": {
            "thunderstorm": 1,
            "thunderstorm with rain": 2,
            "thunderstorm with light rain": 1,
            "thunderstorm with heavy rain": 3,
            "proximity thunderstorm": 1,
            "proximity thunderstorm with rain": 2,
            "proximity thunderstorm with drizzle": 1,
            "thunderstorm with drizzle": 1,
            "thunderstorm with light drizzle": 1
        },
        "atmospheric": {
            "mist": 1,
            "fog": 2,
            "haze": 2,
            "smoke": 1,
            "dust": 1,
            "squalls": 3
        }
    }

    def encode_phrase(phrase):
        encoding = [
            categories["clear_clouds"].get(phrase, 0),  
            categories["rain_drizzle"].get(phrase, 0),  
            categories["snow_sleet"].get(phrase, 0),    
            categories["thunderstorms"].get(phrase, 0),
            categories["atmospheric"].get(phrase, 0)   
        ]
        return encoding

    weather_description = np.zeros((arr.shape[1], c, len(weather_categories)))
    for i in range(c):
        for j in range(arr.shape[1]):
            weather_description[j][i] = np.array(encode_phrase(arr[string_index][j][i]))

    return weather_description, weather_categories