import matplotlib.pyplot as plt
import pandas as pd

# Read data from file
file_path = '../data/city_attributes.csv'
data = pd.read_csv(file_path)

# Extract latitude and longitude
latitudes = data['Latitude']
longitudes = data['Longitude']
city_names = data['City']

# Create the plot
plt.figure(figsize=(10, 6))
plt.scatter(longitudes, latitudes, c='red', marker='o', edgecolor='black', alpha=0.7)

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
