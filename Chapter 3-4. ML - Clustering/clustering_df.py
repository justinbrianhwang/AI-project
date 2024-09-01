## clustering_df.py

def p(str):
    print(str, '\n')

# Load necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings(action='ignore')

# Load rent.csv data
data = pd.read_csv('./assets/rent.csv')
#data.info()

# rent.csv variables
'''
Posted On: Date posted
BHK: Number of bedrooms, hall, and kitchen
Rent: Rental price
Size: Size of the property
Floor: Floor number
Area Type: Type of the area where the house is located
Area Locality: Locality of the area
City: Name of the city
Furnishing Status: Furnishing status of the house or apartment
Tenant Preferred: Type of preferred tenant
Bathroom: Number of bathrooms 
Point of Contact: Contact person 
'''

# Analysis of categorical variables
# p(data['Floor'].value_counts()) # Check how many of each value exist.
# p(data['Area Type'].value_counts())
# p(data['Area Locality'].value_counts())
# p(data['City'].value_counts())
# p(data['Furnishing Status'].value_counts())
# p(data['Point of Contact'].value_counts())

# Renaming columns -> Not changed yet, just mapped.
new_column_name = {
    "Posted On":"Posted_On",
    "BHK":"BHK",
    "Rent":"Rent",
    "Size":"Size",
    "Floor" : "Floor",
    "Area Type" : "Area_Type",
    "Area Locality" : "Area_Locality",
    "City":"City",
    "Furnishing Status":"Furnishing_Status",
    "Tenant Preferred":"Tenant_Preferred",
    "Bathroom":"Bathroom",
    "Point of Contact":"Point_of_Contact"
}
data.rename(columns=new_column_name, inplace=True)

# Sort BHK values in ascending order
data['BHK'].sort_values()

# Check Rent
p(data['Rent'].value_counts())
p(data['Rent'].sort_values())
# Outliers
# Data points that are significantly different from other observations in the dataset.
# -> Think of it as a loner.

# Rent boxplot
# plt.figure(figsize=(8, 6))
# sns.boxplot(x=data['Rent'])
# plt.show()
# Most of the data lies within the range, but one data point is at 3.5. This shows that it's quite isolated.

# Rent Scatter
# plt.figure(figsize=(8, 6))
# sns.scatterplot(x=data.index, y=data['Rent'])
# plt.show()
# Similarly, outliers can be observed.
# When checking for outliers, use Boxplot or Scatterplot.

# Correlation between BHK and Rent
corr_Br = data['BHK'].corr(data['Rent'])
# p(f"Correlation between BHK and Rent: {corr_Br:.2f}")
# The correlation is 0.37.

# Scatter plot visualization
# plt.scatter(data['BHK'], data['Rent'])
# plt.grid(True)
# plt.show()

# Check size
# p(data['Size'].value_counts())
# p(data['Size'].sort_values())

# Size distribution plot
# sns.displot(data['Size'])
# plt.show()

# Relationship between Size and Rent
# plt.scatter(data['Size'], data['Rent'])
# plt.show()
# Since it's not easy to directly analyze data, we use graphs for visualization.

# Check correlations
# 1. Correlation between Rent and BHK
p(f"Correlation between BHK and Rent: {data['BHK'].corr(data['Rent'])}")

# 2. Correlation between Rent and Size
p(f"Correlation between Size and Rent: {data['Size'].corr(data['Rent'])}")

# 3. Correlation between Rent and City
# Since the city is a string, it needs to be converted to a numerical format.
cities = data['City'].unique() # Unique city values
# p(cities) -> Grouped by unique values
for city in cities:
    city_data = data[data['City'] == city]
# Average rent by city after grouping by city
city_mean = data.groupby('City')['Rent'].mean()
# p(city_mean)

data['City_Mean'] = data.groupby('City')['Rent'].transform('mean')
# p(data['City_Mean'])

# Check correlation
# p(f"Correlation between Rent and city average rent: {data['Rent'].corr(data['City_Mean'])}")

# Correlation between City and Rent group with Rent
rent_city = data.groupby('City')['Rent'].corr(data['Rent'])
# p(rent_city)

# Get list of cities
cities = data['City'].unique()
# p(cities)

# Correlation of rent by city
city_rent_corr = {}
for i in cities:
    city_data = data[data['City'] == i]
    correlation = city_data['Rent'].corr(city_data['Rent'])
    city_rent_corr[i] = correlation

# p(city_rent_corr)

# Create a heatmap with numerical data

# Select only numerical variables
# numeric_data = data.select_dtypes(include=['int64', 'float64'])
# numeric_data.corr()
# plt.figure(figsize=(10,8))
# sns.heatmap(numeric_data.corr(), annot=True)
# plt.show()

# Rent by city
# plt.figure(figsize = (10,6))
# sns.boxplot(x='City', y='Rent', data=data)
# plt.grid(True)
# plt.show()

# Calculate average rent
avg_rent_city = data.groupby('City')['Rent'].mean().sort_values(ascending=False)
p(avg_rent_city)

# Convert date data
data['Posted_On'] = pd.to_datetime(data['Posted_On'])
data['Year'] = data['Posted_On'].dt.year
data['Month'] = data['Posted_On'].dt.month
# data['Day'] = data['Posted_On'].dt.day

# p(data['Year'].value_counts())
# p(data['Month'].value_counts())

# Monthly average rent
avg_month_rent = data.groupby(['Year', 'Month'])['Rent'].mean()
# p(avg_month_rent)

# Visualize monthly average rent
# plt.figure(figsize=(12, 6))
# avg_month_rent.plot(kind='line', marker='o')
# plt.grid(True)
# plt.show()

# Model selection
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Select necessary columns
features = ['BHK', 'Size', 'Floor', 'Bathroom']
data1 = data[features + ['Rent']]
p(data1)

# Data preprocessing for Floor column: Extract numbers from strings and convert to float
data1['Floor'] = data1['Floor'].str.extract('(\d+)').astype(float)
# p(data1['Floor'])

# Handle missing values
data1 = data1.dropna() # Drop rows with missing values
data1.info()

# Split data into training and test sets
X = data1[features]
y = data1['Rent']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create linear regression model
lr = LinearRegression()

# Train the model
lr.fit(X_train, y_train)

# Predicted values
pred = lr.predict(X_test)

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, pred)
p(f"Mean Squared Error: {mse}") # 2542273917.555011

# Visualize actual vs predicted values
plt.figure(figsize=(12, 6))
plt.scatter(y_test, pred)
plt.plot(
    [min(y_test), max(y_test)],
    [min(y_test), max(y_test)],
    color='red',
    linestyle='--'
)
plt.grid(True)
plt.show()
