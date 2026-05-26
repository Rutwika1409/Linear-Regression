import pandas as pd
import numpy as np

# Loading Dataset
df = pd.read_csv("data/BigMart Sales.csv")

x = df.head(10)

# Displaying the first 10 rows of the dataset
print(x)

# Identify Missing Data 
df.replace("?", np.nan, inplace=True)

# Handling Missing Values Based on Column meaning
# Missing values in Item_weight and Item_Weight and Outlet_Size

# Imputing missing values in Item_Weight with mean
df['Item_Weight'] = df['Item_Weight'].fillna(
    df['Item_Weight'].mean()
)

# Since Outlet_Size is a categorical variable, we can impute missing values with the mode
df['Outlet_Size'] = df['Outlet_Size'].fillna(
    df['Outlet_Size'].mode()[0]
)

# Checking for duplicates
duplicates = df.duplicated().sum()

# Remove duplicates
df.drop_duplicates(inplace=True)

# Save the cleaned dataset in data folder
df.to_csv(
    "data/BigMart_Sales_Cleaned.csv",
    index=False
)

print("Data Cleaning Completed Successfully")