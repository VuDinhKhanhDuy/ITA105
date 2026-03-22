import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
#Bai 1
df_housing = pd.read_csv('ITA105_Lab_2_Housing.csv')
print(df_housing.describe())
sns.boxplot(data=df_housing[['dien_tich', 'gia']])
plt.show()
Q1 = df_housing.quantile(0.25)
Q3 = df_housing.quantile(0.75)
IQR = Q3 - Q1
outliers_iqr = df_housing[((df_housing < (Q1 - 1.5 * IQR)) | (df_housing > (Q3 + 1.5 * IQR))).any(axis=1)]
z_scores = np.abs(stats.zscore(df_housing.select_dtypes(include=[np.number])))
outliers_z = df_housing[(z_scores > 3).any(axis=1)]
df_housing_cleaned = df_housing[(df_housing['dien_tich'] > 0) & (df_housing['gia'] > 1)]
df_housing_cleaned['gia_log'] = np.log1p(df_housing_cleaned['gia']) 

#Bai 2
df_iot = pd.read_csv('ITA105_Lab_2_Iot.csv')
df_iot['timestamp'] = pd.to_datetime(df_iot['timestamp'])
df_iot.set_index('timestamp', inplace=True)
window = 10
rolling_mean = df_iot['temperature'].rolling(window=window).mean()
rolling_std = df_iot['temperature'].rolling(window=window).std()
upper_bond = rolling_mean + (3 * rolling_std)
lower_bond = rolling_mean - (3 * rolling_std)
outliers_rolling = df_iot[(df_iot['temperature'] > upper_bond) | (df_iot['temperature'] < lower_bond)]
df_iot['temperature'] = df_iot['temperature'].mask(df_iot['temperature'] > 40).interpolate()

#Bai 3
df_eco = pd.read_csv('ITA105_Lab_2_Ecommerce.csv')
sns.scatterplot(x='quantity', y='price', data=df_eco)
plt.title("Price vs Quantity Outliers")
plt.show()
df_eco_cleaned = df_eco[(df_eco['rating'] <= 5) & (df_eco['price'] > 0)]
df_eco_cleaned['price'] = df_eco_cleaned['price'].clip(upper=df_eco_cleaned['price'].quantile(0.99))

#Bai 4
pd.plotting.scatter_matrix(df_housing[['dien_tich', 'gia', 'so_phong']], figsize=(10, 10))
z_area = np.abs(stats.zscore(df_housing['dien_tich']))
z_price = np.abs(stats.zscore(df_housing['gia']))
multivariate_outliers = df_housing[(z_area > 2) & (z_price > 2)] 
