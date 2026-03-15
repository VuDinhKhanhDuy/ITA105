import pandas as pd
import matplotlib.pyplot as plt
# Bài 1
df = pd.read_csv(r'D:\Lab1\Lab1\ITA105_Lab_1.csv')
#A kích thước dữ liệu
print("Kích thước dữ liệu:", df.shape)
#B Thống kê mô tả
print("\nThống kê mô tả:\n", df.describe())
#C dữ liệu thiếu
print("\nDữ liệu thiếu:\n", df.isnull().sum())

#Bài 2
df['Price'] = df['Price'].fillna(df['Price'].mean())
df['StockQuantity'] = df['StockQuantity'].fillna(df['StockQuantity'].median())
df['Category'] = df['Category'].fillna(df['Category'].mode())
print("\nDữ liệu sau khi xử lý thiếu:\n", df.isnull().sum())
df_temp = pd.read_csv(r'D:\Lab1\Lab1\ITA105_Lab_1.csv')
print("\nDữ liệu gốc:\n", df_temp)
print("\nDữ liệu nếu dùng dropna:\n", df_temp.dropna())

#Bài 3
df = df[df["Price"] > 0]
df = df[df["StockQuantity"] >= 0]
df = df[(df["Rating"] >= 0) & (df["Rating"] <= 5)]
print("\nDữ liệu sau khi làm sạch:\n", df)

#Bài 4
df["Price_smooth"] = df["Price"].rolling(3).mean()

plt.figure()
plt.plot(df["Price"], label="Original Price")
plt.plot(df["Price_smooth"], label="Smoothed Price")
plt.title("Price Moving Average")
plt.legend()
plt.show()

#Bài 5
df['Category'] = df['Category'].str.lower()
df['Description'] = df['Description'].str.replace(r'[?!.]', '', regex=True).str.strip()
df['Price_VND'] = df['Price'] * 25000
print("\nDữ liệu sau khi chuẩn hóa (5 dòng đầu):")
print(df[['ProductID', 'Category', 'Price_VND', 'Description']].head())