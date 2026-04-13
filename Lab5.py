import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
import os
import glob

plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (12, 6)


base_path = os.path.dirname(os.path.abspath(__file__))

def get_data_file(keyword):
    files = glob.glob(os.path.join(base_path, f"*{keyword}*"))
    if not files:
        print(f" Không tìm thấy file nào chứa từ khóa: {keyword}")
        return None
    return files[0]

file1 = get_data_file('Supermarket')
if file1:
    df1 = pd.read_csv(file1)
    df1['date'] = pd.to_datetime(df1['date'])
    df1.set_index('date', inplace=True)
    df1['revenue'] = df1['revenue'].ffill()
    
    # Tạo đặc trưng
    df1['year'] = df1.index.year
    df1['month'] = df1.index.month
    
    
    df1.resample('ME')['revenue'].sum().plot(kind='line', marker='o', title='Total Monthly Revenue')
    plt.show()

file2 = get_data_file('Web_traffic')
if file2:
    df2 = pd.read_csv(file2)
    df2['datetime'] = pd.to_datetime(df2['datetime'])
    df2.set_index('datetime', inplace=True)
    df2 = df2.asfreq('h')
    df2['visits'] = df2['visits'].interpolate(method='linear')
    
    df2['hour'] = df2.index.hour
    sns.lineplot(data=df2, x='hour', y='visits', errorbar=None)
    plt.title('Hourly Web Traffic')
    plt.show()


file3 = get_data_file('Stock')
if file3:
    df3 = pd.read_csv(file3)
    df3['date'] = pd.to_datetime(df3['date'])
    df3.set_index('date', inplace=True)
    df3['close_price'] = df3['close_price'].ffill()

    df3['MA7'] = df3['close_price'].rolling(window=7).mean()
    df3['MA30'] = df3['close_price'].rolling(window=30).mean()
    
    
    df3[['close_price', 'MA7', 'MA30']].plot(title='Stock Price Trends')
    plt.show()


file4 = get_data_file('Production')
if file4:
    df4 = pd.read_csv(file4)
    df4['week_start'] = pd.to_datetime(df4['week_start'])
    df4.set_index('week_start', inplace=True)
    df4['production'] = df4['production'].ffill()

    result = seasonal_decompose(df4['production'], model='additive', period=52)
    result.plot()
    plt.show()