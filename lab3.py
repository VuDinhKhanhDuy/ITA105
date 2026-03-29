import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
#Bài 1
df1 = pd.read_csv('ITA105_Lab_3_Sports.csv')
print(df1.info())
print(df1.describe())

# Trực quan hóa biến 'chieu_cao_cm'
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1); sns.histplot(df1['chieu_cao_cm'], kde=True); plt.title('Histogram Chiều cao')
plt.subplot(1, 2, 2); sns.boxplot(x=df1['chieu_cao_cm']); plt.title('Boxplot Chiều cao')
plt.show()

# Chuẩn hóa
scaler_minmax = MinMaxScaler()
scaler_std = StandardScaler()
df1['chieu_cao_minmax'] = scaler_minmax.fit_transform(df1[['chieu_cao_cm']])
df1['chieu_cao_zscore'] = scaler_std.fit_transform(df1[['chieu_cao_cm']])

#Bài 2
df2 = pd.read_csv('ITA105_Lab_3_Health.csv')

# Trực quan hóa để tìm ngoại lệ
plt.figure(figsize=(15, 5))
columns = ['BMI', 'huyet_ap_mmHg', 'cholesterol_mg_dl']
for i, col in enumerate(columns):
    plt.subplot(1, 3, i+1)
    sns.boxplot(y=df2[col])
    plt.title(f'Ngoại lệ của {col}')
plt.show()

# Chuẩn hóa
df2_scaled = df2.copy()
df2_scaled[columns] = scaler_std.fit_transform(df2[columns]) # Ví dụ Z-score

# COMMAND / NHẬN XÉT:
# 1. Biến bị ảnh hưởng nhiều bởi ngoại lệ: Biến 'huyet_ap_mmHg' và 'BMI' thường chứa các giá trị 
#    cực đoan (ví dụ: huyết áp quá cao hoặc BMI vượt ngưỡng béo phì độ 3). 
#    Trong Boxplot, các dấu chấm nằm ngoài râu nến chính là các ngoại lệ này.
# 2. Phương pháp chuẩn hóa phù hợp: Z-Score Normalization phù hợp hơn. 
#    Vì Min-Max Scaling sẽ bị các giá trị ngoại lệ kéo giãn khoảng cách (ví dụ 1 người huyết áp 300 
#    sẽ làm tất cả người bình thường bị nén lại gần mức 0), trong khi Z-Score dựa trên độ lệch 
#    chuẩn nên giữ được đặc điểm phân phối của dữ liệu tốt hơn khi có nhiễu.

#Bài 3
df3 = pd.read_csv('ITA105_Lab_3_Finance.csv')

# Vẽ scatterplot trước khi chuẩn hóa
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
sns.scatterplot(data=df3, x='doanh_thu_musd', y='loi_nhuan_musd')
plt.title('Trước chuẩn hóa')

# Chuẩn hóa Min-Max
df3[['doanh_thu_scaled', 'loi_nhuan_scaled']] = scaler_minmax.fit_transform(df3[['doanh_thu_musd', 'loi_nhuan_musd']])

plt.subplot(1, 2, 2)
sns.scatterplot(data=df3, x='doanh_thu_scaled', y='loi_nhuan_scaled')
plt.title('Sau Min-Max Scaling')
plt.show()

#Bài 4
df4 = pd.read_csv('ITA105_Lab_3_Gaming.csv')

# Kiểm tra missing values
print("Giá trị thiếu:\n", df4.isnull().sum())

# Chuẩn hóa và so sánh phân phối
df4['gio_choi_zscore'] = scaler_std.fit_transform(df4[['gio_choi']])

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1); sns.histplot(df4['gio_choi'], color='blue'); plt.title('Giờ chơi gốc')
plt.subplot(1, 2, 2); sns.histplot(df4['gio_choi_zscore'], color='green'); plt.title('Giờ chơi (Z-score)')
plt.show()