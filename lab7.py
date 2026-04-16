import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PowerTransformer

df = pd.read_csv('ITA105_Lab_7.csv')
numeric_cols = df.select_dtypes(include=[np.number]).columns


print("\n" + "="*50 + "\nBÀI 1: PHÂN TÍCH SKEWNESS\n" + "="*50)
skewness = df[numeric_cols].skew().abs().sort_values(ascending=False)
print("Top 10 cột lệch nhất:\n", df[numeric_cols].skew().loc[skewness.head(10).index])

top_3 = skewness.head(3).index
plt.figure(figsize=(15, 4))
for i, col in enumerate(top_3):
    plt.subplot(1, 3, i+1)
    sns.histplot(df[col], kde=True, color='teal')
    plt.title(f"{col}\nSkew: {df[col].skew():.2f}")
plt.tight_layout()
plt.show()

print("\n[Lý thuyết Bài 1]:")
print("- Xu hướng: Dữ liệu lệch phải (Positive Skew) do Outliers như nhà siêu sang kéo đuôi biểu đồ.")
print("- Tác động: Làm sai lệch đường hồi quy, khiến mô hình dự báo không chuẩn xác.")


print("\n" + "="*50 + "\nBÀI 2: BIẾN ĐỔI DỮ LIỆU\n" + "="*50)
c1, c2, c3 = 'SalePrice', 'LotArea', 'NegSkewIncome'
pt = PowerTransformer(method='yeo-johnson')

res = []
for col in [c1, c2, c3]:
    orig = df[col].skew()
    log_v = np.log(df[col]) if df[col].min() > 0 else np.nan
    bc_v, lmb = stats.boxcox(df[col]) if df[col].min() > 0 else (np.nan, "N/A")
    yj_v = pt.fit_transform(df[[col]]).flatten()
    res.append({'Cột': col, 'Gốc': orig, 'Sau Log': pd.Series(log_v).skew(), 
                'Sau Box-Cox': pd.Series(bc_v).skew(), 'Sau Power(YJ)': pd.Series(yj_v).skew()})

print(pd.DataFrame(res))
print("\n[Lý thuyết Bài 2]:")
print("- Tốt nhất: Yeo-Johnson/Box-Cox vì tìm được Lambda tối ưu để đưa về phân phối chuẩn.")
print("- Lambda (λ): Chỉ số xác định mức độ biến đổi (0 là Log, 0.5 là căn bậc hai).")


print("\n" + "="*50 + "\nBÀI 3: HUẤN LUYỆN MÔ HÌNH\n" + "="*50)
X = df[['LotArea', 'HouseAge', 'Rooms', 'MixedFeature']]
y = df['SalePrice']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

m_a = LinearRegression().fit(X_train, y_train)
m_b = LinearRegression().fit(X_train, np.log(y_train))
pt_x, pt_y = PowerTransformer(), PowerTransformer()
m_c = LinearRegression().fit(pt_x.fit_transform(X_train), pt_y.fit_transform(y_train.values.reshape(-1,1)))

rmse_a = np.sqrt(mean_squared_error(y_test, m_a.predict(X_test)))
rmse_b = np.sqrt(mean_squared_error(y_test, np.exp(m_b.predict(X_test))))
p_c = pt_y.inverse_transform(m_c.predict(pt_x.transform(X_test)))
rmse_c = np.sqrt(mean_squared_error(y_test, p_c))

print(f"RMSE Ver A (Gốc): {rmse_a:.2f} | Ver B (Log): {rmse_b:.2f} | Ver C (Power): {rmse_c:.2f}")


print("\n" + "="*50 + "\nBÀI 4: INSIGHT NGHIỆP VỤ\n" + "="*50)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1); sns.scatterplot(data=df, x='LotArea', y='SalePrice'); plt.title("Trước Transform")
plt.subplot(1, 2, 2); sns.scatterplot(x=np.log(df['LotArea']), y=np.log(df['SalePrice']), color='red'); plt.title("Sau Transform")
plt.show()

print("\n[Lý thuyết Bài 4]:")
print("- Tại sao cần: Loại bỏ nhiễu từ nhà siêu sang, tập trung vào xu hướng nhà ở phổ thông.")
print("- Khuyến nghị: Dùng dữ liệu Transform để định giá nhà tự động chính xác cho số đông khách hàng.")