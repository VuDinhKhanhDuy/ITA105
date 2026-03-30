import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error




np.random.seed(42)
n_samples = 100
#Dữ liệu mẫu
data = {
    'gia': np.random.normal(5000, 2000, n_samples).tolist() + [-100, 50000, 5000], 
    'dien_tich': np.random.normal(70, 20, n_samples).tolist() + [80, 600, 80], 
    'so_phong': np.random.randint(1, 5, n_samples).tolist() + [0, 15, 2],
    'khu_vuc': np.random.choice(['Hà Nội', 'TP.HCM', 'Đà Nẵng'], n_samples + 3).tolist(),
    'tinh_trang': np.random.choice(['Mới', 'Cũ'], n_samples + 3).tolist(),
    'ngay': pd.date_range(start='2025-01-01', periods=n_samples + 3).strftime('%Y-%m-%d').tolist(),
    'mo_ta': np.random.choice(['Căn hộ cao cấp view hồ', 'Nhà giá rẻ trung tâm', 'Biệt thự luxury'], n_samples + 3).tolist()
}

df = pd.DataFrame(data)
df.loc[0:5, 'dien_tich'] = np.nan 


# GIAI ĐOẠN 1: KHÁM PHÁ & LÀM SẠCH

print("--- Thống kê ban đầu ---")
print(df.describe())
df = df.drop_duplicates()

df = df[(df['gia'] > 0) & (df['so_phong'] > 0)]
df['dien_tich'] = df['dien_tich'].fillna(df['dien_tich'].median())

Q1, Q3 = df['gia'].quantile(0.25), df['gia'].quantile(0.75)
IQR = Q3 - Q1
df = df[~((df['gia'] < (Q1 - 1.5 * IQR)) | (df['gia'] > (Q3 + 1.5 * IQR)))]

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1); sns.histplot(df['gia'], kde=True); plt.title('Phân phối Giá')
plt.subplot(1, 2, 2); sns.boxplot(x=df['gia']); plt.title('Boxplot Giá')
plt.tight_layout()
plt.show()


# GIAI ĐOẠN 2: FEATURE ENGINEERING & PIPELINE

df['log_gia'] = np.log1p(df['gia'])
df['ngay'] = pd.to_datetime(df['ngay'])
df['thang'] = df['ngay'].dt.month
df['luxury_score'] = df['mo_ta'].str.contains('luxury|cao cấp', case=False).astype(int)

X = df[['dien_tich', 'so_phong', 'thang', 'luxury_score', 'khu_vuc', 'tinh_trang', 'mo_ta']]
y = df['log_gia']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

num_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('yeo', PowerTransformer(method='yeo-johnson')),
    ('scaler', StandardScaler())
])

cat_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', num_pipe, ['dien_tich', 'so_phong', 'thang', 'luxury_score']),
    ('cat', cat_pipe, ['khu_vuc', 'tinh_trang']),
    ('text', TfidfVectorizer(max_features=50), 'mo_ta')
])

full_pipeline = Pipeline([
    ('pre', preprocessor),
    ('model', XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42))
])

full_pipeline.fit(X_train, y_train)

# ĐÁNH GIÁ & KPI

y_pred = full_pipeline.predict(X_test)
print(f"\n--- ĐÁNH GIÁ MÔ HÌNH ---")
print(f"R2 Score: {r2_score(y_test, y_pred):.4f}")
print(f"MAE (Giá gốc): {mean_absolute_error(np.expm1(y_test), np.expm1(y_pred)):.2f}")

df['gia_du_bao'] = np.expm1(full_pipeline.predict(X))
df['gia_m2'] = df['gia'] / df['dien_tich']

print("\n--- KẾT QUẢ ---")
print(df[['gia', 'gia_du_bao', 'gia_m2']].head())