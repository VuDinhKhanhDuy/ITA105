import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer, PowerTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate

warnings.filterwarnings('ignore')

class OutlierClipper(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        X_array = np.array(X)
        Q1 = np.percentile(X_array, 25, axis=0)
        Q3 = np.percentile(X_array, 75, axis=0)
        IQR = Q3 - Q1
        self.lower_bound_ = Q1 - 1.5 * IQR
        self.upper_bound_ = Q3 + 1.5 * IQR
        return self
    
    def transform(self, X):
        return np.clip(X, self.lower_bound_, self.upper_bound_)

    def get_feature_names_out(self, input_features=None):
        return input_features

def date_features_extractor(df_date_col):
    dates = pd.to_datetime(df_date_col.iloc[:, 0], errors='coerce')
    return np.c_[dates.dt.month, dates.dt.quarter, dates.dt.year]

def get_date_feature_names(transformer, input_features):
    return ["month", "quarter", "year"]

df = pd.read_csv('ITA105_Lab_8.csv')
num_cols = ['LotArea', 'Rooms', 'NoiseFeature']
cat_cols = ['HasGarage', 'Neighborhood', 'Condition']
text_col = 'Description'
date_col = 'SaleDate'

num_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('outlier', OutlierClipper()),
    ('scaler', StandardScaler()),
    ('power', PowerTransformer(method='yeo-johnson'))
])

cat_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

text_pipe = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_features=10))
])

date_pipe = Pipeline([
    ('extract', FunctionTransformer(date_features_extractor, feature_names_out=get_date_feature_names)),
    ('imputer', SimpleImputer(strategy='most_frequent'))
])

preprocessor = ColumnTransformer([
    ('num', num_pipe, num_cols),
    ('cat', cat_pipe, cat_cols),
    ('text', text_pipe, text_col),
    ('date', date_pipe, [date_col])
])

X_demo = df.drop('SalePrice', axis=1).head(10)
preprocessor.fit(X_demo)
print("--- BÀI 1 ---")
print(f"Shape: {preprocessor.transform(X_demo).shape}")
print(f"Features: \n{preprocessor.get_feature_names_out()}\n")

print("--- BÀI 2 ---")
test_cases = {
    "Full": df.head(5).copy(),
    "Missing": df.head(5).copy(),
    "Skewed": df.head(5).copy(),
    "Unseen": df.head(5).copy()
}

test_cases["Missing"][num_cols] = test_cases["Missing"][num_cols].astype(float)
test_cases["Missing"].iloc[0:3, 0:3] = np.nan
test_cases["Skewed"]['LotArea'] = 999999
test_cases["Unseen"].at[0, 'Neighborhood'] = 'Mars'

for name, data in test_cases.items():
    try:
        X_t = data.drop('SalePrice', axis=1, errors='ignore')
        out = preprocessor.transform(X_t)
        print(f"OK: {name} | Shape: {out.shape}")
    except Exception as e:
        print(f"ERR: {name} | {e}")

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
sns.histplot(df['LotArea'], kde=True).set_title("Before")
plt.subplot(1, 2, 2)
X_p = preprocessor.fit_transform(df.drop('SalePrice', axis=1))
sns.histplot(X_p[:, 0], kde=True).set_title("After")
plt.show()

print("\n--- BÀI 3 ---")
X = df.drop('SalePrice', axis=1)
y = df['SalePrice']

full_pipe = Pipeline([
    ('pre', preprocessor),
    ('model', RandomForestRegressor(n_estimators=100, random_state=42))
])

cv = cross_validate(full_pipe, X, y, cv=5, scoring=['neg_mean_absolute_error', 'r2'])
print(f"MAE: {-cv['test_neg_mean_absolute_error'].mean():.2f}")
print(f"R2: {cv['test_r2'].mean():.4f}")

print("\n--- BÀI 4 ---")
full_pipe.fit(X, y)
joblib.dump(full_pipe, 'house_model.pkl')

def predict_price(data):
    m = joblib.load('house_model.pkl')
    return m.predict(data)

new_h = pd.DataFrame({
    'LotArea': [5000], 'Rooms': [3], 'HasGarage': [1], 'NoiseFeature': [0],
    'Neighborhood': ['A'], 'Condition': ['Good'], 
    'Description': ['nice house'], 'SaleDate': ['2026-01-01'], 'ImagePath': ['none']
})

print(f"Predict: {predict_price(new_h)[0]:.2f}")
