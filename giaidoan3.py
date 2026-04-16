import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def create_sample_data(filename='data.csv'):
    data = {
        'price': [2500, 3000, -500, 2800, 15000, 2500, 3100, 4200, 5000, 2200, 3500, 6000, 4800, 7200, 5500],
        'area': [50, 60, 45, 55, 300, 50, 62, 80, 95, 48, 70, 110, 85, 120, 90],
        'rooms': [2, 3, 0, 2, 10, 2, 3, 3, 4, 2, 3, 4, 3, 5, 4],
        'location': ['Hanoi', 'Hanoi', 'HCM', 'Danang', 'Hanoi', 'Hanoi', 'HCM', 'HCM', 'Hanoi', 'Danang', 'HCM', 'Hanoi', 'Danang', 'Hanoi', 'HCM'],
        'description': [
            'Nha dep mat pho', 'Can ho cao cap', 'Nha gia re', 'Dat nen tho cu', 
            'Biet thu xa hoa', 'Nha dep mat pho', 'Can ho trung tam', 'Nha pho hien dai',
            'Biet thu san vuon', 'Nha cap 4', 'Chung cu mini', 'Mat bang kinh doanh',
            'Nha gan bien', 'Biet thu don lap', 'Can ho studio'
        ]
    }
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)

class PropTechAIPipeline:
    def __init__(self, filepath):
        self.df = pd.read_csv(filepath)
        self.tfidf_matrix = None
        self.best_pipeline = None

    def stage_1_cleansing(self):
        self.df['price'] = self.df['price'].fillna(self.df['price'].median())
        self.df['location'] = self.df['location'].fillna(self.df['location'].mode()[0])
        self.df = self.df[(self.df['price'] > 0) & (self.df['rooms'] > 0) & (self.df['area'] > 0)]
        self.df.drop_duplicates(inplace=True)
        
        Q1 = self.df['price'].quantile(0.25)
        Q3 = self.df['price'].quantile(0.75)
        IQR = Q3 - Q1
        upper_bound = Q3 + 1.5 * IQR
        self.df['price'] = np.where(self.df['price'] > upper_bound, upper_bound, self.df['price'])
        
        tfidf = TfidfVectorizer(max_features=50)
        self.tfidf_matrix = tfidf.fit_transform(self.df['description'].fillna(''))

    def stage_2_and_completion_engineering(self):
        self.df['price_per_m2'] = self.df['price'] / self.df['area']
        self.df['area_room_interaction'] = self.df['area'] * self.df['rooms']
        self.df['log_price'] = np.log1p(self.df['price'])
        return self.df

    def stage_2_3_modeling(self):
        X = self.df[['area', 'rooms', 'location', 'area_room_interaction']]
        y = self.df['log_price']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        numeric_features = ['area', 'rooms', 'area_room_interaction']
        categorical_features = ['location']

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', RobustScaler(), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ])

        models = {
            "Linear": LinearRegression(),
            "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
            "GradientBoosting": GradientBoostingRegressor(random_state=42)
        }

        print("--- KẾT QUẢ HUẤN LUYỆN & SO SÁNH ---")
        for name, model in models.items():
            pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model)])
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            print(f"{name} -> R2: {r2:.4f}, RMSE: {rmse:.4f}")
            if name == "RandomForest":
                self.best_pipeline = pipeline

    def stage_4_5_kpi_visual(self):
        print("\n--- PHÂN TÍCH KPI & INSIGHT ---")
        threshold = self.df['price'].quantile(0.95)
        luxury_count = len(self.df[self.df['price'] >= threshold])
        print(f"Luxury Segment (Top 5%): {luxury_count} ban ghi")
        
        if self.tfidf_matrix is not None:
            sim_matrix = cosine_similarity(self.tfidf_matrix)
            duplicates = np.where(sim_matrix > 0.9)
            pairs = [(i, j) for i, j in zip(*duplicates) if i < j]
            print(f"Phat hien {len(pairs)} tin dang trung lap noi dung.")

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        sns.histplot(self.df['price'], kde=True, color='blue')
        plt.title("Phan phoi gia nha (Raw vs Capped)")
        plt.subplot(1, 2, 2)
        sns.scatterplot(x='area', y='price', hue='location', data=self.df)
        plt.title("Tuong quan Dien tich - Gia theo khu vuc")
        plt.show()

    def completion_test_unseen(self):
        print("\n--- GIAI DOAN HOAN THIEN: KIEM THU VOI DU LIEU MOI ---")
        new_data = pd.DataFrame({
            'area': [75, 150],
            'rooms': [3, 5],
            'location': ['Hai Phong', 'Hanoi'], 
            'description': ['Nha moi xay', 'Biet thu pho']
        })
        new_data['area_room_interaction'] = new_data['area'] * new_data['rooms']
        
        if self.best_pipeline:
            predictions = self.best_pipeline.predict(new_data)
            final_prices = np.expm1(predictions)
            print("Du bao gia cho du lieu moi (bao gom Unseen Location):")
            for i, price in enumerate(final_prices):
                print(f"Nha {i+1}: {price:.2f}")

    def run(self):
        self.stage_1_cleansing()
        self.stage_2_and_completion_engineering()
        self.stage_2_3_modeling()
        self.stage_4_5_kpi_visual()
        self.completion_test_unseen()

if __name__ == "__main__":
    if not os.path.exists('data.csv'):
        create_sample_data()
    
    app = PropTechAIPipeline('data.csv')
    app.run()