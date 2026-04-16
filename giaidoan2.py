import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

def create_sample_data():
    data = {
        'price': [2500, 3000, -500, 2800, 15000, 2500, 3100],
        'area': [50, 60, 45, 55, 300, 50, 62],
        'rooms': [2, 3, 0, 2, 10, 2, 3],
        'location': ['Hanoi', 'Hanoi', 'HCM', 'Danang', 'Hanoi', 'Hanoi', 'HCM'],
        'description': [
            'Nha dep mat pho', 
            'Can ho chung cu cao cap', 
            'Nha xau gia re', 
            'Dat nen tho cu', 
            'Biet thu xa hoa bac nhat', 
            'Nha dep mat pho', 
            'Can ho gan trung tam'
        ]
    }
    df = pd.DataFrame(data)
    df.to_csv('data.csv', index=False)

class PropTechDataProcessor:
    def __init__(self, filepath):
        self.df = pd.read_csv(filepath)
        self.tfidf_matrix = None

    def step_1_exploration(self):
        print(self.df.describe())
        print(self.df.isnull().sum())
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        sns.histplot(self.df['price'], kde=True)
        plt.subplot(1, 3, 2)
        sns.boxplot(y=self.df['area'])
        plt.subplot(1, 3, 3)
        sns.violinplot(x='rooms', y='price', data=self.df)
        plt.show()

    def step_2_cleaning(self):
        self.df['price'] = self.df['price'].fillna(self.df['price'].median())
        self.df['location'] = self.df['location'].fillna(self.df['location'].mode()[0])
        self.df = self.df[(self.df['price'] > 0) & (self.df['rooms'] > 0)]
        self.df.drop_duplicates(inplace=True)

    def step_3_outliers(self):
        Q1 = self.df['price'].quantile(0.25)
        Q3 = self.df['price'].quantile(0.75)
        IQR = Q3 - Q1
        upper_bound = Q3 + 1.5 * IQR
        self.df['price'] = np.where(self.df['price'] > upper_bound, upper_bound, self.df['price'])

    def step_4_transformation(self):
        scaler = StandardScaler()
        self.df['area_scaled'] = scaler.fit_transform(self.df[['area']])
        if 'location' in self.df.columns:
            self.df = pd.get_dummies(self.df, columns=['location'])
        tfidf = TfidfVectorizer(max_features=100)
        self.tfidf_matrix = tfidf.fit_transform(self.df['description'].fillna(''))

    def step_5_text_similarity(self):
        if self.tfidf_matrix is not None:
            sim_matrix = cosine_similarity(self.tfidf_matrix)
            duplicates = np.where(sim_matrix > 0.9)
            pairs = [(i, j) for i, j in zip(*duplicates) if i < j]
            for p in pairs:
                print(f"Duplicate suggest: {p[0]} & {p[1]}")

    def run_stage_1(self):
        self.step_1_exploration()
        self.step_2_cleaning()
        self.step_3_outliers()
        self.step_4_transformation()
        self.step_5_text_similarity()
        return self.df

if __name__ == "__main__":
    if not os.path.exists('data.csv'):
        create_sample_data()
    processor = PropTechDataProcessor('data.csv')
    final_df = processor.run_stage_1()
    print(final_df.head())