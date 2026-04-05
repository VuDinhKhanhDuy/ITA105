import pandas as pd
import numpy as np
import os
import nltk
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize


try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def process_lab_task(file_name, text_col, cat_cols, target_word, task_name, backup_data):
    print(f"\n{'='*15} {task_name.upper()} {'='*15}")
    
    
    if os.path.exists(file_name):
        df = pd.read_csv(file_name)
        print(f"[*] Đang xử lý file: {file_name}")
    else:
        df = pd.DataFrame(backup_data)
        print(f"[!] Không tìm thấy {file_name}, đang dùng dữ liệu mô phỏng.")

  
    print(f"- Số dòng trống: {df[text_col].isnull().sum()}")
    df = df.dropna(subset=[text_col]).copy()

    
    le = LabelEncoder()
    for col in cat_cols:
        if col in df.columns:
            df[f'{col}_enc'] = le.fit_transform(df[col].astype(str))
    
    
    df['clean_text'] = df[text_col].str.lower()
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df['clean_text'])
    
    
    words = vectorizer.get_feature_names_out()
    target = target_word.lower()
    
    print(f"- Kết quả tìm kiếm từ liên quan đến '{target}':")
    if target in words:
        word_idx = np.where(words == target)[0][0]
        
        sim = cosine_similarity(tfidf_matrix.T)
        indices = sim[word_idx].argsort()[-6:-1][::-1]
        for i in indices:
            print(f"  + {words[i]}: {sim[word_idx][i]:.4f}")
    else:
        print(f"  ! Từ '{target}' không có trong dữ liệu. Gợi ý: {', '.join(words[:5])}")


d_hotel = {'review_text': ['phòng sạch sẽ', 'khách sạn sạch', 'vị trí tốt'], 'hotel_name': ['Lotus', 'Sunrise', 'Lotus']}
d_match = {'comment_text': ['trận đấu xuất sắc', 'cầu thủ xuất sắc', 'đá hay'], 'team': ['HAGL', 'SLNA', 'Hà Nội FC']}
d_player = {'feedback_text': ['đồ họa đẹp', 'nhân vật đẹp', 'game vui'], 'device': ['PC', 'Mobile', 'Console']}
d_album = {'review_text': ['âm nhạc sáng tạo', 'phối khí sáng tạo', 'lời hay'], 'genre': ['Pop', 'Rock', 'Indie']}



process_lab_task('ITA105_Lab_4_Hotel_reviews.csv', 'review_text', ['hotel_name', 'customer_type'], 'sạch', 'Bài 1: Khách sạn', d_hotel)
process_lab_task('ITA105_Lab_4_Match_comments.csv', 'comment_text', ['team', 'author'], 'xuất', 'Bài 2: Trận đấu', d_match)
process_lab_task('ITA105_Lab_4_Player_feedback.csv', 'feedback_text', ['player_type', 'device'], 'đẹp', 'Bài 3: Feedback', d_player)
process_lab_task('ITA105_Lab_4_Album_reviews.csv', 'review_text', ['genre', 'platform'], 'tạo', 'Bài 4: Album', d_album)