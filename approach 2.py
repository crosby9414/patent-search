!pip install transformers torch pandas scikit-learn tqdm matplotlib
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
nltk.download('stopwords')  # Add this line before importing stopwords
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
import numpy as np
import logging
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, classification_report
from transformers import BertTokenizer, BertModel
import torch
from tqdm import tqdm
from collections import Counter

# 1. Загрузка и предобработка данных
try:
    df = pd.read_csv('Espacenet_search_result_20250510_1442.csv',
                    skiprows=7, encoding='latin1', sep=';')
    print(f"Загружено {len(df)} патентов")
except Exception as e:
    print(f"Ошибка загрузки: {e}")
    raise

# 2. Определение релевантных патентов
relevant_bases = {
    'EP2031600', 'EP2192591', 'EP3709312', 'EP2031599', 'EP3842739',
    'EP1710809', 'EP4323999', 'EP2206122', 'EP2169685', 'EP3709311',
    'EP1986197', 'EP4386776', 'EP2122634', 'EP2428964', 'FR2870627',
    'ES2403013'
}

def extract_pub_number(pub_str):
    if pd.isna(pub_str):
        return ''
    pub_numbers = [pn.strip() for pn in pub_str.split('\r\n') if pn.strip()]
    return pub_numbers[0].split('A')[0] if pub_numbers else ''

df['base_number'] = df['Publication number'].apply(extract_pub_number)
df['relevant'] = df['base_number'].isin(relevant_bases).astype(int)
print(f"\nРаспределение классов:\n{df['relevant'].value_counts()}")

# 3. Подготовка текста
df['text'] = df['Title'].fillna('') + '. ' + df['Abstract'].fillna('')

# 4. Загрузка BERT модели
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased').to(device)

# 5. Функция для получения BERT эмбеддингов
def get_bert_embedding(text, model, tokenizer, device):
    inputs = tokenizer(text, return_tensors='pt',
                      truncation=True, max_length=512,
                      padding='max_length').to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:,0,:].cpu().numpy().flatten()

# 6. Создание эмбеддингов для всех патентов
embeddings = []
for text in tqdm(df['text'], desc="Создание BERT эмбеддингов"):
    embeddings.append(get_bert_embedding(text, model, tokenizer, device))
embeddings = np.array(embeddings)

# 7. Создание эмбеддинга для запроса
query = "debris shield upper tie plate spring"
query_embedding = get_bert_embedding(query, model, tokenizer, device)

# 8. Расчет косинусной схожести
from sklearn.metrics.pairwise import cosine_similarity
df['similarity'] = cosine_similarity([query_embedding], embeddings)[0]

# 9. Оптимизация порога
precisions, recalls, thresholds = precision_recall_curve(df['relevant'], df['similarity'])
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]

df['predicted'] = (df['similarity'] >= optimal_threshold).astype(int)

# 10. Оценка результатов
print("\nКлассификационный отчет:")
print(classification_report(df['relevant'], df['predicted']))

# 11. Визуализация
plt.figure(figsize=(10, 5))
plt.plot(thresholds, precisions[:-1], label='Precision')
plt.plot(thresholds, recalls[:-1], label='Recall')
plt.plot(thresholds, f1_scores[:-1], label='F1')
plt.axvline(optimal_threshold, color='r', linestyle='--')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.legend()
plt.title('Precision-Recall Tradeoff')
plt.savefig('bert_precision_recall.png')

# 12. Вывод топ-N релевантных патентов
top_n = 30
top_patents = df.sort_values('similarity', ascending=False).head(top_n)
print(f"\nТоп-{top_n} наиболее релевантных патентов:")
print(top_patents[['Publication number', 'Title', 'similarity', 'relevant']])
