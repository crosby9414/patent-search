import pandas as pd
import re
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import numpy as np
import spacy
from spacy.lang.en.stop_words import STOP_WORDS

# Шаг 1: Загрузка данных
# Предполагается, что файл доступен локально или через аналогичный путь
try:
    df = pd.read_csv('Espacenet_search_result_20250510_1442.csv', skiprows=7, encoding='latin1', sep=';')
    print(f"Загружено {len(df)} патентов")
except Exception as e:
    print(f"Ошибка загрузки: {e}")
    raise

# Шаг 2: Определение релевантных патентов
relevant_bases = {
    'EP2031600', 'EP2192591', 'EP3709312', 'EP2031599', 'EP3842739',
    'EP1710809', 'EP4323999', 'EP2206122', 'EP2169685', 'EP3709311',
    'EP1986197', 'EP4386776', 'EP2122634', 'EP2428964', 'FR2870627',
    'ES2403013'
}

def get_base_number(pub):
    if pd.isna(pub) or not isinstance(pub, str):
        return ''
    pub_numbers = [pn.strip() for pn in pub.split('\r\n') if pn.strip()]
    if not pub_numbers:
        return ''
    first_pub = pub_numbers[0]
    match = re.match(r'^([A-Z]+[0-9]+)(?:[A-Z0-9]+)?$', first_pub)
    return match.group(1) if match else first_pub

df['base_number'] = df['Publication number'].apply(get_base_number)
df['relevant'] = df['base_number'].isin(relevant_bases).astype(int)
print(f"\nРаспределение классов:\n{df['relevant'].value_counts()}")

# Шаг 3: Предобработка текста
# Объединение Title и Abstract
df['text'] = df['Title'].fillna('') + '. ' + df['Abstract'].fillna('')

# Загрузка spacy для лемматизации и удаления стоп-слов
nlp = spacy.load('en_core_web_sm')

def preprocess_text(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if token.is_alpha and len(token) > 2 and token.text not in STOP_WORDS]
    return ' '.join(tokens)

df['processed_text'] = df['text'].apply(preprocess_text)

# Шаг 4: Генерация эмбеддингов с использованием PatentSBERTa
model = SentenceTransformer('AI-Growth-Lab/PatentSBERTa')
embeddings = model.encode(df['processed_text'].tolist(), show_progress_bar=True)

# Шаг 5: Разделение данных на обучающий и тестовый наборы
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['relevant'], random_state=42)

train_indices = train_df.index.tolist()
test_indices = test_df.index.tolist()

train_embeddings = embeddings[train_indices]
test_embeddings = embeddings[test_indices]

train_labels = train_df['relevant'].values
test_labels = test_df['relevant'].values

# Шаг 6: Обучение классификатора (логистическая регрессия с балансировкой классов)
clf = LogisticRegression(class_weight='balanced', max_iter=1000)
clf.fit(train_embeddings, train_labels)

# Шаг 7: Оценка модели на тестовом наборе
test_pred = clf.predict(test_embeddings)
print("\nClassification Report:")
print(classification_report(test_labels, test_pred))

# Шаг 8: Предсказание на всем наборе данных и вывод топ-30 релевантных патентов
all_pred_proba = clf.predict_proba(embeddings)[:, 1]  # Вероятность класса 1 (релевантный)
df['pred_proba'] = all_pred_proba

top_30 = df.sort_values(by='pred_proba', ascending=False).head(30)
print("\nTop 30 predicted relevant patents:")
print(top_30[['Publication number', 'Title', 'pred_proba', 'relevant']])
