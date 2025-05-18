import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
import re

try:
    df = pd.read_csv('Espacenet_search_result_20250510_1442.csv', skiprows=7, encoding='latin1', sep=';')
    print("Файл успешно загружен!")
    print("\nСтолбцы DataFrame:")
    print(df.columns)
    print("\nПервые 5 строк DataFrame:")
    print(df.head())
except Exception as e:
    print(f"Ошибка при загрузке: {e}")
    raise

required_columns = ['Publication number', 'Title', 'Abstract']
if not all(col in df.columns for col in required_columns):
    missing = [col for col in required_columns if col not in df.columns]
    raise KeyError(f"Отсутствуют необходимые столбцы: {missing}")

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

def check_relevance(pub):
    base_num = get_base_number(pub)
    return 'Relevant' if base_num in relevant_bases else 'Irrelevant'

df['relevance'] = df['Publication number'].apply(check_relevance)

relevant_count = df[df['relevance'] == 'Relevant'].shape[0]
irrelevant_count = df[df['relevance'] == 'Irrelevant'].shape[0]
print(f"\nКоличество релевантных патентов в датасете: {relevant_count}")
print(f"Количество нерелевантных патентов в датасете: {irrelevant_count}")
print(f"Ожидаемое количество уникальных базовых патентов: {len(relevant_bases)}")

df['text'] = df['Title'].astype(str) + '. ' + df['Abstract'].astype(str)

def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text.lower())
    tokens = text.split()
    tokens = [word for word in tokens if len(word) > 2]
    return ' '.join(tokens)

df['processed_text'] = df['text'].apply(preprocess_text)

tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['processed_text'])
feature_names = tfidf.get_feature_names_out()

relevant_indices = df[df['relevance'] == 'Relevant'].index
relevant_tfidf = tfidf_matrix[relevant_indices]
relevant_weights = relevant_tfidf.mean(axis=0).A1
top_terms = [feature_names[i] for i in relevant_weights.argsort()[-10:][::-1]]
print(f"\nТоп-10 терминов из релевантных патентов (по TF-IDF): {top_terms}")

query_text = 'debris shield upper tie plate spring'
print(f"Новый запрос: {query_text}")

model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')
query_embedding = model.encode(query_text)

df['similarity'] = df['processed_text'].apply(lambda x: cosine_similarity([query_embedding], [model.encode(x)])[0][0])

thresholds = [0.53, 0.55, 0.57, 0.59]

for t in thresholds:
    df['predicted'] = df['similarity'].apply(lambda s: 'Relevant' if s >= t else 'Irrelevant')

    y_true = df['relevance'].map({'Relevant': 1, 'Irrelevant': 0})
    y_pred = df['predicted'].map({'Relevant': 1, 'Irrelevant': 0})

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print(f"\n=== Порог: {t} ===")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1: {f1:.2f}")

top_patents = df.sort_values(by='similarity', ascending=False).head(30)

must_have_keywords = ['debris', 'shield', 'upper tie plate', 'spring']
top_patents['has_keywords'] = top_patents['processed_text'].apply(
    lambda x: any(kw in x for kw in must_have_keywords)
)
filtered_top_patents = top_patents[top_patents['has_keywords'] | (top_patents['relevance'] == 'Relevant')]

print("\nТоп-30 релевантных патентов после фильтрации:")
print(filtered_top_patents[['Publication number', 'similarity', 'relevance', 'Title']])

filtered_indices = filtered_top_patents.index
df_filtered = df.copy()
df_filtered['predicted'] = 'Irrelevant'
df_filtered.loc[filtered_indices, 'predicted'] = 'Relevant'

y_true_filtered = df_filtered['relevance'].map({'Relevant': 1, 'Irrelevant': 0})
y_pred_filtered = df_filtered['predicted'].map({'Relevant': 1, 'Irrelevant': 0})

precision_filtered = precision_score(y_true_filtered, y_pred_filtered, zero_division=0)
recall_filtered = recall_score(y_true_filtered, y_pred_filtered, zero_division=0)
f1_filtered = f1_score(y_true_filtered, y_pred_filtered, zero_division=0)

print(f"\nМетрики после фильтрации (на основе топ-30):")
print(f"Precision: {precision_filtered:.2f}")
print(f"Recall: {recall_filtered:.2f}")
print(f"F1: {f1_filtered:.2f}")

wordcloud_text = ' '.join(df['processed_text'].astype(str))
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(wordcloud_text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.savefig('wordcloud.png')
print("\nОблако слов сохранено как wordcloud.png")
