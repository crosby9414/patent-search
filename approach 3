import torch
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, classification_report
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm

# 1. Загрузка данных
try:
    df = pd.read_csv('Espacenet_search_result_20250510_1442.csv',
                    skiprows=7, encoding='latin1', sep=';')
    print(f"Загружено {len(df)} патентов")
except Exception as e:
    print(f"Ошибка загрузки: {e}")
    raise

# Определение релевантных патентов
relevant_bases = {
    'EP2031600', 'EP2192591', 'EP3709312', 'EP2031599', 'EP3842739',
    'EP1710809', 'EP4323999', 'EP2206122', 'EP2169685', 'EP3709311',
    'EP1986197', 'EP4386776', 'EP2122634', 'EP2428964', 'FR2870627',
    'ES2403013'
}

# 2. Подготовка данных
df['text'] = df['Title'].fillna('') + '. ' + df['Abstract'].fillna('')
df['relevant'] = df['Publication number'].apply(lambda x: 1 if any(base in x for base in relevant_bases) else 0)

# 3. Создание Dataset класса
class PatentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# 4. Инициализация модели и токенизатора
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# 5. Подготовка DataLoader с балансировкой классов
dataset = PatentDataset(df['text'].values, df['relevant'].values, tokenizer)

# Вычисление весов для балансировки классов
class_counts = df['relevant'].value_counts().to_dict()
class_weights = 1. / torch.tensor(list(class_counts.values()), dtype=torch.float)
sample_weights = class_weights[df['relevant'].values]

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

dataloader = DataLoader(
    dataset,
    batch_size=8,
    sampler=sampler
)

# 6. Обучение модели
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
optimizer = AdamW(model.parameters(), lr=2e-5)

# Тренировочный цикл
model.train()
for epoch in range(3):  # 3 эпохи обычно достаточно для fine-tuning
    for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        loss.backward()
        optimizer.step()

# 7. Предсказание на всех данных
model.eval()
all_logits = []
with torch.no_grad():
    for batch in DataLoader(dataset, batch_size=8):
        inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
        outputs = model(**inputs)
        all_logits.append(outputs.logits.cpu())

# Объединение результатов
logits = torch.cat(all_logits, dim=0)
probs = torch.softmax(logits, dim=1)[:, 1].numpy()

# 8. Оптимизация порога
precisions, recalls, thresholds = precision_recall_curve(df['relevant'], probs)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]

# 9. Оценка результатов
df['predicted'] = (probs >= optimal_threshold).astype(int)
print("\nClassification Report:")
print(classification_report(df['relevant'], df['predicted']))

# 10. Топ-N результатов
top_n = df[df['predicted'] == 1].sort_values('probs', ascending=False).head(30)
print(f"\nTop 30 predicted relevant patents:")
print(top_n[['Publication number', 'Title', 'probs', 'relevant']])
