import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from deep_translator import GoogleTranslator
import logging

logger = logging.getLogger(__name__)

# --- 1. Загрузка путей из переменных окружения ---
# Эти переменные нужно будет создать в настройках Render
AMAZON_REVIEWS_PATH = os.environ.get("AMAZON_REVIEWS_PATH")
SENTIMENT_MODEL_PATH = os.environ.get("SENTIMENT_MODEL_PATH")
VECTORIZER_PATH = os.environ.get("VECTORIZER_PATH")
MERGED_DF_PATH = os.environ.get("MERGED_DF_PATH")
COLLAB_MODEL_PATH = os.environ.get("COLLAB_MODEL_PATH")
CONTENT_MODEL_PATH = os.environ.get("CONTENT_MODEL_PATH")

# --- 2. Определение классов моделей PyTorch ---
# Скопируйте определения ваших классов из блокнота
class CollaborativeFilteringModel(nn.Module):
    # ... (ваш код класса) ...
    def __init__(self, num_users, num_items, embedding_dim, hidden_dim):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.hidden_layer = nn.Linear(embedding_dim * 2, hidden_dim)
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, user_indices, item_indices):
        user_embedded = self.user_embedding(user_indices)
        item_embedded = self.item_embedding(item_indices)
        concatenated = torch.cat([user_embedded, item_embedded], dim=1)
        hidden_output = self.relu(self.hidden_layer(concatenated))
        output = self.output_layer(hidden_output)
        return output

    def get_similar_titles(self, input_title_index, top_k=100):
        device = self.item_embedding.weight.device
        input_title_index = torch.tensor([input_title_index], device=device)
        input_title_embedding = self.item_embedding(input_title_index)
        all_title_embeddings = self.item_embedding.weight
        similarities = F.cosine_similarity(input_title_embedding, all_title_embeddings)
        similar_title_indices = torch.argsort(similarities, descending=True)[:top_k]
        similar_titles = [index_to_title[idx.item()] for idx in similar_title_indices]
        return similar_titles

class ContentBasedFilteringModel(nn.Module):
    # ... (ваш код класса) ...
    def __init__(self, num_categories, num_authors, num_titles, embedding_dim):
        super(ContentBasedFilteringModel, self).__init__()
        self.category_embedding = nn.Embedding(num_categories, embedding_dim)
        self.author_embedding = nn.Embedding(num_authors, embedding_dim)
        self.title_embedding = nn.Embedding(num_titles, embedding_dim)
        self.sentiment_linear = nn.Linear(4 * embedding_dim, 1)

    def forward(self, category_indices, author_indices, title_indices, sentiment_scores):
        category_embedded = self.category_embedding(category_indices)
        author_embedded = self.author_embedding(author_indices)
        title_embedded = self.title_embedding(title_indices)
        sentiment_expanded = sentiment_scores.unsqueeze(1).expand_as(category_embedded)
        concatenated = torch.cat([category_embedded, author_embedded, title_embedded, sentiment_expanded], dim=1)
        output = self.sentiment_linear(concatenated)
        return output

# --- 3. Глобальные переменные и функции для загрузки данных и моделей ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
merged_df = None
model = None
cbf_model = None
item_to_index = None
index_to_title = None
id_to_english_title = None
translator_ru = None

def startup_event():
    """Функция, которая загружает все данные и модели при старте сервера."""
    global merged_df, model, cbf_model, item_to_index, index_to_title, id_to_english_title, translator_ru
    logger.info("Loading data and models...")

    # Загрузка данных
    logger.info("Loading merged dataframe...")
    merged_df = pd.read_csv(MERGED_DF_PATH)
    merged_df = merged_df.fillna('')
    logger.info("Merged dataframe loaded.")

    # Загрузка маппингов (это нужно делать после загрузки merged_df)
    user_ids = merged_df['User_id'].unique()
    item_ids = merged_df['Title'].unique()
    user_to_index = {user_id: idx for idx, user_id in enumerate(user_ids)}
    item_to_index = {item_id: idx for idx, item_id in enumerate(item_ids)}
    index_to_title = {idx: title for title, idx in item_to_index.items()}
    id_to_english_title = {idx: title for title, idx in item_to_index.items()}

    # Загрузка моделей
    logger.info("Loading collaborative filtering model...")
    model = torch.load(COLLAB_MODEL_PATH, map_location=device, weights_only=False)
    model.to(device)
    model.eval()
    logger.info("Collaborative filtering model loaded.")

    logger.info("Loading content-based filtering model...")
    # Важно: сначала нужно создать экземпляр модели с правильными параметрами
    # num_categories, num_authors, num_titles должны быть определены
    # Если они не определены в этом файле, их нужно передать или вычислить
    # В вашем блокноте они вычисляются так:
    unique_categories = merged_df['categories'].unique()
    unique_authors = merged_df['authors'].unique()
    title_sentiment_aggregated = merged_df.groupby(['Title','authors','categories'])['sentiment_score'].mean().reset_index()
    unique_titles = title_sentiment_aggregated['Title'].unique()
    num_categories = len(unique_categories)
    num_authors = len(unique_authors)
    num_titles = len(unique_titles)
    embedding_dim = 64
    cbf_model = ContentBasedFilteringModel(num_categories, num_authors, num_titles, embedding_dim)
    cbf_model.load_state_dict(torch.load(CONTENT_MODEL_PATH, map_location=device, weights_only=False))
    cbf_model.to(device)
    cbf_model.eval()
    logger.info("Content-based filtering model loaded.")

    # Инициализация переводчика
    translator_ru = GoogleTranslator(source='en', target='ru')
    logger.info("Startup complete!")

def translate_to_ru_for_user(english_titles):
    """Переводит список английских названий на русский."""
    if not english_titles:
        return []
    try:
        return [translator_ru.translate(title) for title in english_titles]
    except Exception as e:
        logger.error(f"Translation error: {e}")
        return english_titles

def find_english_title_by_id(book_id):
    """По числовому ID возвращает английское название."""
    return id_to_english_title.get(book_id, None)

def get_collaborative_recommendations(model, title, num_recommendations=100):
    # ... (ваш код функции из блокнота) ...
    input_title_index = item_to_index[title]
    model.eval()
    with torch.inference_mode():
        similar_titles = model.get_similar_titles(input_title_index, top_k=num_recommendations)
    return similar_titles

def get_content_based_recommendations(content_based_model, collaborative_recommendations):
    # ... (ваш код функции из блокнота) ...
    title_details = title_sentiment_aggregated.set_index('Title')[['categories', 'authors', 'sentiment_score']].to_dict(orient='index')
    details = [title_details[title] for title in collaborative_recommendations]
    category_indices = torch.tensor([category_to_index[detail['categories']] for detail in details], dtype=torch.long)
    author_indices = torch.tensor([author_to_index[detail['authors']] for detail in details], dtype=torch.long)
    title_indices = torch.tensor([title_to_index[title] for title in collaborative_recommendations], dtype=torch.long)
    sentiment_scores = torch.tensor([detail['sentiment_score'] for detail in details], dtype=torch.float32)
    category_indices, author_indices, title_indices, sentiment_scores = category_indices.to(device), author_indices.to(device), title_indices.to(device), sentiment_scores.to(device)
    content_based_model.eval()
    with torch.inference_mode():
        predictions = content_based_model(category_indices, author_indices, title_indices, sentiment_scores)
    sorted_titles = [title for _, title in sorted(zip(predictions, collaborative_recommendations), reverse=True)]
    return sorted_titles