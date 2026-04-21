import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from typing import List
import uvicorn

# Импортируем функции и переменные из вашего модуля с моделью
from model_utils import (
    model, cbf_model, item_to_index, index_to_title,
    get_collaborative_recommendations, get_content_based_recommendations,
    find_english_title_by_id, translate_to_ru_for_user, startup_event
)

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Определяем модель данных для запроса
class RecommendRequest(BaseModel):
    book_id: int  # ID книги, полученный от пользователя

class RecommendResponse(BaseModel):
    original_book_id: int
    recommendations: List[str]

# lifespan менеджер для загрузки модели при старте сервера
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Загрузка модели при запуске
    await startup_event()
    yield
    # Здесь можно добавить код для выгрузки модели, если необходимо

app = FastAPI(lifespan=lifespan)

@app.get("/health")
async def health_check():
    """Эндпоинт для проверки работоспособности сервера."""
    return {"status": "ok"}

@app.post("/recommend", response_model=RecommendResponse)
async def get_recommendations(request: RecommendRequest):
    """Эндпоинт, который будет вызывать бот для получения рекомендаций."""
    book_id = request.book_id
    logger.info(f"Received request for book ID: {book_id}")

    # 1. Находим английское название по ID
    en_title = find_english_title_by_id(book_id)
    if en_title is None:
        raise HTTPException(status_code=404, detail=f"Book with ID {book_id} not found.")

    # 2. Получаем английские рекомендации
    try:
        recs_en = get_collaborative_recommendations(model, en_title, num_recommendations=1000)
        recs_en = get_content_based_recommendations(cbf_model, recs_en)
    except Exception as e:
        logger.error(f"Error getting recommendations: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during recommendation.")

    # 3. Переводим на русский
    recs_ru = translate_to_ru_for_user(recs_en)

    # 4. Возвращаем результат
    return RecommendResponse(original_book_id=book_id, recommendations=recs_ru[:10])

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)