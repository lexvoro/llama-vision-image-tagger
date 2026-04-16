import ollama
from ollama import AsyncClient
from pathlib import Path
import logging
from typing import Dict, List, Optional
import json
from pydantic import BaseModel
from PIL import Image
import os
import asyncio

logger = logging.getLogger(__name__)

# Схемы данных
class ImageDescription(BaseModel):
    description: str

class ImageTags(BaseModel):
    tags: List[str]

class ImageText(BaseModel):
    has_text: bool
    text_content: str

class ImageProcessor:
    def __init__(self, model_name: str = 'llama3.2-vision'):
        self.model_name = model_name
        self.temp_path = Path("temp_processing.jpg")
        self.client = AsyncClient()

    async def process_image(self, image_path: Path, tag_count: int = 10, languages: List[str] = ["en"]) -> Dict:
        """Обработка изображения с поддержкой языков и количества тегов."""
        try:
            if not image_path.exists():
                return {"is_processed": False, "error": "File not found"}

            # Оптимизация изображения
            with Image.open(image_path) as img:
                img.thumbnail((1024, 1024))
                if img.mode in ("RGBA", "P"):
                    img = img.convert("RGB")
                img.save(self.temp_path, "JPEG", quality=85)

            image_path_str = str(self.temp_path.absolute())

            logger.info(f"🔍 Анализ {image_path.name} (Тегов: {tag_count}, Языки: {languages})")
            
            # Запросы к нейросети
            description_res = await self._get_description(image_path_str)
            tags_res = await self._get_tags(image_path_str, tag_count, languages)
            text_res = await self._get_text_content(image_path_str)

            if self.temp_path.exists():
                os.remove(self.temp_path)

            # Распределяем теги по спискам ru/en
            all_generated_tags = tags_res.tags
            en_tags = [t.strip().lower() for t in all_generated_tags if not any(ord(c) > 127 for c in t)]
            ru_tags = [t.strip().lower() for t in all_generated_tags if any(ord(c) > 127 for c in t)]

            return {
                "description": description_res.description,
                "tags": en_tags,
                "tags_ru": ru_tags,
                "text_content": text_res.text_content if text_res.has_text else "",
                "is_processed": True
            }

        except Exception as e:
            logger.error(f"❌ Ошибка обработки {image_path.name}: {str(e)}")
            if self.temp_path.exists():
                os.remove(self.temp_path)
            return {
                "description": "", "tags": [], "tags_ru": [], "text_content": "",
                "is_processed": False, "error": str(e)
            }

    async def _get_description(self, image_path: str) -> ImageDescription:
        prompt = "Describe this image in one short sentence in English."
        response = await self._query_ollama(prompt, image_path, ImageDescription.model_json_schema())
        return ImageDescription.model_validate_json(response)

    async def _get_tags(self, image_path: str, tag_count: int, languages: List[str]) -> ImageTags:
            # Создаем очень жесткую инструкцию
            instructions = []
            if "en" in languages:
                instructions.append(f"{tag_count} tags in English")
            if "ru" in languages:
                instructions.append(f"{tag_count} tags in Russian (кириллица)")
            
            full_instruction = " and ".join(instructions)
            
            prompt = (
                f"Analyze this image and generate {full_instruction}. "
                f"Return them as a single flat JSON list of strings. "
                f"Example format: ['forest', 'лес', 'nature', 'природа']."
            )
            
            response = await self._query_ollama(prompt, image_path, ImageTags.model_json_schema())
            return ImageTags.model_validate_json(response)

    async def _get_text_content(self, image_path: str) -> ImageText:
        prompt = "Is there any text in this image? If yes, transcribe it."
        response = await self._query_ollama(prompt, image_path, ImageText.model_json_schema())
        return ImageText.model_validate_json(response)

    async def _query_ollama(self, prompt: str, image_path: str, format_schema: dict) -> str:
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Используем AsyncClient для асинхронности
                response = await self.client.chat(
                    model=self.model_name,
                    messages=[
                        {'role': 'system', 'content': 'You are a helpful assistant that outputs only JSON.'},
                        {'role': 'user', 'content': prompt, 'images': [image_path]}
                    ],
                    options={
                        'temperature': 0.2,
                        'num_gpu': -1,
                        'repeat_penalty': 1.5
                    },
                    format=format_schema
                )
                return response['message']['content']
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                await asyncio.sleep(1)

def update_image_metadata(folder_path: Path, image_path: str, metadata: Dict) -> None:
    metadata_file = folder_path / "image_metadata.json"
    try:
        all_metadata = {}
        if metadata_file.exists():
            with open(metadata_file, 'r', encoding='utf-8') as f:
                all_metadata = json.load(f)
        
        all_metadata[image_path] = metadata
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(all_metadata, f, indent=4, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Error saving metadata: {str(e)}")
