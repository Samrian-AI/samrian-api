import base64
import io
import logging
import time
from typing import List, Tuple, Union, Optional

import numpy as np
import torch
from transformers import AutoModel, AutoProcessor
from PIL.Image import Image
from PIL.Image import open as open_image

from .base_embedding_model import BaseEmbeddingModel
from ..models.chunk import Chunk

logger = logging.getLogger(__name__)


class ColpaliEmbeddingModel(BaseEmbeddingModel):
    def __init__(self,
                 device: Optional[str] = None,
                 model_name: str = "tsystems/colqwen2.5-3b-multilingual-v1.0",
                 processor_name: str = "tsystems/colqwen2.5-3b-multilingual-v1.0",
                 batch_size: int = 1
                ):

        self.device = device if device else ("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initializing ColpaliEmbeddingModel with device: {self.device}")
        start_time = time.time()

        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            attn_implementation="flash_attention_2" if self.device == "cuda" else "eager",
            trust_remote_code=True
        ).eval()

        self.processor: AutoProcessor = AutoProcessor.from_pretrained(
            processor_name,
            trust_remote_code=True
        )

        self.batch_size = batch_size
        logger.info(f"Colpali configured with batch size: {self.batch_size}")

        total_init_time = time.time() - start_time
        logger.info(f"Colpali initialization time: {total_init_time:.2f} seconds")

    async def embed_for_ingestion(self, chunks: Union[Chunk, List[Chunk]]) -> List[np.ndarray]:
        job_start_time = time.time()
        if isinstance(chunks, Chunk):
            chunks = [chunks]

        if not chunks:
            return []

        logger.info(
            f"Processing {len(chunks)} chunks for Colpali embedding (batch size: {self.batch_size})"
        )

        image_items: List[Tuple[int, Image]] = []
        text_items: List[Tuple[int, str]] = []

        for index, chunk in enumerate(chunks):
            if chunk.metadata.get("is_image"):
                try:
                    content = chunk.content
                    if content.startswith("data:"):
                        content = content.split(",", 1)[1]
                    image_bytes = base64.b64decode(content)
                    image = open_image(io.BytesIO(image_bytes))
                    image_items.append((index, image))
                except Exception as e:
                    logger.error(f"Error processing image chunk {index}: {str(e)}. Falling back to text.")
                    text_items.append((index, chunk.content))
            else:
                text_items.append((index, chunk.content))

        results: List[np.ndarray | None] = [None] * len(chunks)

        if image_items:
            indices_to_process = [item[0] for item in image_items]
            images_to_process = [item[1] for item in image_items]
            for i in range(0, len(images_to_process), self.batch_size):
                batch_indices = indices_to_process[i : i + self.batch_size]
                batch_images = images_to_process[i : i + self.batch_size]
                batch_embeddings = await self.generate_embeddings_batch_images(batch_images)
                for original_index, embedding in zip(batch_indices, batch_embeddings):
                    results[original_index] = embedding

        if text_items:
            indices_to_process = [item[0] for item in text_items]
            texts_to_process = [item[1] for item in text_items]
            for i in range(0, len(texts_to_process), self.batch_size):
                batch_indices = indices_to_process[i : i + self.batch_size]
                batch_texts = texts_to_process[i : i + self.batch_size]
                batch_embeddings = await self.generate_embeddings_batch_texts(batch_texts)
                for original_index, embedding in zip(batch_indices, batch_embeddings):
                    results[original_index] = embedding

        final_results = [res for res in results if res is not None]
        if len(final_results) != len(chunks):
            logger.warning(
                f"Number of embeddings ({len(final_results)}) does not match number of chunks ({len(chunks)})."
            )

        total_time = time.time() - job_start_time
        logger.info(
            f"Total Colpali embed_for_ingestion took {total_time:.2f}s for {len(chunks)} chunks"
        )
        return final_results

    async def embed_for_query(self, text: str) -> np.ndarray:
        start_time = time.time()
        result_np = await self.generate_embeddings(text)
        elapsed = time.time() - start_time
        logger.info(f"Colpali query embedding took {elapsed:.2f}s")
        return result_np

    async def generate_embeddings(self, content: Union[str, Image]) -> np.ndarray:
        if isinstance(content, Image):
            try:
                processed = self.processor(images=content, return_tensors="pt").to(self.model.device)
            except Exception as e:
                logger.error(f"Could not process image with AutoProcessor, error: {e}. Using placeholder text.")
                processed = self.processor(text="[image placeholder]", return_tensors="pt").to(self.model.device)
        else:
            processed = self.processor(text=content, return_tensors="pt", padding=True, truncation=True).to(self.model.device)

        with torch.no_grad():
            # Common way to get sentence embedding: mean of last hidden state
            embeddings: torch.Tensor = self.model(**processed).last_hidden_state.mean(dim=1)

        return embeddings.cpu().to(torch.float32).numpy()[0]


    async def generate_embeddings_batch_images(self, images: List[Image]) -> List[np.ndarray]:
        if not images: return []
        try:
            processed_images = self.processor(images=images, return_tensors="pt", padding=True).to(self.model.device)
        except Exception as e:
            logger.error(f"Batch image processing failed with AutoProcessor: {e}. Returning zero embeddings for batch.")
            # Determine embedding size dynamically or use a common default
            embedding_size = getattr(self.model.config, 'hidden_size', 768)
            return [np.zeros(embedding_size) for _ in images]

        with torch.no_grad():
            image_embeddings = self.model(**processed_images).last_hidden_state.mean(dim=1)

        return [emb.cpu().to(torch.float32).numpy() for emb in image_embeddings]

    async def generate_embeddings_batch_texts(self, texts: List[str]) -> List[np.ndarray]:
        if not texts: return []
        processed_texts = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(self.model.device)
        with torch.no_grad():
            text_embeddings = self.model(**processed_texts).last_hidden_state.mean(dim=1)

        return [emb.cpu().to(torch.float32).numpy() for emb in text_embeddings]
