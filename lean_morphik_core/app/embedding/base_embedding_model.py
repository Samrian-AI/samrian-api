from abc import ABC, abstractmethod
from typing import List, Union, TYPE_CHECKING
import numpy as np # Added for type hint consistency with Colpali output

# Use TYPE_CHECKING to avoid circular import at runtime if Chunk needs BaseEmbeddingModel
if TYPE_CHECKING:
    from ..models.chunk import Chunk
else:
    # Provide a fallback for runtime if needed, or ensure Chunk is defined elsewhere simply
    # For now, assuming Chunk can be resolved or will be defined such that this works.
    # If Chunk itself is simple enough, it could even be defined here or in a common types file.
    class Chunk: # Minimal stub if direct import is an issue and full def not needed for type hint alone
        pass


class BaseEmbeddingModel(ABC):
    @abstractmethod
    async def embed_for_ingestion(self, chunks: Union['Chunk', List['Chunk']]) -> List[np.ndarray]: # Changed to np.ndarray
        """Generate embeddings for input text"""
        pass

    @abstractmethod
    async def embed_for_query(self, text: str) -> List[float]:
        """Generate embeddings for input text"""
        pass
