from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

# Assuming DocumentVector structure is needed for type hinting,
# we might need a generic version or define it here.
# For now, let's use Dict for broader compatibility, or forward reference if models are complex.
# from .pinecone_store import DocumentVector as PineconeDocumentVector # This would create a circular dependency if not careful

class DocumentVector: # A generic placeholder, actual implementation might vary
    id: str
    values: List[float]
    metadata: Dict[str, Any]

    def __init__(self, id: str, values: List[float], metadata: Dict[str, Any]):
        self.id = id
        self.values = values
        self.metadata = metadata


class BaseVectorStore(ABC):
    """
    Abstract base class for vector store operations.
    """

    @abstractmethod
    async def upsert_vectors(self, vectors: List[DocumentVector], namespace: Optional[str] = None) -> Dict[str, Any]:
        """
        Upserts (inserts or updates) vectors into the vector store.

        Args:
            vectors: A list of DocumentVector objects to upsert.
            namespace: Optional namespace within the vector store.

        Returns:
            A dictionary containing the response from the vector store, e.g., upserted count.
        """
        pass

    @abstractmethod
    async def query_vectors(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        namespace: Optional[str] = None,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Queries the vector store for similar vectors.

        Args:
            query_embedding: The embedding of the query.
            top_k: The number of top similar vectors to retrieve.
            namespace: Optional namespace to query within.
            filter_metadata: Optional metadata filter to apply.

        Returns:
            A list of dictionaries, where each dictionary represents a matched vector
            and its metadata.
        """
        pass

    # Optional: Add other common vector store operations if needed
    # @abstractmethod
    # async def delete_vectors(self, ids: List[str], namespace: Optional[str] = None) -> Dict[str, Any]:
    #     pass

    # @abstractmethod
    # async def delete_index(self): # Or delete_namespace / clear_all etc.
    #     pass
