import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel as PydanticBaseModel, Field # Renamed to avoid clash

# Import ABC and the generic DocumentVector from base_vector_store
from .base_vector_store import BaseVectorStore, DocumentVector

# Assuming pinecone-client is installed
try:
    from pinecone import Pinecone, Index
    from pinecone.models.serverless import ServerlessSpecCloudEnum
    from pinecone.core.client.exceptions import NotFoundException, ApiException, PineconeApiException
except ImportError:
    Pinecone = None
    Index = None
    ServerlessSpecCloudEnum = None
    NotFoundException = None
    ApiException = None
    PineconeApiException = None # Ensure this is also handled for tenacity
    logging.warning("pinecone-client not installed. PineconeStore will not be functional.")

# Import tenacity for retries
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)

# A simple Pydantic model for Document data to be stored internally by PineconeStore
# Renamed to avoid conflict with the generic DocumentVector from BaseVectorStore
class PineconeDocumentModel(PydanticBaseModel):
    id: str
    values: List[float]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PineconeStore(BaseVectorStore): # Inherit from BaseVectorStore
    def __init__(self, api_key: str, environment: Optional[str] = None, index_name: str = "lean-morphik-index", dimension: int = 1024, cloud: str = "aws", region: str = "us-east-1"):
        if Pinecone is None:
            raise ImportError("Pinecone client is not installed. Please install with `pip install pinecone-client`")

        self.pinecone = Pinecone(api_key=api_key, environment=environment) # environment is optional for serverless
        self.index_name = index_name
        self.dimension = dimension
        self.cloud = cloud
        self.region = region
        self._index: Optional[Index] = None
        self._connect_to_index() # Initial connection attempt

    # Define retry strategy for Pinecone API calls
    # Retry on specific PineconeApiException or a general ApiException if that's more common.
    # PineconeApiException is more specific for actual API operational errors.
    _pinecone_retry_conditions = retry_if_exception_type(PineconeApiException) | retry_if_exception_type(ApiException)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=_pinecone_retry_conditions,
        reraise=True # Reraise the exception if all retries fail
    )
    def _connect_to_index_with_retry(self):
        """Connects to the Pinecone index with retry logic."""
        # This method encapsulates the original _connect_to_index logic
        # but is called with tenacity's retry decorator.
        # The __init__ can call this version.
        if self.index_name not in self.pinecone.list_indexes().names:
            logger.info(f"Index '{self.index_name}' not found. Creating new serverless index.")
            self.pinecone.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric="cosine", # Or "euclidean", "dotproduct"
                spec={
                    "serverless": {
                        "cloud": self.cloud if self.cloud in ServerlessSpecCloudEnum.__members__ else ServerlessSpecCloudEnum.AWS,
                        "region": self.region
                    }
                }
            )
            logger.info(f"Index '{self.index_name}' created successfully.")
        else:
            logger.info(f"Connecting to existing index '{self.index_name}'.")

        self._index = self.pinecone.Index(self.index_name)
        logger.info(f"Successfully connected to Pinecone index '{self.index_name}'. Stats: {self._index.describe_index_stats()}")


    def _connect_to_index(self):
        # Original _connect_to_index logic now wrapped by _connect_to_index_with_retry
        # This is called from __init__ to make the initial connection attempt.
        # Subsequent accesses use the @property which can also trigger a reconnect if needed.
        try:
            self._connect_to_index_with_retry()
        except (PineconeApiException, ApiException) as e: # Catch specific exceptions from retry
            logger.error(f"Pinecone API error during index connection or creation after retries: {e}")
            raise
        except Exception as e: # Catch any other unexpected error
            logger.error(f"An unexpected error occurred during Pinecone setup: {e}")
            raise


    @property
    def index(self) -> Index:
        if self._index is None:
            logger.warning("Pinecone index not initialized. Attempting to reconnect.")
            try:
                self._connect_to_index_with_retry() # Use retry version for reconnects
            except Exception as e:
                logger.error(f"Failed to reconnect to Pinecone index: {e}")
                raise ConnectionError(f"Failed to connect to Pinecone index after retries: {e}")

            if self._index is None: # If still None after trying to reconnect
                 raise ConnectionError("Failed to connect to Pinecone index.")
        return self._index

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=_pinecone_retry_conditions,
        reraise=True
    )
    async def upsert_vectors(self, vectors: List[DocumentVector], namespace: Optional[str] = None) -> Dict[str, Any]:
        """
        Upserts (inserts or updates) vectors into the Pinecone index.
        Accepts a list of generic DocumentVector objects as per BaseVectorStore.
        """
        if not vectors:
            return {"upserted_count": 0}

        pinecone_sdk_vectors = []
        for v_generic in vectors:
            pinecone_sdk_vectors.append((v_generic.id, v_generic.values, v_generic.metadata))

        # The try-except block for logging specific errors moves inside the retry loop implicitly
        # or can be kept if you want to log each attempt's failure differently.
        # Tenacity handles the raising of the exception after attempts are exhausted.
        logger.info(f"Attempting to upsert {len(pinecone_sdk_vectors)} vectors to namespace '{namespace or 'default'}'.")
        upsert_response = self.index.upsert(vectors=pinecone_sdk_vectors, namespace=namespace)
        logger.info(f"Upsert successful: {upsert_response}")
        return upsert_response.to_dict() if hasattr(upsert_response, "to_dict") else vars(upsert_response)


    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=_pinecone_retry_conditions,
        reraise=True
    )
    async def query_vectors(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        namespace: Optional[str] = None,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Queries the Pinecone index for similar vectors.
        """
        logger.info(f"Attempting to query namespace '{namespace or 'default'}' with top_k={top_k}.")
        query_response = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            namespace=namespace,
            filter=filter_metadata,
            include_metadata=True,
            include_values=False # Typically not needed for query results, saves bandwidth
        )
        logger.info(f"Query successful: {len(query_response.matches)} matches found.")
        return [match.to_dict() for match in query_response.matches]


    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=_pinecone_retry_conditions,
        reraise=True
    )
    async def delete_index(self):
        """Deletes the entire Pinecone index."""
        if self.index_name in self.pinecone.list_indexes().names:
            logger.info(f"Attempting to delete index '{self.index_name}'.")
            self.pinecone.delete_index(self.index_name)
            logger.info(f"Index '{self.index_name}' deleted successfully.")
            self._index = None
        else:
            logger.info(f"Index '{self.index_name}' not found, skipping deletion.")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=_pinecone_retry_conditions,
        reraise=True
    )
    async def delete_vectors(self, ids: List[str], namespace: Optional[str] = None) -> Dict[str, Any]:
        """Deletes vectors by ID from the specified namespace."""
        if not ids:
            return {}
        logger.info(f"Attempting to delete {len(ids)} vectors from namespace '{namespace or 'default'}'.")
        delete_response = self.index.delete(ids=ids, namespace=namespace)
        logger.info(f"Delete vectors successful: {delete_response}")
        return delete_response if isinstance(delete_response, dict) else {}

# Example Usage (for testing purposes, typically not in the class file)
# Note: The main() function here is synchronous, but the store methods are async.
# To test async methods, main() would also need to be async and run with asyncio.run().
# The original main() was synchronous and would not work directly with async methods without adjustment.
# For brevity, the main() function's internal calls are not updated to await,
# as its primary purpose here is illustrative of class usage, not a runnable test.
async def main():
    # This requires PINECONE_API_KEY environment variable to be set
    # For testing, we use the generic DocumentVector from the base class
    import os
    # api_key = os.environ.get("PINECONE_API_KEY") # Keep this for actual runs
    # if not api_key:
    #     print("Please set the PINECONE_API_KEY environment variable.")
    #     return

    # # Ensure you have a serverless environment or configure for pod-based
    # # For serverless, environment is not strictly needed for Pinecone client init
    # # but good to be aware of.
    # # For pod-based, you would need PINECONE_ENVIRONMENT.
    # # store = PineconeStore(api_key=api_key, environment="your-pinecone-environment", index_name="my-test-index", dimension=3)

    # # Using serverless defaults for cloud and region
    # store = PineconeStore(api_key=api_key, index_name="my-lean-test-index", dimension=128, cloud="aws", region="us-west-2")


    # try:
    #     print(f"Pinecone index instance: {store.index}")

    #     # Upsert example using generic DocumentVector
    #     vectors_to_upsert = [
    #         DocumentVector(id="vec1", values=[0.1] * 128, metadata={"genre": "drama"}),
    #         DocumentVector(id="vec2", values=[0.2] * 128, metadata={"genre": "comedy"}),
    #         DocumentVector(id="vec3", values=[0.3] * 128, metadata={"genre": "drama"}),
    #     ]
    #     upsert_result = await store.upsert_vectors(vectors_to_upsert, namespace="test-ns")
    #     print(f"Upsert result: {upsert_result}")

    #     # Give some time for indexing
    #     import time
    #     time.sleep(5)


    #     # Query example
    #     query_emb = [0.11] * 128
    #     query_results = await store.query_vectors(query_embedding=query_emb, top_k=2, namespace="test-ns")
    #     print(f"Query results for [0.11...]: {query_results}")

    #     query_results_filtered = await store.query_vectors(
    #         query_embedding=query_emb,
    #         top_k=2,
    #         namespace="test-ns",
    #         filter_metadata={"genre": "comedy"}
    #     )
    #     print(f"Query results filtered for genre 'comedy': {query_results_filtered}")

        # Delete vectors by ID
        # delete_vec_result = await store.delete_vectors(ids=["vec1"], namespace="test-ns")
        # print(f"Delete vector 'vec1' result: {delete_vec_result}")
        # time.sleep(2)
        # query_results_after_delete = await store.query_vectors(query_embedding=query_emb, top_k=2, namespace="test-ns")
        # print(f"Query results after deleting vec1: {query_results_after_delete}")


    # except Exception as e:
    #     print(f"An error occurred: {e}")
    # finally:
        # Clean up: delete the index
        # print("Attempting to delete index...")
        # await store.delete_index()
        # pass

if __name__ == "__main__":
    import asyncio
    # asyncio.run(main()) # Comment out for non-testing use
    pass
