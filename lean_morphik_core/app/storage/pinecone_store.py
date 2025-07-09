import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

# Assuming pinecone-client is installed
try:
    from pinecone import Pinecone, Index
    from pinecone.models.serverless import ServerlessSpecCloudEnum
    from pinecone.core.client.exceptions import NotFoundException, ApiException
except ImportError:
    Pinecone = None
    Index = None
    ServerlessSpecCloudEnum = None
    NotFoundException = None
    ApiException = None
    logging.warning("pinecone-client not installed. PineconeStore will not be functional.")

logger = logging.getLogger(__name__)

# A simple Pydantic model for Document data to be stored
class DocumentVector(BaseModel):
    id: str
    values: List[float]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PineconeStore:
    def __init__(self, api_key: str, environment: Optional[str] = None, index_name: str = "lean-morphik-index", dimension: int = 1024, cloud: str = "aws", region: str = "us-east-1"):
        if Pinecone is None:
            raise ImportError("Pinecone client is not installed. Please install with `pip install pinecone-client`")

        self.pinecone = Pinecone(api_key=api_key, environment=environment) # environment is optional for serverless
        self.index_name = index_name
        self.dimension = dimension
        self.cloud = cloud
        self.region = region
        self._index: Optional[Index] = None
        self._connect_to_index()

    def _connect_to_index(self):
        try:
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

        except ApiException as e:
            logger.error(f"Pinecone API error during index connection or creation: {e}")
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred during Pinecone setup: {e}")
            raise

    @property
    def index(self) -> Index:
        if self._index is None:
            # This case should ideally not be hit if __init__ is successful
            logger.warning("Pinecone index not initialized. Attempting to reconnect.")
            self._connect_to_index()
            if self._index is None: # If still None after trying to reconnect
                 raise ConnectionError("Failed to connect to Pinecone index.")
        return self._index

    async def upsert_vectors(self, vectors: List[DocumentVector], namespace: Optional[str] = None) -> Dict[str, Any]:
        """
        Upserts (inserts or updates) vectors into the Pinecone index.
        """
        if not vectors:
            return {"upserted_count": 0}

        pinecone_vectors = [(v.id, v.values, v.metadata) for v in vectors]
        try:
            logger.info(f"Upserting {len(pinecone_vectors)} vectors to namespace '{namespace or 'default'}'.")
            upsert_response = self.index.upsert(vectors=pinecone_vectors, namespace=namespace)
            logger.info(f"Upsert response: {upsert_response}")
            return upsert_response.to_dict() if hasattr(upsert_response, "to_dict") else vars(upsert_response)
        except ApiException as e:
            logger.error(f"Pinecone API error during upsert: {e}")
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred during upsert: {e}")
            raise

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
        try:
            logger.info(f"Querying namespace '{namespace or 'default'}' with top_k={top_k}.")
            query_response = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                namespace=namespace,
                filter=filter_metadata,
                include_metadata=True,
                include_values=False # Typically not needed for query results, saves bandwidth
            )
            logger.info(f"Query response: {len(query_response.matches)} matches found.")
            # Convert Match objects to dictionaries for easier use
            return [match.to_dict() for match in query_response.matches]
        except ApiException as e:
            logger.error(f"Pinecone API error during query: {e}")
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred during query: {e}")
            raise

    async def delete_index(self):
        """Deletes the entire Pinecone index."""
        try:
            if self.index_name in self.pinecone.list_indexes().names:
                logger.info(f"Deleting index '{self.index_name}'.")
                self.pinecone.delete_index(self.index_name)
                logger.info(f"Index '{self.index_name}' deleted successfully.")
                self._index = None
            else:
                logger.info(f"Index '{self.index_name}' not found, skipping deletion.")
        except ApiException as e:
            logger.error(f"Pinecone API error during index deletion: {e}")
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred during index deletion: {e}")
            raise

    async def delete_vectors(self, ids: List[str], namespace: Optional[str] = None) -> Dict[str, Any]:
        """Deletes vectors by ID from the specified namespace."""
        if not ids:
            return {}
        try:
            logger.info(f"Deleting {len(ids)} vectors from namespace '{namespace or 'default'}'.")
            delete_response = self.index.delete(ids=ids, namespace=namespace)
            # The response for delete is often empty or just a success indicator
            logger.info(f"Delete response: {delete_response}")
            return delete_response if isinstance(delete_response, dict) else {} # Ensure dict
        except ApiException as e:
            logger.error(f"Pinecone API error during vector deletion: {e}")
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred during vector deletion: {e}")
            raise

# Example Usage (for testing purposes, typically not in the class file)
async def main():
    # This requires PINECONE_API_KEY environment variable to be set
    import os
    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        print("Please set the PINECONE_API_KEY environment variable.")
        return

    # Ensure you have a serverless environment or configure for pod-based
    # For serverless, environment is not strictly needed for Pinecone client init
    # but good to be aware of.
    # For pod-based, you would need PINECONE_ENVIRONMENT.
    # store = PineconeStore(api_key=api_key, environment="your-pinecone-environment", index_name="my-test-index", dimension=3)

    # Using serverless defaults for cloud and region
    store = PineconeStore(api_key=api_key, index_name="my-lean-test-index", dimension=128, cloud="aws", region="us-west-2")


    try:
        print(f"Pinecone index instance: {store.index}")

        # Upsert example
        vectors_to_upsert = [
            DocumentVector(id="vec1", values=[0.1] * 128, metadata={"genre": "drama"}),
            DocumentVector(id="vec2", values=[0.2] * 128, metadata={"genre": "comedy"}),
            DocumentVector(id="vec3", values=[0.3] * 128, metadata={"genre": "drama"}),
        ]
        upsert_result = await store.upsert_vectors(vectors_to_upsert, namespace="test-ns")
        print(f"Upsert result: {upsert_result}")

        # Give some time for indexing
        import time
        time.sleep(5)


        # Query example
        query_emb = [0.11] * 128
        query_results = await store.query_vectors(query_embedding=query_emb, top_k=2, namespace="test-ns")
        print(f"Query results for [0.11...]: {query_results}")

        query_results_filtered = await store.query_vectors(
            query_embedding=query_emb,
            top_k=2,
            namespace="test-ns",
            filter_metadata={"genre": "comedy"}
        )
        print(f"Query results filtered for genre 'comedy': {query_results_filtered}")

        # Delete vectors by ID
        # delete_vec_result = await store.delete_vectors(ids=["vec1"], namespace="test-ns")
        # print(f"Delete vector 'vec1' result: {delete_vec_result}")
        # time.sleep(2)
        # query_results_after_delete = await store.query_vectors(query_embedding=query_emb, top_k=2, namespace="test-ns")
        # print(f"Query results after deleting vec1: {query_results_after_delete}")


    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Clean up: delete the index
        # print("Attempting to delete index...")
        # await store.delete_index()
        pass

if __name__ == "__main__":
    import asyncio
    # asyncio.run(main()) # Comment out for non-testing use
    pass
