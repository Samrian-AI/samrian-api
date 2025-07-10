import logging
from typing import List, Dict, Any, Optional, Union, Tuple

# Repurposed components
from ..ingestion.morphik_parser import MorphikParser
from ..embedding.colpali_embedding_model import ColpaliEmbeddingModel
from ..models.chunk import Chunk # Assuming Chunk is used by parser and for embedding

# Storage Abstractions and their concrete implementations
from ..storage.base_vector_store import BaseVectorStore, DocumentVector
from ..storage.base_graph_store import BaseGraphStore, Node, Relationship
# Concrete stores are not directly imported here if passed via DI, but their types might be needed for hints
# from ..storage.pinecone_store import PineconeStore
# from ..storage.neo4j_store import Neo4jStore


logger = logging.getLogger(__name__)

class RAGOrchestrator:
    def __init__(self,
                 parser: MorphikParser,
                 embedding_model: ColpaliEmbeddingModel,
                 vector_store: BaseVectorStore,
                 graph_store: BaseGraphStore
                ):
        """
        Initializes the RAGOrchestrator with dependency injection.
        Components (parser, embedding_model, vector_store, graph_store)
        should be pre-configured and passed in.
        """
        logger.info("Initializing RAGOrchestrator with injected components...")

        self.parser = parser
        logger.info("MorphikParser provided.")

        self.embedding_model = embedding_model
        logger.info("ColpaliEmbeddingModel provided.")

        self.vector_store = vector_store # Expects an instance conforming to BaseVectorStore
        logger.info("VectorStore (e.g., PineconeStore) provided.")

        self.graph_store = graph_store # Expects an instance conforming to BaseGraphStore
        logger.info("GraphStore (e.g., Neo4jStore) provided.")

        logger.info("RAGOrchestrator initialization complete.")

    async def ingest_file(self, file_path: str, file_name: str, document_id: str, vector_store_namespace: Optional[str] = None):
        logger.info(f"Starting ingestion for file: {file_name} (doc_id: {document_id}) from path: {file_path}")
        try:
            with open(file_path, "rb") as f:
                file_bytes = f.read()
        except IOError as e:
            logger.error(f"Error reading file {file_path}: {e}")
            raise

        logger.info(f"Parsing file {file_name} to text...")
        metadata_from_parser, text_content = await self.parser.parse_file_to_text(file_bytes, file_name)
        if not text_content.strip():
            logger.warning(f"No text content extracted from {file_name}. Skipping further processing for this file.")
            return

        logger.info(f"Splitting text content from {file_name} into chunks...")
        chunks: List[Chunk] = await self.parser.split_text(text_content)
        if not chunks:
            logger.warning(f"No chunks generated from {file_name}. Skipping embedding and storage for this file.")
            return

        logger.info(f"Generated {len(chunks)} chunks for {file_name}.")
        logger.info(f"Embedding {len(chunks)} chunks for {file_name}...")
        chunk_embeddings = await self.embedding_model.embed_for_ingestion(chunks)
        logger.info(f"Successfully embedded {len(chunk_embeddings)} chunks.")

        # Use the generic DocumentVector for the list, as defined by BaseVectorStore
        document_vectors_to_upsert: List[DocumentVector] = []
        for i, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings)):
            embedding_list = embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)
            chunk_metadata = {
                "document_id": document_id,
                "file_name": file_name,
                "chunk_number": i,
                "original_content": chunk.content,
                **chunk.metadata,
                **metadata_from_parser
            }
            document_vectors_to_upsert.append(
                DocumentVector( # Using the generic DocumentVector from base_vector_store
                    id=f"{document_id}_chunk_{i}",
                    values=embedding_list,
                    metadata=chunk_metadata
                )
            )

        if document_vectors_to_upsert:
            logger.info(f"Upserting {len(document_vectors_to_upsert)} vectors to VectorStore (namespace: {vector_store_namespace})...")
            # Use self.vector_store (which is BaseVectorStore)
            await self.vector_store.upsert_vectors(document_vectors_to_upsert, namespace=vector_store_namespace)
            logger.info(f"Successfully upserted vectors to VectorStore for {file_name}.")
        else:
            logger.warning(f"No vectors to upsert to VectorStore for {file_name}.")

        logger.info(f"Storing graph data in GraphStore for {file_name} (doc_id: {document_id})...")
        # Use the generic Node and Relationship from base_graph_store
        doc_node = Node(label="Document", properties={"id": document_id, "name": file_name, **metadata_from_parser})
        await self.graph_store.add_node(doc_node) # Use self.graph_store
        logger.info(f"Added Document node to GraphStore: {document_id}")

        for i, chunk in enumerate(chunks):
            chunk_node_id = f"{document_id}_chunk_{i}"
            chunk_node_props = {
                "id": chunk_node_id,
                "document_id": document_id,
                "chunk_number": i,
                "text_preview": chunk.content[:100] + "..." # text_preview is fine for graph node
            }
            chunk_graph_node = Node(label="Chunk", properties=chunk_node_props)
            await self.graph_store.add_node(chunk_graph_node) # Use self.graph_store

            doc_to_chunk_rel = Relationship(
                source_node_label="Document",
                source_node_properties={"id": document_id},
                target_node_label="Chunk",
                target_node_properties={"id": chunk_node_id},
                type="HAS_CHUNK",
                properties={} # Add empty properties dict if none
            )
            await self.graph_store.add_relationship(doc_to_chunk_rel) # Use self.graph_store
        logger.info(f"Added {len(chunks)} Chunk nodes and relationships to GraphStore for {file_name}.")
        logger.info(f"Ingestion complete for file: {file_name}")

    async def query(self,
                    query_text: str,
                    vector_store_top_k: int = 5, # Renamed for clarity
                    vector_store_namespace: Optional[str] = None, # Renamed for clarity
                    vector_store_filter_metadata: Optional[Dict[str, Any]] = None, # Renamed for clarity
                    synthesis_llm_model: str = "gpt-3.5-turbo",
                    synthesis_llm_api_key: Optional[str] = None,
                    synthesis_llm_base_url: Optional[str] = None
                   ) -> Dict[str, Any]:
        logger.info(f"Received query: '{query_text}'")
        logger.info("Embedding query...")
        query_embedding_tensor = await self.embedding_model.embed_for_query(query_text)
        query_embedding: List[float] = query_embedding_tensor.tolist() if hasattr(query_embedding_tensor, 'tolist') else list(query_embedding_tensor)
        logger.info("Query embedded successfully.")

        logger.info(f"Querying VectorStore (top_k={vector_store_top_k}, namespace={vector_store_namespace})...")
        # Use self.vector_store (which is BaseVectorStore)
        retrieved_vector_results = await self.vector_store.query_vectors(
            query_embedding=query_embedding,
            top_k=vector_store_top_k,
            namespace=vector_store_namespace,
            filter_metadata=vector_store_filter_metadata
        )
        logger.info(f"Retrieved {len(retrieved_vector_results)} results from VectorStore.")

        retrieved_chunks_content = []
        for match in retrieved_vector_results:
            if match.get('metadata') and 'original_content' in match['metadata']:
                 retrieved_chunks_content.append(match['metadata']['original_content'])
            elif match.get('metadata') and 'text_preview' in match['metadata']: # Fallback if original_content not there
                 retrieved_chunks_content.append(match['metadata']['text_preview'])
            else:
                logger.warning(f"Could not find text content for VectorStore match ID {match.get('id')}. Metadata: {match.get('metadata')}")

        context_for_llm = "\n\n---\n\n".join(retrieved_chunks_content)
        if not context_for_llm.strip(): # Check if context is empty or only whitespace
            logger.warning("No context retrieved from VectorStore. LLM will answer without specific context.")

        prompt_template = """
        You are a helpful AI assistant. Answer the user's query based on the provided context.
        If the context does not contain the answer, say "I don't have enough information to answer that."
        Do not make up information.

        Context:
        {context}

        Query: {query}

        Answer:
        """
        final_prompt = prompt_template.format(context=context_for_llm, query=query_text)

        logger.info("Synthesizing answer with LLM...")
        import litellm
        try:
            llm_params = {
                "model": synthesis_llm_model,
                "messages": [
                    {"role": "system", "content": "You are a helpful AI assistant."},
                    {"role": "user", "content": final_prompt}
                ],
                "max_tokens": 500,
                "temperature": 0.7
            }
            if synthesis_llm_api_key:
                llm_params["api_key"] = synthesis_llm_api_key
            if synthesis_llm_base_url:
                llm_params["base_url"] = synthesis_llm_base_url

            response = await litellm.acompletion(**llm_params)
            answer = response.choices[0].message.content
            logger.info("LLM synthesis complete.")
        except Exception as e:
            logger.error(f"Error during LLM synthesis: {e}")
            answer = "There was an error generating the answer." # Generic error message

        return {
            "answer": answer,
            "retrieved_vector_store_matches": retrieved_vector_results, # Return results from vector store
        }

    async def close_stores(self):
        logger.info("Closing store connections if applicable...")
        # Graph store has an explicit close method as per BaseGraphStore
        if self.graph_store and hasattr(self.graph_store, 'close'):
            await self.graph_store.close()
        # Vector stores might not always have an explicit close (e.g. HTTP based clients like Pinecone)
        # but if BaseVectorStore defined a close(), it would be called here.
        if self.vector_store and hasattr(self.vector_store, 'close'): # Check if close method exists
             await self.vector_store.close() # type: ignore # Assuming close method exists if defined in an ABC
        logger.info("Store connections handled.")

# Commented out example to keep the file clean for library use
# async def example_orchestrator_run():
#     pass # Actual example code omitted for brevity but would be here
# if __name__ == "__main__":
#     pass # Actual example code omitted
    pass
