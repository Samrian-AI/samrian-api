import logging
from typing import List, Dict, Any, Optional, Union, Tuple

# Repurposed components
from ..ingestion.morphik_parser import MorphikParser
from ..embedding.colpali_embedding_model import ColpaliEmbeddingModel
from ..models.chunk import Chunk # Assuming Chunk is used by parser and for embedding

# New storage adapters
from ..storage.pinecone_store import PineconeStore, DocumentVector as PineconeDocumentVector
from ..storage.neo4j_store import Neo4jStore, Neo4jNode, Neo4jRelationship

logger = logging.getLogger(__name__)

class RAGOrchestrator:
    def __init__(self,
                 # Required arguments first
                 pinecone_api_key: str,
                 neo4j_uri: str,
                 # Parser configuration
                 parser_chunk_size: int = 1000,
                 parser_chunk_overlap: int = 200,
                 parser_use_unstructured_api: bool = False,
                 parser_unstructured_api_key: Optional[str] = None,
                 parser_assemblyai_api_key: Optional[str] = None,
                 parser_video_frame_sample_rate: int = 120,
                 parser_use_contextual_chunking: bool = False,
                 parser_contextual_chunking_llm_model_name: Optional[str] = "claude-3-sonnet-20240229",
                 parser_contextual_chunking_llm_api_key: Optional[str] = None,
                 parser_contextual_chunking_llm_base_url: Optional[str] = None,
                 parser_video_vision_model_name: Optional[str] = "gpt-4-vision-preview",
                 parser_video_vision_api_key: Optional[str] = None,
                 parser_video_vision_base_url: Optional[str] = None,
                 # Embedding model configuration
                 embedding_device: Optional[str] = None,
                 embedding_model_name: str = "tsystems/colqwen2.5-3b-multilingual-v1.0",
                 embedding_processor_name: str = "tsystems/colqwen2.5-3b-multilingual-v1.0",
                 embedding_batch_size: int = 1,
                 # Pinecone configuration
                 pinecone_environment: Optional[str] = None,
                 pinecone_index_name: str = "lean-morphik-rag",
                 pinecone_dimension: int = 1024,
                 pinecone_cloud: str = "aws",
                 pinecone_region: str = "us-east-1",
                 # Neo4j configuration
                 neo4j_user: Optional[str] = None,
                 neo4j_password: Optional[str] = None,
                 neo4j_database: str = "neo4j"
                ):

        logger.info("Initializing RAGOrchestrator...")

        self.parser = MorphikParser(
            chunk_size=parser_chunk_size,
            chunk_overlap=parser_chunk_overlap,
            use_unstructured_api=parser_use_unstructured_api,
            unstructured_api_key=parser_unstructured_api_key,
            assemblyai_api_key=parser_assemblyai_api_key,
            video_frame_sample_rate=parser_video_frame_sample_rate,
            use_contextual_chunking=parser_use_contextual_chunking,
            contextual_chunking_llm_model_name=parser_contextual_chunking_llm_model_name,
            contextual_chunking_llm_api_key=parser_contextual_chunking_llm_api_key,
            contextual_chunking_llm_base_url=parser_contextual_chunking_llm_base_url,
            video_vision_model_name=parser_video_vision_model_name,
            video_vision_api_key=parser_video_vision_api_key,
            video_vision_base_url=parser_video_vision_base_url,
        )
        logger.info("MorphikParser initialized.")

        self.embedding_model = ColpaliEmbeddingModel(
            device=embedding_device,
            model_name=embedding_model_name,
            processor_name=embedding_processor_name,
            batch_size=embedding_batch_size
        )
        logger.info("ColpaliEmbeddingModel initialized.")

        self.pinecone_store = PineconeStore(
            api_key=pinecone_api_key,
            environment=pinecone_environment,
            index_name=pinecone_index_name,
            dimension=pinecone_dimension,
            cloud=pinecone_cloud,
            region=pinecone_region
        )
        logger.info("PineconeStore initialized.")

        self.neo4j_store = Neo4jStore(
            uri=neo4j_uri,
            user=neo4j_user,
            password=neo4j_password,
            database=neo4j_database
        )
        logger.info("Neo4jStore initialized.")
        logger.info("RAGOrchestrator initialization complete.")

    async def ingest_file(self, file_path: str, file_name: str, document_id: str, pinecone_namespace: Optional[str] = None):
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

        pinecone_vectors = []
        for i, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings)):
            embedding_list = embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)
            # Storing full chunk content in metadata for easier retrieval in RAG
            chunk_metadata = {
                "document_id": document_id,
                "file_name": file_name,
                "chunk_number": i,
                "original_content": chunk.content, # Ensure full content is stored
                **chunk.metadata,
                **metadata_from_parser
            }
            pinecone_vectors.append(
                PineconeDocumentVector(
                    id=f"{document_id}_chunk_{i}",
                    values=embedding_list,
                    metadata=chunk_metadata
                )
            )

        if pinecone_vectors:
            logger.info(f"Upserting {len(pinecone_vectors)} vectors to Pinecone (namespace: {pinecone_namespace})...")
            await self.pinecone_store.upsert_vectors(pinecone_vectors, namespace=pinecone_namespace)
            logger.info(f"Successfully upserted vectors to Pinecone for {file_name}.")
        else:
            logger.warning(f"No vectors to upsert to Pinecone for {file_name}.")

        logger.info(f"Storing graph data in Neo4j for {file_name} (doc_id: {document_id})...")
        doc_node = Neo4jNode(label="Document", properties={"id": document_id, "name": file_name, **metadata_from_parser})
        await self.neo4j_store.add_node(doc_node)
        logger.info(f"Added Document node to Neo4j: {document_id}")

        for i, chunk in enumerate(chunks):
            chunk_node_id = f"{document_id}_chunk_{i}"
            chunk_node_props = {
                "id": chunk_node_id,
                "document_id": document_id,
                "chunk_number": i,
                "text_preview": chunk.content[:100] + "..."
            }
            chunk_neo4j_node = Neo4jNode(label="Chunk", properties=chunk_node_props)
            await self.neo4j_store.add_node(chunk_neo4j_node)
            doc_to_chunk_rel = Neo4jRelationship(
                source_node_label="Document",
                source_node_properties={"id": document_id},
                target_node_label="Chunk",
                target_node_properties={"id": chunk_node_id},
                type="HAS_CHUNK"
            )
            await self.neo4j_store.add_relationship(doc_to_chunk_rel)
        logger.info(f"Added {len(chunks)} Chunk nodes and relationships to Neo4j for {file_name}.")
        logger.info(f"Ingestion complete for file: {file_name}")

    async def query(self,
                    query_text: str,
                    pinecone_top_k: int = 5,
                    pinecone_namespace: Optional[str] = None,
                    pinecone_filter_metadata: Optional[Dict[str, Any]] = None,
                    synthesis_llm_model: str = "gpt-3.5-turbo",
                    synthesis_llm_api_key: Optional[str] = None,
                    synthesis_llm_base_url: Optional[str] = None
                   ) -> Dict[str, Any]:
        logger.info(f"Received query: '{query_text}'")
        logger.info("Embedding query...")
        query_embedding_tensor = await self.embedding_model.embed_for_query(query_text)
        query_embedding: List[float] = query_embedding_tensor.tolist() if hasattr(query_embedding_tensor, 'tolist') else list(query_embedding_tensor)
        logger.info("Query embedded successfully.")

        logger.info(f"Querying Pinecone (top_k={pinecone_top_k}, namespace={pinecone_namespace})...")
        pinecone_results = await self.pinecone_store.query_vectors(
            query_embedding=query_embedding,
            top_k=pinecone_top_k,
            namespace=pinecone_namespace,
            filter_metadata=pinecone_filter_metadata
        )
        logger.info(f"Retrieved {len(pinecone_results)} results from Pinecone.")

        retrieved_chunks_content = []
        for match in pinecone_results:
            if match.get('metadata') and 'original_content' in match['metadata']: # Prioritize full content
                 retrieved_chunks_content.append(match['metadata']['original_content'])
            elif match.get('metadata') and 'text_preview' in match['metadata']:
                 retrieved_chunks_content.append(match['metadata']['text_preview'])
            else:
                logger.warning(f"Could not find text content for Pinecone match ID {match.get('id')}. Metadata: {match.get('metadata')}")

        context_for_llm = "\n\n---\n\n".join(retrieved_chunks_content)
        if not context_for_llm.strip():
            logger.warning("No context retrieved from vector store or graph store. LLM will answer without specific context.")

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
            answer = "There was an error generating the answer."

        return {
            "answer": answer,
            "retrieved_pinecone_matches": pinecone_results,
        }

    async def close_stores(self):
        logger.info("Closing store connections...")
        if self.neo4j_store:
            await self.neo4j_store.close()
        logger.info("Store connections closed.")

# Commented out example to keep the file clean for library use
# async def example_orchestrator_run():
#     pass # Actual example code omitted for brevity but would be here
# if __name__ == "__main__":
#     pass # Actual example code omitted
    pass
