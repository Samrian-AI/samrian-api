import logging
from typing import List, Dict, Any, Optional, Union, Tuple

# Repurposed components
from ..ingestion.morphik_parser import MorphikParser
from ..embedding.colpali_embedding_model import ColpaliEmbeddingModel
from ..models.chunk import Chunk # Assuming Chunk is used by parser and for embedding

# New storage adapters
from ..storage.pinecone_store import PineconeStore, DocumentVector as PineconeDocumentVector
from ..storage.neo4j_store import Neo4jStore, Neo4jNode, Neo4jRelationship

# For LLM-based entity/relationship extraction for Neo4j (conceptual)
# This would be the refactored version of graph_service.extract_entities_from_text
# from ..graph_extraction.llm_extractor import LlmGraphExtractor # Example path

logger = logging.getLogger(__name__)

class RAGOrchestrator:
    def __init__(self,
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
                 pinecone_api_key: str,
                 pinecone_environment: Optional[str] = None, # Optional for serverless
                 pinecone_index_name: str = "lean-morphik-rag",
                 pinecone_dimension: int = 1024, # This should match Colpali's output dimension
                 pinecone_cloud: str = "aws",
                 pinecone_region: str = "us-east-1",
                 # Neo4j configuration
                 neo4j_uri: str,
                 neo4j_user: Optional[str] = None,
                 neo4j_password: Optional[str] = None,
                 neo4j_database: str = "neo4j",
                 # LLM for Graph Extraction (if used directly by orchestrator)
                 # graph_llm_model_name: Optional[str] = "gpt-3.5-turbo", # Example
                 # graph_llm_api_key: Optional[str] = None
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
        # TODO: Confirm Colpali output dimension and ensure pinecone_dimension matches.
        # For "tsystems/colqwen2.5-3b-multilingual-v1.0", the dimension is 1024.

        self.pinecone_store = PineconeStore(
            api_key=pinecone_api_key,
            environment=pinecone_environment,
            index_name=pinecone_index_name,
            dimension=pinecone_dimension, # Should match embedding model output
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

        # Conceptual: Initialize LLM Graph Extractor if it's a separate component
        # self.llm_graph_extractor = LlmGraphExtractor(
        # model_name=graph_llm_model_name,
        # api_key=graph_llm_api_key
        # )
        # logger.info("LlmGraphExtractor initialized.")

        logger.info("RAGOrchestrator initialization complete.")

    async def ingest_file(self, file_path: str, file_name: str, document_id: str, pinecone_namespace: Optional[str] = None):
        """
        Processes a single file: parses, chunks, embeds, and stores in Pinecone and Neo4j.
        """
        logger.info(f"Starting ingestion for file: {file_name} (doc_id: {document_id}) from path: {file_path}")

        try:
            with open(file_path, "rb") as f:
                file_bytes = f.read()
        except IOError as e:
            logger.error(f"Error reading file {file_path}: {e}")
            raise

        # 1. Parse file to text
        # metadata_from_parser will be dict, text_content will be str
        logger.info(f"Parsing file {file_name} to text...")
        metadata_from_parser, text_content = await self.parser.parse_file_to_text(file_bytes, file_name)
        if not text_content.strip():
            logger.warning(f"No text content extracted from {file_name}. Skipping further processing for this file.")
            return

        # 2. Split text into chunks (List[Chunk])
        # Chunk objects have 'content' and 'metadata'
        logger.info(f"Splitting text content from {file_name} into chunks...")
        chunks: List[Chunk] = await self.parser.split_text(text_content)
        if not chunks:
            logger.warning(f"No chunks generated from {file_name}. Skipping embedding and storage for this file.")
            return

        logger.info(f"Generated {len(chunks)} chunks for {file_name}.")

        # 3. Embed chunks (List[np.ndarray])
        logger.info(f"Embedding {len(chunks)} chunks for {file_name}...")
        chunk_embeddings = await self.embedding_model.embed_for_ingestion(chunks)
        logger.info(f"Successfully embedded {len(chunk_embeddings)} chunks.")

        # 4. Store in Pinecone
        pinecone_vectors = []
        for i, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings)):
            # Ensure embedding is List[float] for PineconeDocumentVector
            embedding_list = embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)

            chunk_metadata = {
                "document_id": document_id,
                "file_name": file_name,
                "chunk_number": i,
                **chunk.metadata, # Original chunk metadata
                **metadata_from_parser # Metadata from the initial parsing step
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

        # 5. (Conceptual) Extract graph data from chunks/text and store in Neo4j
        # This part requires the LlmGraphExtractor component.
        # For now, let's simulate adding some basic nodes representing the document and chunks.
        logger.info(f"Storing graph data in Neo4j for {file_name} (doc_id: {document_id})...")

        # Add a node for the document itself
        doc_node = Neo4jNode(label="Document", properties={"id": document_id, "name": file_name, **metadata_from_parser})
        await self.neo4j_store.add_node(doc_node)
        logger.info(f"Added Document node to Neo4j: {document_id}")

        for i, chunk in enumerate(chunks):
            chunk_node_id = f"{document_id}_chunk_{i}"
            chunk_node_props = {
                "id": chunk_node_id,
                "document_id": document_id,
                "chunk_number": i,
                "text_preview": chunk.content[:100] + "..." # Store a preview
                # Add more chunk-specific metadata if needed
            }
            chunk_neo4j_node = Neo4jNode(label="Chunk", properties=chunk_node_props)
            await self.neo4j_store.add_node(chunk_neo4j_node)

            # Add relationship from Document to Chunk
            doc_to_chunk_rel = Neo4jRelationship(
                source_node_label="Document",
                source_node_properties={"id": document_id},
                target_node_label="Chunk",
                target_node_properties={"id": chunk_node_id},
                type="HAS_CHUNK"
            )
            await self.neo4j_store.add_relationship(doc_to_chunk_rel)
        logger.info(f"Added {len(chunks)} Chunk nodes and relationships to Neo4j for {file_name}.")

        # TODO: Implement actual LLM-based entity/relationship extraction here
        # entities, relationships = await self.llm_graph_extractor.extract_from_text(text_content, document_id)
        # for entity in entities:
        #     await self.neo4j_store.add_node(entity) # Assuming LlmGraphExtractor returns Neo4jNode compatible objects
        # for rel in relationships:
        #     await self.neo4j_store.add_relationship(rel) # Assuming LlmGraphExtractor returns Neo4jRelationship compatible

        logger.info(f"Ingestion complete for file: {file_name}")


    async def query(self,
                    query_text: str,
                    pinecone_top_k: int = 5,
                    pinecone_namespace: Optional[str] = None,
                    pinecone_filter_metadata: Optional[Dict[str, Any]] = None,
                    # Neo4j query parameters (conceptual for now)
                    # neo4j_query_cypher: Optional[str] = None,
                    # neo4j_query_params: Optional[Dict[str, Any]] = None,
                    # LLM for answer synthesis
                    synthesis_llm_model: str = "gpt-3.5-turbo", # Example
                    synthesis_llm_api_key: Optional[str] = None,
                    synthesis_llm_base_url: Optional[str] = None
                   ) -> Dict[str, Any]:
        """
        Performs a RAG query:
        1. Embeds the query.
        2. Retrieves relevant chunks from Pinecone.
        3. (Conceptual) Retrieves relevant graph data from Neo4j.
        4. Synthesizes an answer using an LLM with the retrieved context.
        """
        logger.info(f"Received query: '{query_text}'")

        # 1. Embed the query
        logger.info("Embedding query...")
        query_embedding_tensor = await self.embedding_model.embed_for_query(query_text)
        query_embedding: List[float] = query_embedding_tensor.tolist() if hasattr(query_embedding_tensor, 'tolist') else list(query_embedding_tensor)
        logger.info("Query embedded successfully.")

        # 2. Retrieve from Pinecone
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
            # Pinecone returns metadata, which should contain the original chunk text or enough to reconstruct it
            # For this example, let's assume 'text' is directly in metadata.
            # This needs to align with how data was upserted in `ingest_file`.
            # The current PineconeStore example upserts DocumentVector which has 'id', 'values', 'metadata'.
            # The 'metadata' in DocumentVector comes from chunk_metadata in ingest_file.
            # That chunk_metadata includes the original chunk.content.
            # However, PineconeStore.query_vectors returns Match objects.
            # Match object has id, score, values (optional), metadata (optional).
            # Let's adjust to get text if it was stored.

            # The `ingest_file` stores `chunk.content` inside the `chunk_metadata` which is then
            # passed as `metadata` to `PineconeDocumentVector`.
            # Pinecone's query result `match.metadata` should contain this.
            # A better way would be to have a dedicated 'text' field in the metadata.
            # For now, let's assume we need to reconstruct or find it.
            # The MorphikParser chunk.content is what we need.

            # The current PineconeStore example doesn't explicitly store the raw text with the vector.
            # Let's assume for now the orchestrator would need to fetch the text based on IDs,
            # OR we modify PineconeStore/ingestion to store text.
            # For simplicity in this orchestrator, we'll assume metadata contains 'text'.
            # THIS IS A SIMPLIFICATION. In a real app, ensure text is stored or retrievable.
            if match.get('metadata') and 'text_preview' in match['metadata']: # Assuming text_preview was stored from chunk
                 retrieved_chunks_content.append(match['metadata']['text_preview'])
            elif match.get('metadata') and 'original_content' in match['metadata']: # Or a more complete field
                 retrieved_chunks_content.append(match['metadata']['original_content'])
            else:
                # Fallback or warning if text not found
                logger.warning(f"Could not find text content for Pinecone match ID {match.get('id')}. Metadata: {match.get('metadata')}")


        # 3. (Conceptual) Retrieve from Neo4j
        neo4j_context_str = ""
        # if neo4j_query_cypher:
        #     logger.info(f"Querying Neo4j with Cypher: {neo4j_query_cypher}")
        #     neo4j_raw_results = await self.neo4j_store.run_cypher_query(neo4j_query_cypher, neo4j_query_params)
        #     # Process neo4j_raw_results into a string or structured data for the LLM
        #     neo4j_context_str = " ".join([str(item) for item in neo4j_raw_results]) # Simple conversion
        #     logger.info(f"Retrieved context from Neo4j: {neo4j_context_str[:100]}...")

        # 4. Synthesize answer using LLM
        # Prepare context for LLM
        context_for_llm = "\n\n---\n\n".join(retrieved_chunks_content)
        if neo4j_context_str:
            context_for_llm += f"\n\n--- Graph Context ---\n{neo4j_context_str}"

        if not context_for_llm.strip():
            logger.warning("No context retrieved from vector store or graph store. LLM will answer without specific context.")
            # Optionally return a message indicating no context, or let LLM try without.
            # For now, let LLM try.
            # return {"answer": "I could not find any relevant information to answer your query.", "sources": []}


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
        # Use a generic LLM client or litellm directly here
        import litellm
        try:
            llm_params = {
                "model": synthesis_llm_model,
                "messages": [
                    {"role": "system", "content": "You are a helpful AI assistant."},
                    {"role": "user", "content": final_prompt}
                ],
                "max_tokens": 500, # Configurable
                "temperature": 0.7 # Configurable
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
            # Consider re-raising or returning a more specific error structure

        return {
            "answer": answer,
            "retrieved_pinecone_matches": pinecone_results, # Full match objects
            # "retrieved_neo4j_results": neo4j_raw_results if neo4j_query_cypher else []
        }

    async def close_stores(self):
        """Closes connections to the data stores."""
        logger.info("Closing store connections...")
        # Pinecone client does not have an explicit close method in v3.x for HTTP.
        # Connections are managed per-request.
        # If using gRPC client, it would have a close method.
        if self.neo4j_store:
            await self.neo4j_store.close()
        logger.info("Store connections closed.")


# Example Usage (for testing purposes, typically not in the class file)
async def example_orchestrator_run():
    import os
    from dotenv import load_dotenv
    load_dotenv()

    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    neo4j_uri = os.environ.get("NEO4J_URI")
    neo4j_user = os.environ.get("NEO4J_USER")
    neo4j_password = os.environ.get("NEO4J_PASSWORD")
    # Optional: AssemblyAI API key for video parsing in MorphikParser
    assemblyai_key = os.environ.get("ASSEMBLYAI_API_KEY")
    # Optional: OpenAI API key for LLM synthesis (if using OpenAI model via litellm)
    openai_key = os.environ.get("OPENAI_API_KEY")


    if not all([pinecone_api_key, neo4j_uri, neo4j_user, neo4j_password]):
        print("Missing one or more environment variables: PINECONE_API_KEY, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD")
        return

    orchestrator = RAGOrchestrator(
        pinecone_api_key=pinecone_api_key,
        pinecone_index_name="lean-morphik-demo-idx", # Use a unique index name
        pinecone_dimension=1024, # Ensure this matches Colpali output
        neo4j_uri=neo4j_uri,
        neo4j_user=neo4j_user,
        neo4j_password=neo4j_password,
        parser_assemblyai_api_key=assemblyai_key, # Pass AssemblyAI key
        synthesis_llm_api_key=openai_key # Pass OpenAI key for litellm
    )

    # Create a dummy file for ingestion
    dummy_file_path = "dummy_document.txt"
    with open(dummy_file_path, "w") as f:
        f.write("This is a test document about apples and oranges. Apples are red, oranges are orange. Both are fruits.")

    try:
        # Ingest the dummy file
        print(f"\n--- Ingesting {dummy_file_path} ---")
        await orchestrator.ingest_file(file_path=dummy_file_path, file_name="dummy_document.txt", document_id="doc001")
        print("Ingestion complete.")

        # Give Pinecone a moment to index (especially for new indexes)
        import time
        print("Waiting for indexing...")
        time.sleep(10)

        # Query
        print("\n--- Querying ---")
        query_result = await orchestrator.query(
            query_text="Tell me about apples.",
            pinecone_top_k=2,
            synthesis_llm_model="gpt-3.5-turbo" # Ensure litellm can access this
        )
        print("\nQuery Result:")
        print(f"  Answer: {query_result.get('answer')}")
        print(f"  Pinecone Matches ({len(query_result.get('retrieved_pinecone_matches', []))}):")
        for i, match in enumerate(query_result.get('retrieved_pinecone_matches', [])):
            print(f"    Match {i+1}: ID={match.get('id')}, Score={match.get('score', 'N/A')}")
            # print(f"      Metadata: {match.get('metadata')}")


    except Exception as e:
        print(f"\nAn error occurred during the example run: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up dummy file
        if os.path.exists(dummy_file_path):
            os.remove(dummy_file_path)

        # Clean up Pinecone index (optional, for testing)
        # print("\n--- Attempting to delete Pinecone index ---")
        # await orchestrator.pinecone_store.delete_index()

        # Close Neo4j connection
        await orchestrator.close_stores()
        print("\nExample run finished.")

if __name__ == "__main__":
    import asyncio
    # To run the example:
    # 1. Ensure you have .env file with PINECONE_API_KEY, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
    #    And optionally ASSEMBLYAI_API_KEY, OPENAI_API_KEY (or other LLM provider key for litellm)
    # 2. Install dependencies: pip install python-dotenv pinecone-client neo4j torch sentence-transformers Pillow litellm openai unstructured filetype
    # asyncio.run(example_orchestrator_run())
    pass
