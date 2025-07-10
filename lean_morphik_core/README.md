# Lean Morphik Core

This project contains surgically extracted and refactored core components from the original Morphik application, focusing on document ingestion, embedding, and a lightweight RAG (Retrieval Augmented Generation) orchestration.

## Overview

The `lean_morphik_core` provides a streamlined set of tools to:
1.  Parse various document types (text, PDF, DOCX, Markdown, JSON, video) into text and then into manageable chunks.
2.  Generate embeddings for these chunks using the Colpali embedding model.
3.  Store these embeddings and associated metadata in a Pinecone vector store.
4.  Store document and chunk relationships (and potentially extracted entities/relationships) in a Neo4j graph database.
5.  Orchestrate these processes and provide a RAG query interface to retrieve relevant information and synthesize answers using an LLM.

## Architecture

The system is composed of the following key modules located under the `app/` directory:

### 1. Ingestion (`app/ingestion/`)
   *   **`MorphikParser`**: Handles parsing of different file formats (including text, PDF, DOCX, video transcriptions via AssemblyAI) into raw text. It then splits this text into chunks suitable for embedding. It supports standard chunking and an optional contextual chunking mechanism that uses an LLM to add contextual information to each chunk.
       *   Uses `unstructured.io` for parsing various document types.
       *   Uses a custom `RecursiveCharacterTextSplitter` for text splitting.
       *   Can integrate with AssemblyAI for video transcription and an LLM (via `litellm`) for contextual chunking.

### 2. Embedding (`app/embedding/`)
   *   **`ColpaliEmbeddingModel`**: A wrapper around a sentence-transformer compatible model (defaulting to `tsystems/colqwen2.5-3b-multilingual-v1.0`) for generating dense vector embeddings from text chunks. It utilizes the `transformers` library.

### 3. Storage Adapters (`app/storage/`)
   *   **`PineconeStore`**: Manages interaction with a Pinecone vector database. It handles the upsertion of document chunk embeddings and querying for relevant chunks based on vector similarity.
   *   **`Neo4jStore`**: Manages interaction with a Neo4j graph database. It's used to store nodes (e.g., documents, chunks) and relationships between them (e.g., a document `HAS_CHUNK` a chunk). This can be extended to store more complex graph structures like extracted entities and their relationships.

### 4. Orchestration (`app/orchestrator/`)
   *   **`RAGOrchestrator`**: The central component that ties together parsing, embedding, storage, and retrieval.
       *   **Ingestion Flow**: Takes a file, uses `MorphikParser` to get text and then chunks, uses `ColpaliEmbeddingModel` to get embeddings, and then stores data in `PineconeStore` and `Neo4jStore`.
       *   **Query Flow (RAG)**: Takes a user query, embeds it, retrieves relevant chunks from `PineconeStore` (and potentially related data from `Neo4jStore` in future extensions), then uses an LLM (via `litellm`) to synthesize an answer based on the retrieved context.

### 5. Data Models (`app/models/`)
    *   **`Chunk`**: Represents a piece of text extracted from a document, along with its metadata.
    *   Other models as needed for data representation.

## Setup and Installation

1.  **Clone the repository (or ensure you have this `lean_morphik_core` folder).**

2.  **Create and activate a Python virtual environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

3.  **Install dependencies:**
    The project uses `pyproject.toml` for dependency management. Install the core dependencies:
    ```bash
    pip install .
    ```
    This will install all dependencies listed in the `[project.dependencies]` section of `pyproject.toml`.

4.  **Set up Environment Variables for API Keys:**
    The application requires API keys for various services. Create a `.env` file in the `lean_morphik_core` directory (or ensure these variables are set in your environment):
    ```env
    # Pinecone
    PINECONE_API_KEY="your_pinecone_api_key"
    # Optional: PINECONE_ENVIRONMENT="gcp-starter" # If using non-serverless Pinecone
    # Optional: PINECONE_CLOUD="aws" # or "gcp", "azure" for serverless
    # Optional: PINECONE_REGION="us-east-1" # for serverless

    # Neo4j
    NEO4J_URI="bolt://your_neo4j_host:port" # e.g., bolt://localhost:7687
    NEO4J_USER="your_neo4j_username"
    NEO4J_PASSWORD="your_neo4j_password"
    # Optional: NEO4J_DATABASE="neo4j" # Default database

    # LLM Provider (e.g., OpenAI, Anthropic, etc., for litellm)
    # One or more of these depending on the models you use for synthesis or contextual chunking
    OPENAI_API_KEY="your_openai_api_key"
    ANTHROPIC_API_KEY="your_anthropic_api_key"
    # ... any other keys required by litellm for your chosen models

    # AssemblyAI (for video transcription in MorphikParser)
    ASSEMBLYAI_API_KEY="your_assemblyai_api_key"

    # Unstructured.io API (if using the API version of unstructured, typically not needed for local parsing)
    # UNSTRUCTURED_API_KEY="your_unstructured_api_key"
    ```
    Refer to the `RAGOrchestrator` constructor and `MorphikParser` for specific models and their corresponding API key expectations (e.g., `parser_contextual_chunking_llm_api_key`, `synthesis_llm_api_key`). `litellm` will typically pick up keys from standard environment variable names.

## How to Run

The primary way to interact with the system is through the `RAGOrchestrator`. You can instantiate and use it in your Python scripts.

**Example Snippet (conceptual):**
```python
import asyncio
from app.orchestrator.rag_orchestrator import RAGOrchestrator

async def main():
    # Ensure your .env file is loaded or environment variables are set
    # For example, using python-dotenv:
    # from dotenv import load_dotenv
    # load_dotenv()
    # pinecone_api_key = os.getenv("PINECONE_API_KEY")
    # ... etc. for other keys

    orchestrator = RAGOrchestrator(
        # Provide necessary API keys and configuration if not using defaults
        # or if they are not picked up from environment variables by underlying classes
        pinecone_api_key="your_pinecone_key_loaded_somehow",
        # pinecone_environment="your_env", # if needed
        pinecone_index_name="my-lean-rag-index",
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="password",
        # parser_assemblyai_api_key="your_assemblyai_key", # if parsing videos
        # synthesis_llm_api_key="your_llm_provider_key" # for the query LLM
    )

    # Example: Ingest a file
    # Ensure 'dummy_document.txt' exists or use a valid path
    try:
        with open("dummy_document.txt", "w") as f:
            f.write("This is a test document about AI and RAG systems.")

        print("Ingesting document...")
        await orchestrator.ingest_file(
            file_path="dummy_document.txt",
            file_name="dummy_document.txt",
            document_id="doc001"
        )
        print("Ingestion complete.")

        # Allow some time for indexing in Pinecone (especially for new indexes)
        await asyncio.sleep(5)

        # Example: Query
        print("\nQuerying...")
        query_result = await orchestrator.query(
            query_text="What are RAG systems?",
            synthesis_llm_model="gpt-3.5-turbo" # Ensure litellm can access this
        )
        print(f"Answer: {query_result.get('answer')}")

        # Access retrieved sources if needed
        # for match in query_result.get('retrieved_pinecone_matches', []):
        #     print(f" - Source ID: {match.get('id')}, Score: {match.get('score')}")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        await orchestrator.close_stores()
        # Clean up dummy file
        import os
        if os.path.exists("dummy_document.txt"):
            os.remove("dummy_document.txt")

if __name__ == "__main__":
    # Note: Running asyncio.run(main()) here might require this file to be
    # executable or called via python -m.
    # For simplicity, this example is best run from a separate script.
    # To run this example:
    # 1. Save the above into a file like `run_example.py` in `lean_morphik_core`.
    # 2. Ensure your .env file is set up in `lean_morphik_core`.
    # 3. Run `python run_example.py`.
    pass
```
*Note: The example usage code from `rag_orchestrator.py` (the `example_orchestrator_run` function) can also serve as a more detailed example if placed in a runnable script.*


## How to Run Tests

The project uses `pytest` for testing.

1.  **Install test dependencies:**
    If you haven't already installed them with the main dependencies, or if you installed with `pip install .` (which doesn't install optional dependencies by default):
    ```bash
    pip install .[test]
    ```
    Alternatively, install manually:
    ```bash
    pip install pytest pytest-asyncio pytest-mock
    ```

2.  **Run tests:**
    Navigate to the `lean_morphik_core` directory in your terminal and run:
    ```bash
    pytest
    ```
    This will discover and run all tests in the `tests/` subdirectory.

    You can also run specific test files or tests:
    ```bash
    pytest tests/unit/test_morphik_parser.py
    pytest tests/integration/test_rag_orchestrator.py::test_query_successful
    ```

## Important Note on Pinecone Metadata
The RAG orchestrator's query functionality relies on retrieving text content from the metadata of vectors stored in Pinecone. The current `ingest_file` method in `RAGOrchestrator` stores basic document and chunk identifiers in Pinecone metadata. For the RAG loop to effectively use the exact chunk text from Pinecone during queries, consider modifying `RAGOrchestrator.ingest_file` to explicitly include `chunk.content` (or a relevant preview like `text_preview`) within the metadata of each `PineconeDocumentVector` being upserted. The integration tests for `RAGOrchestrator` highlight this aspect.

---

This `lean_morphik_core` aims to provide a robust, decoupled foundation for building advanced RAG applications.
```
