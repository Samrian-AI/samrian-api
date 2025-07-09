# Morphik RAG Extraction and Reuse Plan

This document provides a technical guide on how to extract the RAG (Retrieval-Augmented Generation) components from the Morphik project and reuse them in other projects.

## 1. Project Structure Analysis

The Morphik project is structured in a modular way. The core logic is located in the `core` directory. Here's a high-level overview of the key directories and their purpose:

*   **`core/`**: Contains the main application logic.
    *   **`core/agent/`**: Likely contains the agent logic that orchestrates the RAG pipeline.
    *   **`core/completion/`**: Handles the completion logic, interacting with LLMs.
    *   **`core/database/`**: Manages database interactions.
    *   **`core/embedding/`**: Contains the logic for generating embeddings for documents and queries. This is a key component for RAG.
    *   **`core/parser/`**: Responsible for parsing different document types (e.g., PDF, images). This is crucial for "Visual-first Retrieval" and "Diagram Intelligence."
    *   **`core/reranker/`**: Contains logic for re-ranking retrieved documents to improve relevance.
    *   **`core/routes/`**: Defines the API endpoints.
    *   **`core/services/`**: Contains the business logic of the application.
    *   **`core/storage/`**: Manages file storage.
    *   **`core/vector_store/`**: Manages the vector database, which is essential for RAG.
    *   **`core/tools/`**: Contains various tools used by the agent.
*   **`ee/`**: Likely contains enterprise-specific features.
*   **`scripts/`**: Contains various scripts for managing the project.
*   **`sdks/`**: Contains software development kits for interacting with the Morphik API.

## 2. Deep Dive into RAG Functionality

Now, let's analyze the key RAG-related functionalities of Morphik and where to find them in the codebase.

### 2.1. Visual-first Retrieval and Diagram Intelligence

This feature implies that Morphik can understand the visual layout of documents and extract information from diagrams. The key components for this are located in the `core/parser/` and `core/embedding/` directories.

*   **`core/parser/`**: This directory contains the logic for parsing different document types, including PDFs and images. The `MorphikParser` uses the `unstructured` library to parse documents and extract text and layout information. It also has a dedicated `VideoParser` for video files.
*   **`core/embedding/`**: This directory contains the logic for generating multimodal embeddings. The `ColpaliEmbeddingModel` uses the `tsystems/colqwen2.5-3b-multilingual-v1.0` model, which is a multimodal model that can generate embeddings for both text and images. This is the key to "Visual-first Retrieval" and "Direct Page Embedding."

### 2.2. Knowledge Graphs

Morphik's ability to use Knowledge Graphs suggests a structured approach to data representation. The relevant components are in `core/services/graph_service.py` and `core/tools/graph_tools.py`.

*   **`core/services/graph_service.py`**: This service is responsible for building and querying the knowledge graph. It uses an LLM to extract entities and relationships from text, resolves entities to create a clean graph, and then uses the graph to enhance retrieval.
*   **`core/tools/graph_tools.py`**: These tools are used by the agent to interact with the knowledge graph.

### 2.3. Direct Page Embedding

This feature suggests that Morphik can embed entire pages without losing context. This is implemented in the `core/embedding/` directory using the `ColpaliEmbeddingModel` which can generate embeddings for entire pages (as images).

## 3. Extraction and Reuse Plan

Now, let's outline a plan to extract the barebones RAG logic from Morphik.

### 3.1. Conceptual Extraction Plan

1.  **Identify Core Data Structures**: We need to understand how Morphik represents document chunks, embeddings, and knowledge graph triples.
2.  **Identify Key Models**: We need to identify the specific models used for multimodal embedding, text embedding, and entity/relationship extraction. The key model for multimodal embedding is `tsystems/colqwen2.5-3b-multilingual-v1.0`.
3.  **Identify Core Algorithms/Pipelines**: We need to understand the logic for intelligent chunking, embedding generation, vector search, knowledge graph construction, and prompt augmentation.

### 3.2. Technical Steps for Extraction and Reuse

#### A. Ingestion Pipeline Extraction

1.  **Document Parsing & Layout Analysis**:
    *   **Goal**: Replicate Morphik's ability to parse documents and extract visual and textual information.
    *   **Location**: `core/parser/morphik_parser.py`
    *   **Key Components**:
        *   `MorphikParser`: The main parser class.
        *   `unstructured` library: Used for parsing various document types.
        *   `VideoParser`: Used for parsing video files.
        *   `StandardChunker` and `ContextualChunker`: Used for chunking text.
    *   **Extraction**:
        *   Create a `DocumentProcessor` class that encapsulates the logic from `MorphikParser`.
        *   This class should be able to handle different file types (videos, PDFs, etc.) and use the appropriate parsing strategy.
        *   For a barebones RAG, you can start with the `StandardChunker`. The `ContextualChunker` is a more advanced feature that you can integrate later.
        *   You will need to install the `unstructured` and `filetype` libraries.

2.  **Multimodal Embedding Generation**:
    *   **Goal**: Generate high-quality embeddings for document chunks.
    *   **Location**: `core/embedding/colpali_embedding_model.py`
    *   **Key Components**:
        *   `ColpaliEmbeddingModel`: The main embedding model class.
        *   `tsystems/colqwen2.5-3b-multilingual-v1.0` model: The multimodal model used for generating embeddings.
    *   **Extraction**:
        *   Create an `EmbeddingGenerator` class that encapsulates the logic from `ColpaliEmbeddingModel`.
        *   This class will need to handle both image and text inputs and generate the corresponding embeddings.
        *   You will need to install the `colpali-engine`, `torch`, `transformers`, and `Pillow` libraries.
        *   You will also need to download the `tsystems/colqwen2.5-3b-multilingual-v1.0` model from the Hugging Face Hub.

#### B. Retrieval Pipeline Extraction

1.  **Vector Store**:
    *   **Goal**: Store and search embeddings efficiently.
    *   **Location**: `core/vector_store/multi_vector_store.py`
    *   **Key Components**:
        *   `MultiVectorStore`: The main vector store class.
        *   `pgvector` extension: Used for storing and querying vectors in PostgreSQL.
        *   `max_sim` function: A custom SQL function for calculating similarity between multi-vector embeddings.
        *   Binary Quantization: A technique for compressing vectors.
    *   **Extraction**:
        *   Set up a PostgreSQL database with the `pgvector` extension.
        *   Create the `multi_vector_embeddings` table and the `max_sim` function in your database. You can find the SQL for this in the `initialize` method of the `MultiVectorStore` class.
        *   Create a `VectorIndex` class that encapsulates the logic from `MultiVectorStore`.
        *   This class will need to handle binary quantization, storing and retrieving embeddings, and querying for similar chunks using the `max_sim` function.
        *   For a barebones RAG, you can start by storing the content of the chunks directly in the database. You can integrate external storage later.

2.  **Knowledge Graph**:
    *   **Goal**: Build and use a knowledge graph to enhance retrieval.
    *   **Location**: `core/services/graph_service.py`
    *   **Key Components**:
        *   `GraphService`: The main service for managing the knowledge graph.
        *   `extract_entities_from_text`: A method for extracting entities and relationships from text using an LLM.
        *   `EntityResolver`: A class for resolving entities.
    *   **Extraction**:
        *   Set up a database to store the graph data.
        *   Implement the `GraphService` class.
        *   Use an LLM to extract entities and relationships from text.
        *   Implement an entity resolution component.
        *   Integrate the `GraphService` with your RAG pipeline.

3.  **Query Processing**:
    *   **Goal**: Convert a user query into an embedding.
    *   **Location**: `core/embedding/colpali_embedding_model.py`
    *   **Extraction**: Reuse the `EmbeddingGenerator` to embed the user query.

#### C. Augmentation and Generation

1.  **Prompt Engineering Strategy**:
    *   **Goal**: Construct an effective prompt for the LLM.
    *   **Extraction**: Create a `PromptBuilder` class that formats the retrieved context.
2.  **LLM Integration**:
    *   **Goal**: Send the augmented prompt to an LLM.
    *   **Extraction**: Use a local or API-based LLM.

This plan provides a comprehensive guide to extracting and reusing the advanced RAG functionalities of Morphik. By following these steps, you can build your own barebones RAG pipeline with features like Visual-first Retrieval, Knowledge Graphs, and Direct Page Embedding.
