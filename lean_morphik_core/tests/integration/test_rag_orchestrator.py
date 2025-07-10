import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch, mock_open

from lean_morphik_core.app.orchestrator.rag_orchestrator import RAGOrchestrator
from lean_morphik_core.app.models.chunk import Chunk
# Import base store types for mocking, and generic DocumentVector, Node, Relationship
from lean_morphik_core.app.storage.base_vector_store import BaseVectorStore, DocumentVector
from lean_morphik_core.app.storage.base_graph_store import BaseGraphStore, Node, Relationship
# Specific store model imports are no longer needed here if we use generic types for assertions
# from lean_morphik_core.app.storage.pinecone_store import DocumentVector as PineconeDocumentVector
# from lean_morphik_core.app.storage.neo4j_store import Neo4jNode, Neo4jRelationship
import numpy as np

# DEFAULT_CONFIG is no longer needed as orchestrator takes instances directly.

@pytest.fixture
def mock_parser():
    # This will be an instance of MorphikParser, mocked
    parser = AsyncMock(spec=MagicMock) # Use AsyncMock for async methods, spec for attribute access if needed for type hints
    parser.parse_file_to_text = AsyncMock(return_value=({}, "Parsed text content"))
    parser.split_text = AsyncMock(return_value=[Chunk(content="Chunk 1", metadata={})])
    return parser

@pytest.fixture
def mock_embedding_model():
    # This will be an instance of ColpaliEmbeddingModel, mocked
    model = AsyncMock(spec=MagicMock)
    model.embed_for_ingestion = AsyncMock(return_value=[np.array([0.1, 0.2, 0.3])]) # Ensure return is a list of arrays
    model.embed_for_query = AsyncMock(return_value=np.array([0.4, 0.5, 0.6]))
    return model

@pytest.fixture
def mock_vector_store():
    # This will be an instance of BaseVectorStore (e.g., a mocked PineconeStore)
    store = AsyncMock(spec=BaseVectorStore)
    store.upsert_vectors = AsyncMock()
    store.query_vectors = AsyncMock(return_value=[
        {"id": "doc1_chunk_0", "score": 0.9, "metadata": {"original_content": "Retrieved chunk 1 content"}}
    ])
    return store

@pytest.fixture
def mock_graph_store():
    # This will be an instance of BaseGraphStore (e.g., a mocked Neo4jStore)
    store = AsyncMock(spec=BaseGraphStore)
    store.add_node = AsyncMock()
    store.add_relationship = AsyncMock()
    store.close = AsyncMock() # Ensure close is part of the spec if BaseGraphStore defines it
    return store

@pytest.fixture
def orchestrator(mock_parser, mock_embedding_model, mock_vector_store, mock_graph_store):
    # Patches on RAGOrchestrator's dependencies are no longer needed here,
    # as we are injecting mocked instances directly.
    return RAGOrchestrator(
        parser=mock_parser,
        embedding_model=mock_embedding_model,
        vector_store=mock_vector_store,
        graph_store=mock_graph_store
    )

@pytest.mark.asyncio
async def test_ingest_file_successful(orchestrator: RAGOrchestrator, mock_parser, mock_embedding_model, mock_vector_store, mock_graph_store):
    file_path = "dummy/path/to/file.txt"
    file_name = "file.txt"
    document_id = "doc1"
    vector_store_namespace = "test_ns" # Updated variable name

    mock_parser.split_text.return_value = [Chunk(content="Chunk 1", metadata={})] # Ensure return value is set for this test

    mock_builtin_open_instance = mock_open(read_data=b"file content")
    with patch("builtins.open", mock_builtin_open_instance):
        await orchestrator.ingest_file(file_path, file_name, document_id, vector_store_namespace)

    mock_builtin_open_instance.assert_called_once_with(file_path, "rb")
    mock_parser.parse_file_to_text.assert_awaited_once_with(b"file content", file_name)
    mock_parser.split_text.assert_awaited_once_with("Parsed text content")
    # Ensure the argument to embed_for_ingestion is a list of Chunks
    mock_embedding_model.embed_for_ingestion.assert_awaited_once_with([Chunk(content="Chunk 1", metadata={})])


    mock_vector_store.upsert_vectors.assert_awaited_once()
    args, kwargs = mock_vector_store.upsert_vectors.call_args
    upserted_vectors: List[DocumentVector] = args[0] # Uses generic DocumentVector
    assert len(upserted_vectors) == 1
    assert upserted_vectors[0].id == f"{document_id}_chunk_0"
    assert upserted_vectors[0].values == [0.1, 0.2, 0.3] # np.array was converted to list by orchestrator
    assert upserted_vectors[0].metadata["document_id"] == document_id
    assert upserted_vectors[0].metadata["file_name"] == file_name
    assert upserted_vectors[0].metadata["original_content"] == "Chunk 1"
    assert kwargs["namespace"] == vector_store_namespace

    # Assert calls to graph_store using generic Node/Relationship types
    # First call to add_node should be the Document node
    doc_node_call = mock_graph_store.add_node.call_args_list[0]
    doc_node_arg: Node = doc_node_call[0][0] # Argument is a generic Node
    assert doc_node_arg.label == "Document"
    assert doc_node_arg.properties["id"] == document_id

    # Second call to add_node should be the Chunk node
    chunk_node_call = mock_graph_store.add_node.call_args_list[1]
    chunk_node_arg: Node = chunk_node_call[0][0]
    assert chunk_node_arg.label == "Chunk"
    assert chunk_node_arg.properties["id"] == f"{document_id}_chunk_0"

    # Call to add_relationship
    rel_call = mock_graph_store.add_relationship.call_args_list[0]
    rel_arg: Relationship = rel_call[0][0] # Argument is a generic Relationship
    assert rel_arg.type == "HAS_CHUNK"
    assert rel_arg.source_node_properties["id"] == document_id
    assert rel_arg.target_node_properties["id"] == f"{document_id}_chunk_0"


@pytest.mark.asyncio
async def test_ingest_file_no_text_content(orchestrator: RAGOrchestrator, mock_parser): # Removed unused store mocks
    mock_parser.parse_file_to_text.return_value = ({}, "  ") # Simulate empty text extracted
    with patch("builtins.open", mock_open(read_data=b"file content")):
        await orchestrator.ingest_file("file.txt", "file.txt", "doc1")
    mock_parser.split_text.assert_not_awaited() # Should not proceed to splitting

@pytest.mark.asyncio
async def test_ingest_file_no_chunks(orchestrator: RAGOrchestrator, mock_parser, mock_embedding_model): # Removed unused store mocks
    mock_parser.split_text.return_value = [] # Simulate no chunks produced
    with patch("builtins.open", mock_open(read_data=b"file content")):
        await orchestrator.ingest_file("file.txt", "file.txt", "doc1")
    mock_embedding_model.embed_for_ingestion.assert_not_awaited() # Should not proceed to embedding

@pytest.mark.asyncio
async def test_ingest_file_io_error(orchestrator: RAGOrchestrator, mock_parser): # Removed unused store mocks
    with patch("builtins.open", mock_open()) as mocked_file:
        mocked_file.side_effect = IOError("File not found")
        with pytest.raises(IOError):
            await orchestrator.ingest_file("nonexistent.txt", "nonexistent.txt", "doc1")
    mock_parser.parse_file_to_text.assert_not_awaited()

@pytest.mark.asyncio
@patch("lean_morphik_core.app.orchestrator.rag_orchestrator.litellm.acompletion")
async def test_query_successful(mock_litellm_acompletion, orchestrator: RAGOrchestrator, mock_embedding_model, mock_vector_store):
    query_text = "Tell me about apples"
    mock_llm_response = MagicMock()
    mock_llm_response.choices = [MagicMock(message=MagicMock(content="Apples are fruits."))]
    mock_litellm_acompletion.return_value = mock_llm_response

    # Use the updated parameter names for query method
    result = await orchestrator.query(
        query_text,
        vector_store_top_k=5,
        vector_store_namespace=None,
        vector_store_filter_metadata=None,
        synthesis_llm_model="test-llm"
    )

    mock_embedding_model.embed_for_query.assert_awaited_once_with(query_text)
    mock_vector_store.query_vectors.assert_awaited_once_with(
        query_embedding=[0.4, 0.5, 0.6], # np.array converted to list by orchestrator
        top_k=5,
        namespace=None,
        filter_metadata=None
    )
    mock_litellm_acompletion.assert_awaited_once()
    call_args = mock_litellm_acompletion.call_args[1]
    assert call_args["model"] == "test-llm"
    messages = call_args["messages"]
    assert "Retrieved chunk 1 content" in messages[1]["content"] # From mock_vector_store setup
    assert query_text in messages[1]["content"]
    assert result["answer"] == "Apples are fruits."
    # Key for retrieved matches also updated
    assert result["retrieved_vector_store_matches"][0]["metadata"]["original_content"] == "Retrieved chunk 1 content"

@pytest.mark.asyncio
@patch("lean_morphik_core.app.orchestrator.rag_orchestrator.litellm.acompletion")
async def test_query_no_vector_store_results(mock_litellm_acompletion, orchestrator: RAGOrchestrator, mock_vector_store): # Renamed mock
    mock_vector_store.query_vectors.return_value = [] # Simulate no results
    mock_llm_response = MagicMock()
    mock_llm_response.choices = [MagicMock(message=MagicMock(content="I don't have enough information."))]
    mock_litellm_acompletion.return_value = mock_llm_response

    result = await orchestrator.query("query with no results")

    mock_litellm_acompletion.assert_awaited_once()
    messages = mock_litellm_acompletion.call_args[1]["messages"]
    # Ensure context is empty or appropriately formatted when no results
    assert "Context:\n\n\nQuery:" in messages[1]["content"] or "Context:\n\nQuery:" in messages[1]["content"]
    assert result["answer"] == "I don't have enough information."


@pytest.mark.asyncio
@patch("lean_morphik_core.app.orchestrator.rag_orchestrator.litellm.acompletion")
async def test_query_llm_error(mock_litellm_acompletion, orchestrator: RAGOrchestrator): # Removed unused store mocks
    mock_litellm_acompletion.side_effect = Exception("LLM API error")
    result = await orchestrator.query("test query")
    mock_litellm_acompletion.assert_awaited_once() # Ensure it was actually called
    assert result["answer"] == "There was an error generating the answer."

@pytest.mark.asyncio
async def test_close_stores(orchestrator: RAGOrchestrator, mock_graph_store, mock_vector_store):
    # Mock vector_store to also have a close method for testing this scenario
    mock_vector_store.close = AsyncMock()

    await orchestrator.close_stores()

    mock_graph_store.close.assert_awaited_once()
    mock_vector_store.close.assert_awaited_once() # Assert if BaseVectorStore had close()

@pytest.mark.asyncio
async def test_ingest_file_vector_store_metadata_stores_original_chunk_content(orchestrator: RAGOrchestrator, mock_parser, mock_vector_store): # Renamed
    specific_chunk_content = "This is the specific full chunk content for this test."
    mock_parser.split_text.return_value = [Chunk(content=specific_chunk_content, metadata={"source": "test_source"})]
    mock_parser.parse_file_to_text.return_value = ({"parser_meta": "value"}, "Parsed text")

    with patch("builtins.open", mock_open(read_data=b"data")):
        await orchestrator.ingest_file("file.txt", "file.txt", "doc123")

    mock_vector_store.upsert_vectors.assert_awaited_once()
    args, _ = mock_vector_store.upsert_vectors.call_args
    upserted_vector: DocumentVector = args[0][0] # generic DocumentVector

    assert upserted_vector.metadata["document_id"] == "doc123"
    assert upserted_vector.metadata["parser_meta"] == "value"
    assert upserted_vector.metadata["source"] == "test_source"
    assert "original_content" in upserted_vector.metadata
    assert upserted_vector.metadata["original_content"] == specific_chunk_content

@pytest.mark.asyncio
@patch("lean_morphik_core.app.orchestrator.rag_orchestrator.litellm.acompletion")
async def test_query_retrieves_text_from_vector_store_metadata_if_present(mock_litellm_acompletion, orchestrator: RAGOrchestrator, mock_vector_store): # Renamed
    mock_vector_store.query_vectors.return_value = [
        {"id": "match1", "score": 0.9, "metadata": {"original_content": "Original content text 1"}},
        {"id": "match2", "score": 0.8, "metadata": {"original_content": "Original content text 2"}},
        {"id": "match3", "score": 0.7, "metadata": {"other_key": "some_value"}}, # No original_content or text_preview
    ]
    mock_llm_response = MagicMock()
    mock_llm_response.choices = [MagicMock(message=MagicMock(content="LLM answer"))]
    mock_litellm_acompletion.return_value = mock_llm_response

    await orchestrator.query("A query")

    mock_litellm_acompletion.assert_awaited_once()
    final_prompt_for_llm = mock_litellm_acompletion.call_args[1]["messages"][1]["content"]

    assert "Original content text 1" in final_prompt_for_llm
    assert "Original content text 2" in final_prompt_for_llm
    expected_context_str = "Original content text 1\n\n---\n\nOriginal content text 2" # Only two items have content
    assert expected_context_str in final_prompt_for_llm
    # To check for the warning log:
    # import logging
    # with caplog.at_level(logging.WARNING):
    #    await orchestrator.query("A query") # Call again or ensure previous call is captured
    # assert "Could not find text content for VectorStore match ID match3" in caplog.text
