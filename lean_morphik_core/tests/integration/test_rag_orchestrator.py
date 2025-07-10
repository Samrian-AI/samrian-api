import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch, mock_open

from lean_morphik_core.app.orchestrator.rag_orchestrator import RAGOrchestrator
from lean_morphik_core.app.models.chunk import Chunk
from lean_morphik_core.app.storage.pinecone_store import DocumentVector as PineconeDocumentVector
from lean_morphik_core.app.storage.neo4j_store import Neo4jNode, Neo4jRelationship
import numpy as np

# Default configuration for the orchestrator in tests
DEFAULT_CONFIG = {
    "pinecone_api_key": "fake_pinecone_key",
    "pinecone_index_name": "test-index",
    "neo4j_uri": "bolt://localhost:7687",
    "neo4j_user": "neo4j",
    "neo4j_password": "password",
}

@pytest.fixture
def mock_parser():
    parser = MagicMock()
    parser.parse_file_to_text = AsyncMock(return_value=({}, "Parsed text content"))
    # For test_ingest_file_pinecone_metadata_stores_original_chunk_content, we use specific content later
    parser.split_text = AsyncMock(return_value=[Chunk(content="Chunk 1", metadata={})])
    return parser

@pytest.fixture
def mock_embedding_model():
    model = MagicMock()
    model.embed_for_ingestion = AsyncMock(return_value=[np.array([0.1, 0.2, 0.3])])
    model.embed_for_query = AsyncMock(return_value=np.array([0.4, 0.5, 0.6]))
    return model

@pytest.fixture
def mock_pinecone_store():
    store = MagicMock()
    store.upsert_vectors = AsyncMock()
    store.query_vectors = AsyncMock(return_value=[
        {"id": "doc1_chunk_0", "score": 0.9, "metadata": {"original_content": "Retrieved chunk 1 content"}} # Assuming original_content is stored
    ])
    return store

@pytest.fixture
def mock_neo4j_store():
    store = MagicMock()
    store.add_node = AsyncMock()
    store.add_relationship = AsyncMock()
    store.close = AsyncMock()
    return store

@pytest.fixture
@patch("lean_morphik_core.app.orchestrator.rag_orchestrator.MorphikParser")
@patch("lean_morphik_core.app.orchestrator.rag_orchestrator.ColpaliEmbeddingModel")
@patch("lean_morphik_core.app.orchestrator.rag_orchestrator.PineconeStore")
@patch("lean_morphik_core.app.orchestrator.rag_orchestrator.Neo4jStore")
def orchestrator(MockNeo4jStore, MockPineconeStore, MockColpaliEmbeddingModel, MockMorphikParser,
                 mock_parser, mock_embedding_model, mock_pinecone_store, mock_neo4j_store):
    MockMorphikParser.return_value = mock_parser
    MockColpaliEmbeddingModel.return_value = mock_embedding_model
    MockPineconeStore.return_value = mock_pinecone_store
    MockNeo4jStore.return_value = mock_neo4j_store
    config = {**DEFAULT_CONFIG, "pinecone_dimension": 3}
    return RAGOrchestrator(**config)


@pytest.mark.asyncio
async def test_ingest_file_successful(orchestrator: RAGOrchestrator, mock_parser, mock_embedding_model, mock_pinecone_store, mock_neo4j_store):
    file_path = "dummy/path/to/file.txt"
    file_name = "file.txt"
    document_id = "doc1"
    pinecone_namespace = "test_ns"

    # Ensure mock_parser.split_text returns a chunk with specific content for this test if needed for other assertions
    # For this specific test, the default "Chunk 1" is fine.

    with patch("builtins.open", mock_open(read_data=b"file content")):
        await orchestrator.ingest_file(file_path, file_name, document_id, pinecone_namespace)

    mocked_file_open.assert_called_once_with(file_path, "rb")
    mock_parser.parse_file_to_text.assert_awaited_once_with(b"file content", file_name)
    mock_parser.split_text.assert_awaited_once_with("Parsed text content")
    mock_embedding_model.embed_for_ingestion.assert_awaited_once_with([Chunk(content="Chunk 1", metadata={})])

    mock_pinecone_store.upsert_vectors.assert_awaited_once()
    args, kwargs = mock_pinecone_store.upsert_vectors.call_args
    upserted_vectors: List[PineconeDocumentVector] = args[0]
    assert len(upserted_vectors) == 1
    assert upserted_vectors[0].id == f"{document_id}_chunk_0"
    assert upserted_vectors[0].values == [0.1, 0.2, 0.3]
    assert upserted_vectors[0].metadata["document_id"] == document_id
    assert upserted_vectors[0].metadata["file_name"] == file_name
    assert upserted_vectors[0].metadata["original_content"] == "Chunk 1" # Due to RAG Orchestrator change
    assert kwargs["namespace"] == pinecone_namespace

    doc_node_call = mock_neo4j_store.add_node.call_args_list[0]
    doc_node_arg: Neo4jNode = doc_node_call[0][0]
    assert doc_node_arg.label == "Document"
    assert doc_node_arg.properties["id"] == document_id

    chunk_node_call = mock_neo4j_store.add_node.call_args_list[1]
    chunk_node_arg: Neo4jNode = chunk_node_call[0][0]
    assert chunk_node_arg.label == "Chunk"
    assert chunk_node_arg.properties["id"] == f"{document_id}_chunk_0"

    rel_call = mock_neo4j_store.add_relationship.call_args_list[0]
    rel_arg: Neo4jRelationship = rel_call[0][0]
    assert rel_arg.type == "HAS_CHUNK"


@pytest.mark.asyncio
async def test_ingest_file_no_text_content(orchestrator: RAGOrchestrator, mock_parser, mock_embedding_model, mock_pinecone_store, mock_neo4j_store):
    mock_parser.parse_file_to_text.return_value = ({}, "  ")
    with patch("builtins.open", mock_open(read_data=b"file content")):
        await orchestrator.ingest_file("file.txt", "file.txt", "doc1")
    mock_parser.split_text.assert_not_awaited()

@pytest.mark.asyncio
async def test_ingest_file_no_chunks(orchestrator: RAGOrchestrator, mock_parser, mock_embedding_model, mock_pinecone_store, mock_neo4j_store):
    mock_parser.split_text.return_value = []
    with patch("builtins.open", mock_open(read_data=b"file content")):
        await orchestrator.ingest_file("file.txt", "file.txt", "doc1")
    mock_embedding_model.embed_for_ingestion.assert_not_awaited()

@pytest.mark.asyncio
async def test_ingest_file_io_error(orchestrator: RAGOrchestrator, mock_parser):
    with patch("builtins.open", mock_open()) as mocked_file:
        mocked_file.side_effect = IOError("File not found")
        with pytest.raises(IOError):
            await orchestrator.ingest_file("nonexistent.txt", "nonexistent.txt", "doc1")
    mock_parser.parse_file_to_text.assert_not_awaited()

@pytest.mark.asyncio
@patch("litellm.acompletion") # Corrected patch path
async def test_query_successful(mock_litellm_acompletion, orchestrator: RAGOrchestrator, mock_embedding_model, mock_pinecone_store):
    query_text = "Tell me about apples"
    mock_llm_response = MagicMock()
    mock_llm_response.choices = [MagicMock(message=MagicMock(content="Apples are fruits."))]
    mock_litellm_acompletion.return_value = mock_llm_response

    result = await orchestrator.query(query_text, synthesis_llm_model="test-llm")

    mock_embedding_model.embed_for_query.assert_awaited_once_with(query_text)
    mock_pinecone_store.query_vectors.assert_awaited_once_with(
        query_embedding=[0.4, 0.5, 0.6], top_k=5, namespace=None, filter_metadata=None
    )
    mock_litellm_acompletion.assert_awaited_once()
    call_args = mock_litellm_acompletion.call_args[1]
    assert call_args["model"] == "test-llm"
    messages = call_args["messages"]
    # Updated to check for 'original_content' based on mock_pinecone_store and RAG orchestrator logic
    assert "Retrieved chunk 1 content" in messages[1]["content"]
    assert query_text in messages[1]["content"]
    assert result["answer"] == "Apples are fruits."
    assert result["retrieved_pinecone_matches"][0]["metadata"]["original_content"] == "Retrieved chunk 1 content"

@pytest.mark.asyncio
@patch("litellm.acompletion") # Corrected patch path
async def test_query_no_pinecone_results(mock_litellm_acompletion, orchestrator: RAGOrchestrator, mock_pinecone_store):
    mock_pinecone_store.query_vectors.return_value = []
    mock_llm_response = MagicMock()
    mock_llm_response.choices = [MagicMock(message=MagicMock(content="I don't have enough information."))]
    mock_litellm_acompletion.return_value = mock_llm_response

    result = await orchestrator.query("query with no results")
    mock_litellm_acompletion.assert_awaited_once()
    messages = mock_litellm_acompletion.call_args[1]["messages"]
    assert "Context:\n\n\nQuery:" in messages[1]["content"]
    assert result["answer"] == "I don't have enough information."

@pytest.mark.asyncio
@patch("litellm.acompletion") # Corrected patch path
async def test_query_llm_error(mock_litellm_acompletion, orchestrator: RAGOrchestrator):
    mock_litellm_acompletion.side_effect = Exception("LLM API error")
    result = await orchestrator.query("test query")
    assert result["answer"] == "There was an error generating the answer."

@pytest.mark.asyncio
async def test_close_stores(orchestrator: RAGOrchestrator, mock_neo4j_store):
    await orchestrator.close_stores()
    mock_neo4j_store.close.assert_awaited_once()

@pytest.mark.asyncio
async def test_ingest_file_pinecone_metadata_stores_original_chunk_content(orchestrator: RAGOrchestrator, mock_parser, mock_pinecone_store):
    specific_chunk_content = "This is the specific full chunk content for this test."
    mock_parser.split_text.return_value = [Chunk(content=specific_chunk_content, metadata={"source": "test_source"})]
    mock_parser.parse_file_to_text.return_value = ({"parser_meta": "value"}, "Parsed text")

    with patch("builtins.open", mock_open(read_data=b"data")):
        await orchestrator.ingest_file("file.txt", "file.txt", "doc123")

    mock_pinecone_store.upsert_vectors.assert_awaited_once()
    args, _ = mock_pinecone_store.upsert_vectors.call_args
    upserted_vector_metadata = args[0][0].metadata

    assert upserted_vector_metadata["document_id"] == "doc123"
    assert upserted_vector_metadata["parser_meta"] == "value"
    assert upserted_vector_metadata["source"] == "test_source"
    # Updated assertion: RAGOrchestrator now stores 'original_content'
    assert "original_content" in upserted_vector_metadata
    assert upserted_vector_metadata["original_content"] == specific_chunk_content

@pytest.mark.asyncio
@patch("litellm.acompletion") # Corrected patch path
async def test_query_retrieves_text_from_pinecone_metadata_if_present(mock_litellm_acompletion, orchestrator: RAGOrchestrator, mock_pinecone_store):
    mock_pinecone_store.query_vectors.return_value = [
        {"id": "match1", "score": 0.9, "metadata": {"original_content": "Original content text 1"}}, # Changed from text_preview
        {"id": "match2", "score": 0.8, "metadata": {"original_content": "Original content text 2"}},
        {"id": "match3", "score": 0.7, "metadata": {"other_key": "some_value"}},
    ]
    mock_llm_response = MagicMock()
    mock_llm_response.choices = [MagicMock(message=MagicMock(content="LLM answer"))]
    mock_litellm_acompletion.return_value = mock_llm_response

    await orchestrator.query("A query")

    mock_litellm_acompletion.assert_awaited_once()
    final_prompt_for_llm = mock_litellm_acompletion.call_args[1]["messages"][1]["content"]

    assert "Original content text 1" in final_prompt_for_llm # Check for original_content
    assert "Original content text 2" in final_prompt_for_llm
    expected_context_str = "Original content text 1\n\n---\n\nOriginal content text 2"
    assert expected_context_str in final_prompt_for_llm
    # caplog = LogCaptureFixture; import logging; caplog.set_level(logging.WARNING)
    # assert "Could not find text content for Pinecone match ID match3" in caplog.text
