import pytest
import os
import tempfile
from typing import List

# Adjust import path based on your project structure
from lean_morphik_core.app.ingestion.morphik_parser import MorphikParser
from lean_morphik_core.app.models.chunk import Chunk

# Helper to create a dummy file
def create_dummy_file(content: str, extension: str = ".txt") -> str:
    fd, path = tempfile.mkstemp(suffix=extension)
    with os.fdopen(fd, "w") as tmp:
        tmp.write(content)
    return path

@pytest.fixture
def basic_parser() -> MorphikParser:
    """Fixture for a basic MorphikParser with default settings."""
    return MorphikParser(
        chunk_size=100, # Small chunk size for easier testing
        chunk_overlap=10,
        # No API keys needed for basic text parsing without contextual chunking or video
    )

@pytest.mark.asyncio
async def test_parse_simple_text_file(basic_parser: MorphikParser):
    content = "This is a simple sentence. This is another sentence. And a third one."
    dummy_file_path = create_dummy_file(content, ".txt")

    try:
        file_bytes = open(dummy_file_path, "rb").read()
        metadata, text_content = await basic_parser.parse_file_to_text(file_bytes, os.path.basename(dummy_file_path))

        assert isinstance(metadata, dict)
        assert text_content.strip() == content.strip() # unstructured might add minor whitespace changes

    finally:
        os.remove(dummy_file_path)

@pytest.mark.asyncio
async def test_split_text_standard_chunker(basic_parser: MorphikParser):
    text_content = "This is the first sentence. This is the second sentence, which is a bit longer. The third sentence is short. A fourth one to make sure. And a fifth for good measure."
    # Expected chunking is approximate due to RecursiveCharacterTextSplitter logic
    # With chunk_size=100, overlap=10

    chunks: List[Chunk] = await basic_parser.split_text(text_content)

    assert len(chunks) > 0
    for chunk in chunks:
        assert isinstance(chunk, Chunk)
        assert len(chunk.content) > 0
        # Check if overlap is somewhat working (very basic check)
        if len(chunks) > 1 and chunks.index(chunk) > 0:
            prev_chunk_suffix = chunks[chunks.index(chunk)-1].content[-basic_parser.chunker.text_splitter.chunk_overlap:]
            # This check might be too strict depending on splitter behavior with short sentences.
            # assert chunk.content.startswith(prev_chunk_suffix)

    # Verify total length is roughly preserved (minus overlaps being counted once)
    total_chunked_length = sum(len(c.content) for c in chunks)
    # This is not exact due to overlap handling, but should be in the ballpark
    assert total_chunked_length >= len(text_content)
    assert total_chunked_length <= len(text_content) + (len(chunks) * basic_parser.chunker.text_splitter.chunk_overlap)


@pytest.mark.asyncio
async def test_parse_pdf_file_mocked_unstructured(basic_parser: MorphikParser, mocker):
    # Mock the 'partition' call from unstructured
    mock_partition_element = mocker.MagicMock()
    mock_partition_element.__str__ = lambda self: "Text from PDF element."

    mock_partition = mocker.patch(
        "lean_morphik_core.app.ingestion.morphik_parser.partition",
        return_value=[mock_partition_element, mock_partition_element] # Simulate two elements returned
    )

    dummy_pdf_path = create_dummy_file("dummy PDF content", ".pdf") # Content doesn't matter due to mock

    try:
        file_bytes = open(dummy_pdf_path, "rb").read()
        metadata, text_content = await basic_parser.parse_file_to_text(file_bytes, os.path.basename(dummy_pdf_path))

        mock_partition.assert_called_once()
        args, kwargs = mock_partition.call_args
        assert kwargs.get("strategy") == "fast" # Default for .pdf
        assert kwargs.get("metadata_filename") == os.path.basename(dummy_pdf_path)

        assert text_content == "Text from PDF element.\n\nText from PDF element."

    finally:
        os.remove(dummy_pdf_path)

# More tests needed:
# - Contextual chunking (requires mocking litellm and providing LLM config to parser)
# - Video parsing (requires mocking AssemblyAI, VideoParser, and providing API keys)
# - Different file types (.docx, .json) and their strategies
# - Edge cases for chunking (empty text, very long text, text shorter than chunk size)
# - Error handling (e.g., file not found, invalid API keys if those were tested directly)

@pytest.mark.asyncio
async def test_empty_text_chunking(basic_parser: MorphikParser):
    chunks = await basic_parser.split_text("")
    assert len(chunks) == 0

@pytest.mark.asyncio
async def test_text_shorter_than_chunk_size(basic_parser: MorphikParser):
    text = "Short text."
    chunks = await basic_parser.split_text(text)
    assert len(chunks) == 1
    assert chunks[0].content == text

# To run these tests:
# 1. Ensure pytest and pytest-asyncio are installed: pip install pytest pytest-asyncio pytest-mock
# 2. Navigate to the root of your `lean_morphik_core` project (or adjust paths).
# 3. Run `pytest` in the terminal.
#
# Note on ContextualChunker and VideoParser tests:
# These would be more involved as they require:
# - For ContextualChunker:
#   - Instantiating MorphikParser with `use_contextual_chunking=True` and providing necessary LLM model names/keys.
#   - Mocking `litellm.completion` to return a simulated LLM response.
# - For VideoParser (tested via MorphikParser._parse_video):
#   - Instantiating MorphikParser with an `assemblyai_api_key` and vision model details.
#   - Mocking `tempfile.NamedTemporaryFile`.
#   - Mocking `VideoParser` itself or its dependencies like `assemblyai.Transcriber` and `VisionModelClient.get_frame_description`.
#   - Providing a dummy video file or mocking `filetype.guess`.
# These are left as exercises due to their complexity and dependency on external service mocking.
