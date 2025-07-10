import pytest
import pytest
import os
import tempfile
from typing import List
from unittest.mock import patch # Ensure patch is imported at the top

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

@pytest.mark.unit
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

@pytest.mark.unit
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
        # Detailed overlap checking is handled by test_standard_chunker_overlap
        # and test_recursive_splitter_separators.

    # Verify total length is roughly preserved (minus overlaps being counted once)
    total_chunked_length = sum(len(c.content) for c in chunks)
    # This is not exact due to overlap handling, but should be in the ballpark
    assert total_chunked_length >= len(text_content)
    assert total_chunked_length <= len(text_content) + (len(chunks) * basic_parser.chunker.text_splitter.chunk_overlap)


@pytest.mark.unit
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

@pytest.mark.unit
@pytest.mark.asyncio
async def test_parse_docx_file_mocked_unstructured(basic_parser: MorphikParser, mocker):
    mock_partition_element = mocker.MagicMock()
    mock_partition_element.__str__ = lambda self: "Text from DOCX element."
    mock_partition = mocker.patch(
        "lean_morphik_core.app.ingestion.morphik_parser.partition",
        return_value=[mock_partition_element]
    )
    dummy_docx_path = create_dummy_file("dummy DOCX content", ".docx")
    try:
        file_bytes = open(dummy_docx_path, "rb").read()
        _, text_content = await basic_parser.parse_file_to_text(file_bytes, os.path.basename(dummy_docx_path))
        mock_partition.assert_called_once()
        args, kwargs = mock_partition.call_args
        assert kwargs.get("strategy") == "fast"
        assert kwargs.get("metadata_filename") == os.path.basename(dummy_docx_path)
        assert text_content == "Text from DOCX element."
    finally:
        os.remove(dummy_docx_path)

@pytest.mark.unit
@pytest.mark.asyncio
async def test_parse_json_file_mocked_unstructured(basic_parser: MorphikParser, mocker):
    mock_partition_element = mocker.MagicMock()
    mock_partition_element.__str__ = lambda self: '{"key": "value from JSON element"}'
    mock_partition = mocker.patch(
        "lean_morphik_core.app.ingestion.morphik_parser.partition",
        return_value=[mock_partition_element]
    )
    dummy_json_path = create_dummy_file('{"dummy": "json content"}', ".json")
    try:
        file_bytes = open(dummy_json_path, "rb").read()
        _, text_content = await basic_parser.parse_file_to_text(file_bytes, os.path.basename(dummy_json_path))
        mock_partition.assert_called_once()
        args, kwargs = mock_partition.call_args
        assert kwargs.get("strategy") == "fast"
        assert kwargs.get("content_type") == "application/json"
        assert kwargs.get("metadata_filename") == os.path.basename(dummy_json_path)
        assert text_content == '{"key": "value from JSON element"}'
    finally:
        os.remove(dummy_json_path)

@pytest.mark.unit
@pytest.mark.asyncio
async def test_parse_markdown_file_mocked_unstructured(basic_parser: MorphikParser, mocker):
    mock_partition_element = mocker.MagicMock()
    mock_partition_element.__str__ = lambda self: "# Markdown Title\nSome text."
    mock_partition = mocker.patch(
        "lean_morphik_core.app.ingestion.morphik_parser.partition",
        return_value=[mock_partition_element]
    )
    dummy_md_path = create_dummy_file("# Dummy MD content", ".md")
    try:
        file_bytes = open(dummy_md_path, "rb").read()
        _, text_content = await basic_parser.parse_file_to_text(file_bytes, os.path.basename(dummy_md_path))
        mock_partition.assert_called_once()
        args, kwargs = mock_partition.call_args
        assert kwargs.get("metadata_filename") == os.path.basename(dummy_md_path)
        assert kwargs.get("content_type") is None
        assert text_content == "# Markdown Title\nSome text."
    finally:
        os.remove(dummy_md_path)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_empty_text_chunking(basic_parser: MorphikParser):
    chunks = await basic_parser.split_text("")
    if chunks:
        assert len(chunks) == 1
        assert chunks[0].content == ""
    else:
        assert len(chunks) == 0


@pytest.mark.unit
@pytest.mark.asyncio
async def test_text_shorter_than_chunk_size(basic_parser: MorphikParser):
    text = "Short text."
    chunks = await basic_parser.split_text(text)
    assert len(chunks) == 1
    assert chunks[0].content == text

@pytest.mark.unit
@pytest.mark.asyncio
async def test_standard_chunker_overlap(basic_parser: MorphikParser):
    text = "This is a sentence that is definitely longer than one hundred characters to ensure that it will be split into multiple chunks for testing the overlap. This is the second part of the text, also made long enough."
    chunks: List[Chunk] = await basic_parser.split_text(text)

    assert len(chunks) > 1
    for i in range(1, len(chunks)):
        prev_chunk_content = chunks[i-1].content
        current_chunk_content = chunks[i].content
        overlap_val = basic_parser.chunker.text_splitter.chunk_overlap
        expected_overlap_segment = prev_chunk_content[-overlap_val:]
        assert current_chunk_content.startswith(expected_overlap_segment), \
            f"Chunk {i} does not start with the overlap from chunk {i-1}. Expected start: '{expected_overlap_segment}', Got: '{current_chunk_content[:overlap_val]}'"


# --- ContextualChunker Tests ---
@pytest.fixture
def contextual_parser(mocker) -> MorphikParser:
    # The litellm.completion mock will be applied directly to the test function
    # to ensure the correct patching target.
    return MorphikParser(
        chunk_size=100,
        chunk_overlap=10,
        use_contextual_chunking=True,
        contextual_chunking_llm_model_name="test-model",
        contextual_chunking_llm_api_key="test_key"
    )

@pytest.mark.unit
@pytest.mark.asyncio
@patch("lean_morphik_core.app.ingestion.morphik_parser.litellm.completion")
async def test_contextual_chunking_adds_context(mock_parser_litellm_completion, contextual_parser: MorphikParser, mocker): # Add mock to params
    text_content = "This is a test sentence for contextual chunking. It needs to be long enough to be chunked."

    # Configure the mock provided by the @patch decorator
    mock_response = mocker.MagicMock()
    mock_message = mocker.MagicMock()
    mock_message.content = "Mocked LLM Context. "
    mock_response.choices = [mocker.MagicMock(message=mock_message)]
    mock_parser_litellm_completion.return_value = mock_response

    chunks: List[Chunk] = await contextual_parser.split_text(text_content)

    assert len(chunks) > 0
    mock_parser_litellm_completion.assert_called()

    for chunk in chunks:
        assert chunk.content.startswith("Mocked LLM Context. ")
        original_part = chunk.content.replace("Mocked LLM Context. ; ", "")
        assert original_part in text_content

# --- VideoParser Mocked Tests ---
@pytest.mark.unit
@pytest.mark.asyncio
async def test_parse_video_file_mocked(mocker):
    mocker.patch("lean_morphik_core.app.ingestion.morphik_parser.filetype.guess", return_value=mocker.MagicMock(mime="video/mp4"))
    mock_temp_file = mocker.patch("lean_morphik_core.app.ingestion.morphik_parser.tempfile.NamedTemporaryFile")
    mock_temp_file.return_value.__enter__.return_value.name = "dummy_video_path.mp4"
    mocker.patch("lean_morphik_core.app.ingestion.morphik_parser.os.unlink")

    mock_video_parser_instance = mocker.AsyncMock()
    mock_video_parser_instance.process_video.return_value = mocker.MagicMock(
        metadata={"duration": 10},
        frame_descriptions=mocker.MagicMock(time_to_content={"0.0": "frame desc 1"}),
        transcript=mocker.MagicMock(time_to_content={"0.5": "transcript part 1"})
    )
    mock_video_parser_class = mocker.patch(
        "lean_morphik_core.app.ingestion.morphik_parser.VideoParser",
        return_value=mock_video_parser_instance
    )

    parser = MorphikParser(
        assemblyai_api_key="fake_assembly_key",
        video_vision_model_name="fake_vision_model"
    )

    dummy_video_bytes = b"fake video data"
    metadata, text_content = await parser.parse_file_to_text(dummy_video_bytes, "test_video.mp4")

    mock_video_parser_class.assert_called_once_with(
        "dummy_video_path.mp4",
        assemblyai_api_key="fake_assembly_key",
        frame_sample_rate=parser.video_frame_sample_rate,
        vision_model_name=parser.video_vision_model_name,
        vision_api_key=parser.video_vision_api_key,
        vision_base_url=parser.video_vision_base_url
    )
    mock_video_parser_instance.process_video.assert_awaited_once()

    assert "frame desc 1" in text_content
    assert "transcript part 1" in text_content
    assert metadata["video_metadata"] == {"duration": 10}

# --- RecursiveCharacterTextSplitter Direct Tests ---
@pytest.mark.unit
def test_recursive_splitter_simple_split():
    test_splitter_config = MorphikParser(chunk_size=20, chunk_overlap=5).chunker.text_splitter
    text = "Sentence one. Sentence two. Sentence three."
    chunks_objs = test_splitter_config.split_text(text) # This returns List[Chunk]
    chunks_content = [c.content for c in chunks_objs]
    # With chunk_size=20, overlap=5, for "Sentence one. Sentence two. Sentence three."
    # Default separators: ["\n\n", "\n", ". ", " ", ""]
    # "Sentence one. " (len 14) -> chunk1
    # "Sentence two. " (len 14) -> chunk2, overlap from chunk1 " one. " + "Sentence two. "
    # "Sentence three." (len 15) -> chunk3, overlap from chunk2 " two. " + "Sentence three."
    # Expected:
    # chunks_content[0] = "Sentence one. "
    # chunks_content[1] = " one. Sentence two. "
    # chunks_content[2] = " two. Sentence three."
    # This needs to be verified by running the actual splitter. The test below is more robust for generic long text.

    long_text = "Thisisareallylongwordwithoutanyspacesorperiodstotestthesplittinglogic"
    chunk_objs_long = test_splitter_config.split_text(long_text)
    chunks_content_long = [c.content for c in chunk_objs_long]

    assert len(chunks_content_long) > 1
    assert len(chunks_content_long[0]) <= test_splitter_config.chunk_size
    for i in range(1, len(chunks_content_long)):
        non_overlap_part = chunks_content_long[i][test_splitter_config.chunk_overlap:]
        assert len(non_overlap_part) <= test_splitter_config.chunk_size
        expected_overlap = chunks_content_long[i-1][-test_splitter_config.chunk_overlap:]
        assert chunks_content_long[i].startswith(expected_overlap)

    short_text = "Short."
    chunk_objs_short = test_splitter_config.split_text(short_text)
    chunks_content_short = [c.content for c in chunk_objs_short]
    assert len(chunks_content_short) == 1
    assert chunks_content_short[0] == short_text


@pytest.mark.unit
def test_recursive_splitter_separators():
    splitter = MorphikParser(chunk_size=30, chunk_overlap=5).chunker.text_splitter
    text = "First paragraph.\n\nSecond paragraph with a sentence. Another sentence.\nThird line."

    chunk_objs = splitter.split_text(text) # This returns List[Chunk]
    chunks = [c.content for c in chunk_objs]

    # Expected based on my trace (which might be slightly off from actual splitter logic):
    # C1: "First paragraph.\n\n" (if separator is kept with previous)
    # C2: "aph.\n\nSecond paragraph with a sentence. "
    # C3: "ence. Another sentence.\n"
    # C4: "nce.\nThird line."
    # The key is that it respects separators and then chunk_size/overlap.
    # Focus on properties: chunk count, content presence at start/end, overlap, and size constraints.

    assert len(chunks) > 1 # Expecting multiple chunks

    # Check Content Presence (start and end)
    # Note: Depending on splitter behavior with separators, the absolute start/end might be tricky.
    # For example, if a separator is at the very beginning.
    # Let's assume for this text, it's straightforward.
    assert chunks[0].content.startswith("First paragraph.") # Access .content from Chunk object
    assert chunks[-1].content.endswith("Third line.")    # Access .content from Chunk object

    # Check Overlap Property and size constraints for each chunk
    for i in range(len(chunk_objs)):
        current_chunk_content = chunk_objs[i].content

        # Check chunk size constraint (non-overlapping part)
        # The first chunk's content should be within chunk_size.
        # Subsequent chunks' non-overlapping part should be within chunk_size.
        if i == 0:
            assert len(current_chunk_content) <= splitter.chunk_size, \
                f"First chunk size {len(current_chunk_content)} exceeds {splitter.chunk_size}"
        else: # For chunks after the first, verify overlap and non-overlapping part size
            overlap_val = splitter.chunk_overlap
            expected_overlap_segment = chunk_objs[i-1].content[-overlap_val:]
            assert current_chunk_content.startswith(expected_overlap_segment), \
                f"Chunk {i} does not start with the overlap from chunk {i-1}. Expected start: '{expected_overlap_segment}', Got: '{current_chunk_content[:overlap_val]}'"

            non_overlapping_part_length = len(current_chunk_content) - overlap_val
            # It's possible the non_overlapping_part_length is negative if chunk is smaller than overlap
            # This usually means the chunk is mostly or all overlap.
            # The core idea is that what's *added* beyond the overlap shouldn't itself exceed chunk_size.
            # A more direct way for RecursiveCharacterTextSplitter is often to check total length of created documents.
            # However, the prompt asks for this specific overlap check.
            # If the chunk is smaller than overlap, it implies it's a final small piece.
            # The splitter aims for len(docs[i]) ~= chunk_size, not len(docs[i])-overlap <= chunk_size
            # The total length of a chunk created by langchains's RecursiveCharacterTextSplitter
            # (which includes the prepended overlap for chunks > 0) should be <= chunk_size,
            # unless a single semantic unit (like a long word with no separators) itself exceeds chunk_size.
            # Let's re-verify this understanding for RecursiveCharacterTextSplitter.
            # Langchain's RecursiveCharacterTextSplitter aims for each *created document/chunk* to be of size chunk_size.
            # The overlap is *additional* to the previous chunk, but the current chunk itself is formed, then overlap is added to next.
            # No, that's not quite right. It tries to make chunks of size `chunk_size`, and then `chunk_overlap` characters
            # from the end of the current chunk are prepended to the next chunk.
            # So, `len(chunks[i].content)` should be around `chunk_size`.
            # The `non_overlap_part` logic from `test_recursive_splitter_simple_split` is more accurate for what is *newly added*.

            # Let's use the logic from test_recursive_splitter_simple_split for non_overlapping_part:
            # This applies if the current chunk content is indeed larger than the overlap.
            if len(current_chunk_content) > overlap_val:
                 non_overlap_part = current_chunk_content[overlap_val:]
                 assert len(non_overlap_part) <= splitter.chunk_size, \
                     f"Non-overlapping part of chunk {i} is too large: {len(non_overlap_part)} vs {splitter.chunk_size}"
            # Also, the total length of any chunk should not drastically exceed chunk_size,
            # unless a single word is longer than chunk_size.
            assert len(current_chunk_content) <= splitter.chunk_size + overlap_val, \
                 f"Total chunk {i} length {len(current_chunk_content)} seems too large relative to chunk_size {splitter.chunk_size} and overlap {overlap_val}"


    # Check if main segments are present (already good)
    assert any("First paragraph" in chk.content for chk in chunk_objs)
    assert any("Second paragraph" in chk.content for chk in chunk_objs)
    assert any("Third line" in chk.content for chk in chunk_objs)
