import pytest
import numpy as np
import torch
from PIL import Image as PILImageModule # To avoid conflict with an Image variable
from unittest.mock import patch, MagicMock, mock_open

from lean_morphik_core.app.embedding.colpali_embedding_model import ColpaliEmbeddingModel
from lean_morphik_core.app.models.chunk import Chunk # Adjust if your Chunk model is different

# Default embedding dimension for mocks
MOCK_EMBEDDING_DIM = 768

@pytest.fixture
def mock_automodel_and_processor(mocker):
    """Mocks AutoModel and AutoProcessor."""
    mock_model_instance = MagicMock()
    mock_model_instance.device = torch.device("cpu") # Default to CPU for tests
    mock_model_instance.config.hidden_size = MOCK_EMBEDDING_DIM

    # Mock the model's forward pass to return a tensor of the correct shape
    # The output structure is model_output.last_hidden_state
    mock_output_tensor = torch.randn(1, 10, MOCK_EMBEDDING_DIM) # (batch_size, sequence_length, hidden_size)
    mock_model_output = MagicMock()
    mock_model_output.last_hidden_state = mock_output_tensor
    mock_model_instance.return_value = mock_model_output # For when model is called like a function
    mock_model_instance.__call__ = MagicMock(return_value=mock_model_output) # If called as model(**inputs)


    mock_processor_instance = MagicMock()
    # Mock processor calls to return a dict like {'input_ids': ..., 'attention_mask': ...}
    # For simplicity, make it return a dict that can be passed to model's mock
    mock_processed_inputs = {"input_ids": torch.randint(0, 100, (1,10)), "attention_mask": torch.ones(1,10)}
    mock_processor_instance.return_value = mock_processed_inputs
    mock_processor_instance.__call__ = MagicMock(return_value=mock_processed_inputs)


    mocker.patch("lean_morphik_core.app.embedding.colpali_embedding_model.AutoModel.from_pretrained", return_value=mock_model_instance)
    mocker.patch("lean_morphik_core.app.embedding.colpali_embedding_model.AutoProcessor.from_pretrained", return_value=mock_processor_instance)

    return mock_model_instance, mock_processor_instance

@pytest.fixture
def embedding_model(mock_automodel_and_processor):
    """Provides an instance of ColpaliEmbeddingModel with mocked dependencies."""
    # Mock torch device checks for predictable behavior
    with patch("torch.backends.mps.is_available", return_value=False), \
         patch("torch.cuda.is_available", return_value=False):
        model = ColpaliEmbeddingModel(model_name="test_model", processor_name="test_processor", device="cpu")
    # After __init__, the model and processor attributes are set on the instance
    # We can further refine the mocks if needed, e.g., specific behavior for model.to()
    model.model, model.processor = mock_automodel_and_processor
    model.model.to = MagicMock(return_value=model.model) # Mock .to(device) call
    model.processor.to = MagicMock(return_value=model.processor) # Should not be called on processor usually

    # Ensure the mocked model.device reflects the test environment
    model.model.device = torch.device("cpu")

    # Mock the .to(self.model.device) calls within the processor's return value
    mock_processed_output = {"input_ids": torch.randn(1,5), "attention_mask": torch.ones(1,5)}
    mock_processed_output_with_to = MagicMock()
    mock_processed_output_with_to.to = MagicMock(return_value=mock_processed_output)
    model.processor.return_value = mock_processed_output_with_to

    return model

@pytest.mark.unit
def test_colpali_initialization_cpu(mock_automodel_and_processor):
    mock_model, mock_processor = mock_automodel_and_processor
    with patch("torch.backends.mps.is_available", return_value=False), \
         patch("torch.cuda.is_available", return_value=False):
        model = ColpaliEmbeddingModel(model_name="test/model", processor_name="test/processor", device="cpu")

    assert model.device == "cpu"
    mock_model.from_pretrained.assert_called_once_with("test/model", torch_dtype=torch.bfloat16, device_map="cpu", attn_implementation="eager", trust_remote_code=True)
    mock_processor.from_pretrained.assert_called_once_with("test/processor", trust_remote_code=True)
    assert model.batch_size == 1 # Default

@pytest.mark.unit
@pytest.mark.parametrize("cuda_available, mps_available, expected_device_str", [
    (True, False, "cuda"),
    (False, True, "mps"),
    (False, False, "cpu"),
])
def test_colpali_initialization_device_auto_select(mock_automodel_and_processor, cuda_available, mps_available, expected_device_str):
    mock_model, _ = mock_automodel_and_processor
    with patch("torch.cuda.is_available", return_value=cuda_available), \
         patch("torch.backends.mps.is_available", return_value=mps_available):
        model = ColpaliEmbeddingModel() # Use default constructor args

    assert model.device == expected_device_str
    # Check if attn_implementation is correctly set for CUDA
    expected_attn_impl = "flash_attention_2" if expected_device_str == "cuda" else "eager"
    mock_model.from_pretrained.assert_called_with(
        "tsystems/colqwen2.5-3b-multilingual-v1.0", # default model
        torch_dtype=torch.bfloat16,
        device_map=expected_device_str,
        attn_implementation=expected_attn_impl,
        trust_remote_code=True
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_generate_embeddings_text(embedding_model: ColpaliEmbeddingModel):
    text = "This is a test sentence."
    embedding = await embedding_model.generate_embeddings(text)

    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (MOCK_EMBEDDING_DIM,)
    embedding_model.processor.assert_called_with(text=text, return_tensors="pt", padding=True, truncation=True)
    # embedding_model.model.assert_called_once() # Model is called via __call__

@pytest.mark.unit
@pytest.mark.asyncio
async def test_generate_embeddings_image(embedding_model: ColpaliEmbeddingModel, mocker):
    # Create a dummy image using PIL
    dummy_pil_image = PILImageModule.new('RGB', (60, 30), color = 'red')

    # Mock the processor call for an image
    mock_processed_image_input = {"pixel_values": torch.randn(1,3,30,60)}
    mock_processed_image_input_with_to = MagicMock()
    mock_processed_image_input_with_to.to = MagicMock(return_value=mock_processed_image_input)
    embedding_model.processor.return_value = mock_processed_image_input_with_to

    embedding = await embedding_model.generate_embeddings(dummy_pil_image)

    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (MOCK_EMBEDDING_DIM,)
    embedding_model.processor.assert_called_with(images=dummy_pil_image, return_tensors="pt")
    # embedding_model.model.assert_called_once()

@pytest.mark.unit
@pytest.mark.asyncio
async def test_embed_for_query(embedding_model: ColpaliEmbeddingModel):
    query_text = "Test query"
    embedding = await embedding_model.embed_for_query(query_text)

    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (MOCK_EMBEDDING_DIM,)
    # generate_embeddings is called internally, so its mocks will be checked

# --- Test embed_for_ingestion ---
@pytest.mark.unit
@pytest.mark.asyncio
async def test_embed_for_ingestion_single_text_chunk(embedding_model: ColpaliEmbeddingModel):
    chunk = Chunk(content="Text content", metadata={"is_image": False})
    embeddings = await embedding_model.embed_for_ingestion(chunk)

    assert len(embeddings) == 1
    assert isinstance(embeddings[0], np.ndarray)
    assert embeddings[0].shape == (MOCK_EMBEDDING_DIM,)

@pytest.mark.unit
@pytest.mark.asyncio
async def test_embed_for_ingestion_multiple_text_chunks(embedding_model: ColpaliEmbeddingModel):
    chunks = [
        Chunk(content="Text 1", metadata={"is_image": False}),
        Chunk(content="Text 2", metadata={"is_image": False})
    ]
    # Configure processor and model for batch outputs
    # For processor, assume it handles list of texts and returns batched tensors
    mock_batched_text_input = {"input_ids": torch.randn(2,5), "attention_mask": torch.ones(2,5)}
    mock_batched_text_input_with_to = MagicMock()
    mock_batched_text_input_with_to.to = MagicMock(return_value=mock_batched_text_input)
    embedding_model.processor.return_value = mock_batched_text_input_with_to

    # For model, assume it returns batched embeddings
    mock_batched_model_output = MagicMock()
    mock_batched_model_output.last_hidden_state = torch.randn(2, 5, MOCK_EMBEDDING_DIM) # (batch_size, seq_len, hidden_size)
    embedding_model.model.return_value = mock_batched_model_output
    embedding_model.model.__call__ = MagicMock(return_value=mock_batched_model_output)


    embeddings = await embedding_model.embed_for_ingestion(chunks)

    assert len(embeddings) == 2
    assert isinstance(embeddings[0], np.ndarray)
    assert embeddings[0].shape == (MOCK_EMBEDDING_DIM,)
    assert isinstance(embeddings[1], np.ndarray)

@pytest.mark.unit
@pytest.mark.asyncio
async def test_embed_for_ingestion_single_image_chunk(embedding_model: ColpaliEmbeddingModel, mocker):
    # Base64 encoded tiny red dot GIF
    b64_content = "R0lGODlhAQABAIABAP8AAP///yH5BAEAAAEALAAAAAABAAEAAAICRAEAOw=="
    chunk = Chunk(content=f"data:image/gif;base64,{b64_content}", metadata={"is_image": True})

    mock_image_obj = MagicMock(spec=PILImageModule.Image)
    mocker.patch("lean_morphik_core.app.embedding.colpali_embedding_model.open_image", return_value=mock_image_obj)

    # Mock processor for image
    mock_processed_image_input = {"pixel_values": torch.randn(1,3,1,1)}
    mock_processed_image_input_with_to = MagicMock()
    mock_processed_image_input_with_to.to = MagicMock(return_value=mock_processed_image_input)
    embedding_model.processor.return_value = mock_processed_image_input_with_to

    # Mock model for single image
    mock_single_image_model_output = MagicMock()
    mock_single_image_model_output.last_hidden_state = torch.randn(1, 1, MOCK_EMBEDDING_DIM)
    embedding_model.model.return_value = mock_single_image_model_output
    embedding_model.model.__call__ = MagicMock(return_value=mock_single_image_model_output)


    embeddings = await embedding_model.embed_for_ingestion(chunk)

    assert len(embeddings) == 1
    assert isinstance(embeddings[0], np.ndarray)
    assert embeddings[0].shape == (MOCK_EMBEDDING_DIM,)
    # lean_morphik_core.app.embedding.colpali_embedding_model.open_image.assert_called_once() # Fails with mocker

@pytest.mark.unit
@pytest.mark.asyncio
async def test_embed_for_ingestion_mixed_chunks_batching(embedding_model: ColpaliEmbeddingModel, mocker):
    # Base64 encoded tiny red dot GIF
    b64_content = "R0lGODlhAQABAIABAP8AAP///yH5BAEAAAEALAAAAAABAAEAAAICRAEAOw=="
    chunks = [
        Chunk(content="Text chunk 1", metadata={"is_image": False}),
        Chunk(content=f"data:image/gif;base64,{b64_content}", metadata={"is_image": True}),
        Chunk(content="Text chunk 2", metadata={"is_image": False}),
    ]
    embedding_model.batch_size = 2 # Test batching

    mock_image_obj = MagicMock(spec=PILImageModule.Image)
    mocker.patch("lean_morphik_core.app.embedding.colpali_embedding_model.open_image", return_value=mock_image_obj)

    # This test requires more intricate mocking of processor and model for mixed batches
    # or separate calls. The current implementation separates image and text processing.

    # Mock for generate_embeddings_batch_texts
    # Mock processor for text batch
    mock_text_batch_input = {"input_ids": torch.randn(2,5), "attention_mask": torch.ones(2,5)}
    mock_text_batch_input_with_to = MagicMock()
    mock_text_batch_input_with_to.to = MagicMock(return_value=mock_text_batch_input)

    # Mock model for text batch
    mock_text_batch_model_output = MagicMock()
    mock_text_batch_model_output.last_hidden_state = torch.randn(2, 5, MOCK_EMBEDDING_DIM)

    # Mock for generate_embeddings_batch_images
    # Mock processor for image batch
    mock_image_batch_input = {"pixel_values": torch.randn(1,3,1,1)}
    mock_image_batch_input_with_to = MagicMock()
    mock_image_batch_input_with_to.to = MagicMock(return_value=mock_image_batch_input)

    # Mock model for image batch
    mock_image_batch_model_output = MagicMock()
    mock_image_batch_model_output.last_hidden_state = torch.randn(1, 1, MOCK_EMBEDDING_DIM)

    # Side effect for processor: return text-formatted then image-formatted
    embedding_model.processor.side_effect = [
        mock_text_batch_input_with_to,  # For the two text chunks
        mock_image_batch_input_with_to  # For the one image chunk
    ]
    # Side effect for model: return text embeds then image embeds
    embedding_model.model.side_effect = [
        mock_text_batch_model_output, # For text
        mock_image_batch_model_output  # For image
    ]
    # If __call__ is used:
    embedding_model.model.__call__.side_effect = [
        mock_text_batch_model_output,
        mock_image_batch_model_output
    ]


    embeddings = await embedding_model.embed_for_ingestion(chunks)

    assert len(embeddings) == 3
    assert all(isinstance(e, np.ndarray) and e.shape == (MOCK_EMBEDDING_DIM,) for e in embeddings)
    # Further checks could verify that processor and model were called correctly for text and image batches.

@pytest.mark.unit
@pytest.mark.asyncio
async def test_embed_for_ingestion_empty_list(embedding_model: ColpaliEmbeddingModel):
    embeddings = await embedding_model.embed_for_ingestion([])
    assert embeddings == []

@pytest.mark.unit
@pytest.mark.asyncio
async def test_generate_embeddings_batch_texts_empty(embedding_model: ColpaliEmbeddingModel):
    result = await embedding_model.generate_embeddings_batch_texts([])
    assert result == []

@pytest.mark.unit
@pytest.mark.asyncio
async def test_generate_embeddings_batch_images_empty(embedding_model: ColpaliEmbeddingModel):
    result = await embedding_model.generate_embeddings_batch_images([])
    assert result == []

@pytest.mark.unit
@pytest.mark.asyncio
async def test_image_processing_failure_fallback_to_text(embedding_model: ColpaliEmbeddingModel, mocker):
    chunk_img_bad = Chunk(content="not_a_base64_string", metadata={"is_image": True})
    mocker.patch("base64.b64decode", side_effect=Exception("bad decode"))

    # Mock text processing for the fallback
    mock_text_input = {"input_ids": torch.randn(1,5), "attention_mask": torch.ones(1,5)}
    mock_text_input_with_to = MagicMock()
    mock_text_input_with_to.to = MagicMock(return_value=mock_text_input)
    embedding_model.processor.return_value = mock_text_input_with_to # Assume it falls back to text processing

    mock_text_model_output = MagicMock()
    mock_text_model_output.last_hidden_state = torch.randn(1, 5, MOCK_EMBEDDING_DIM)
    embedding_model.model.return_value = mock_text_model_output
    embedding_model.model.__call__ = MagicMock(return_value=mock_text_model_output)

    embeddings = await embedding_model.embed_for_ingestion([chunk_img_bad])

    assert len(embeddings) == 1
    assert embeddings[0].shape == (MOCK_EMBEDDING_DIM,)
    # Check that processor was called with text (the original bad content)
    # This depends on the exact fallback logic in ColpaliEmbeddingModel
    # The current Colpali model logs error and appends original content to text_items.
    embedding_model.processor.assert_any_call(text=[chunk_img_bad.content], return_tensors="pt", padding=True, truncation=True)

@pytest.mark.unit
@pytest.mark.asyncio
async def test_image_processor_pil_failure_in_generate_embeddings(embedding_model: ColpaliEmbeddingModel, mocker):
    dummy_pil_image = PILImageModule.new('RGB', (60, 30), color = 'red')

    # Mock the processor to raise an exception when processing the image
    embedding_model.processor.side_effect = [Exception("PIL processing error"), MagicMock()] # First call fails, second (fallback) succeeds

    # Mock for the text fallback processor call
    mock_fallback_text_input = {"input_ids": torch.randn(1,5), "attention_mask": torch.ones(1,5)}
    mock_fallback_text_input_with_to = MagicMock()
    mock_fallback_text_input_with_to.to = MagicMock(return_value=mock_fallback_text_input)

    # Re-assign processor mock for the fallback
    embedding_model.processor = MagicMock(side_effect=[Exception("PIL processing error"), mock_fallback_text_input_with_to])


    embedding = await embedding_model.generate_embeddings(dummy_pil_image)

    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (MOCK_EMBEDDING_DIM,)
    # Check that processor was called for image first, then for text fallback
    calls = embedding_model.processor.call_args_list
    assert calls[0][1]['images'] == dummy_pil_image # First call with image
    assert calls[1][1]['text'] == "[image placeholder]"  # Second call with placeholder text

@pytest.mark.unit
@pytest.mark.asyncio
async def test_batch_image_processing_failure_returns_zeros(embedding_model: ColpaliEmbeddingModel):
    dummy_pil_image1 = PILImageModule.new('RGB', (10, 10), color='blue')
    dummy_pil_image2 = PILImageModule.new('RGB', (10, 10), color='green')

    # Make processor fail for batch image processing
    embedding_model.processor.side_effect = Exception("Batch image processing failed")

    embeddings = await embedding_model.generate_embeddings_batch_images([dummy_pil_image1, dummy_pil_image2])

    assert len(embeddings) == 2
    assert np.array_equal(embeddings[0], np.zeros(MOCK_EMBEDDING_DIM))
    assert np.array_equal(embeddings[1], np.zeros(MOCK_EMBEDDING_DIM))
    embedding_model.processor.assert_called_once_with(images=[dummy_pil_image1, dummy_pil_image2], return_tensors="pt", padding=True)
