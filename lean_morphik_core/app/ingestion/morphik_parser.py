import io
import logging
import os
import tempfile
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import filetype
from unstructured.partition.auto import partition

from ..models.chunk import Chunk  # Adjusted import path
from .base_parser import BaseParser  # Adjusted import path
from .video.parse_video import VideoParser # Adjusted import path
# Removed load_config as it's tied to morphik.toml

# Custom RecursiveCharacterTextSplitter replaces langchain's version


logger = logging.getLogger(__name__)


class BaseChunker(ABC):
    """Base class for text chunking strategies"""

    @abstractmethod
    def split_text(self, text: str) -> List[Chunk]:
        """Split text into chunks"""
        pass


class StandardChunker(BaseChunker):
    """Standard chunking using langchain's RecursiveCharacterTextSplitter"""

    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def split_text(self, text: str) -> List[Chunk]:
        return self.text_splitter.split_text(text)


class RecursiveCharacterTextSplitter:
    def __init__(self, chunk_size: int, chunk_overlap: int, length_function=len, separators=None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.length_function = length_function
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

    def split_text(self, text: str) -> list[str]:
        chunks = self._split_recursive(text, self.separators)
        return [Chunk(content=chunk, metadata={}) for chunk in chunks]

    def _split_recursive(self, text: str, separators: list[str]) -> list[str]:
        if self.length_function(text) <= self.chunk_size:
            return [text] if text else []
        if not separators:
            # No separators left, split at chunk_size boundaries
            return [text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]
        sep = separators[0]
        if sep:
            splits = text.split(sep)
        else:
            # Last fallback: split every character
            splits = list(text)
        chunks = []
        current = ""
        for part in splits:
            add_part = part + (sep if sep and part != splits[-1] else "")
            if self.length_function(current + add_part) > self.chunk_size:
                if current:
                    chunks.append(current)
                current = add_part
            else:
                current += add_part
        if current:
            chunks.append(current)
        # If any chunk is too large, recurse further
        final_chunks = []
        for chunk in chunks:
            if self.length_function(chunk) > self.chunk_size and len(separators) > 1:
                final_chunks.extend(self._split_recursive(chunk, separators[1:]))
            else:
                final_chunks.append(chunk)
        # Handle overlap
        if self.chunk_overlap > 0 and len(final_chunks) > 1:
            overlapped = []
            for i in range(len(final_chunks)):
                chunk = final_chunks[i]
                if i > 0:
                    prev = final_chunks[i - 1]
                    overlap = prev[-self.chunk_overlap :]
                    chunk = overlap + chunk
                overlapped.append(chunk)
            return overlapped
        return final_chunks


class ContextualChunker(BaseChunker):
    """Contextual chunking using LLMs to add context to each chunk"""

    DOCUMENT_CONTEXT_PROMPT = """
    <document>
    {doc_content}
    </document>
    """

    CHUNK_CONTEXT_PROMPT = """
    Here is the chunk we want to situate within the whole document
    <chunk>
    {chunk_content}
    </chunk>

    Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk.
    Answer only with the succinct context and nothing else.
    """

    def __init__(self,
                 chunk_size: int,
                 chunk_overlap: int,
                 contextual_chunking_llm_model_name: str, # e.g. "claude-3-sonnet-20240229"
                 contextual_chunking_llm_api_key: Optional[str] = None, # Specific API key if needed by litellm
                 contextual_chunking_llm_base_url: Optional[str] = None # Specific base URL if needed
                ):
        self.standard_chunker = StandardChunker(chunk_size, chunk_overlap)
        self.contextual_chunking_llm_model_name = contextual_chunking_llm_model_name
        self.contextual_chunking_llm_api_key = contextual_chunking_llm_api_key
        self.contextual_chunking_llm_base_url = contextual_chunking_llm_base_url
        # Removed direct dependency on morphik.toml and core.config
        logger.info(f"Initialized ContextualChunker with model_name={self.contextual_chunking_llm_model_name}")

    def _situate_context(self, doc: str, chunk: str) -> str:
        import litellm

        # Model name is now passed directly
        model_name = self.contextual_chunking_llm_model_name

        # Create system and user messages
        system_message = {
            "role": "system",
            "content": "You are an AI assistant that situates a chunk within a document for the purposes of improving search retrieval of the chunk.",
        }

        # Add document context and chunk to user message
        user_message = {
            "role": "user",
            "content": f"{self.DOCUMENT_CONTEXT_PROMPT.format(doc_content=doc)}\n\n{self.CHUNK_CONTEXT_PROMPT.format(chunk_content=chunk)}",
        }

        # Prepare parameters for litellm
        model_params = {
            "model": model_name,
            "messages": [system_message, user_message],
            "max_tokens": 1024, # Default, can be overridden if passed in __init__
            "temperature": 0.0, # Default, can be overridden
        }

        if self.contextual_chunking_llm_api_key:
            model_params["api_key"] = self.contextual_chunking_llm_api_key
        if self.contextual_chunking_llm_base_url:
            model_params["base_url"] = self.contextual_chunking_llm_base_url

        # Removed iteration over self.model_config as it's no longer used.
        # Specific model parameters (e.g., max_tokens, temperature) could be added to __init__
        # and passed here if more granular control is needed.

        # Use litellm for completion
        response = litellm.completion(**model_params)
        return response.choices[0].message.content

    def split_text(self, text: str) -> List[Chunk]:
        base_chunks = self.standard_chunker.split_text(text)
        contextualized_chunks = []

        for chunk in base_chunks:
            context = self._situate_context(text, chunk.content)
            content = f"{context}; {chunk.content}"
            contextualized_chunks.append(Chunk(content=content, metadata=chunk.metadata))

        return contextualized_chunks


class MorphikParser(BaseParser):
    """Unified parser that handles different file types and chunking strategies"""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        use_unstructured_api: bool = False,
        unstructured_api_key: Optional[str] = None,
        assemblyai_api_key: Optional[str] = None,
        # anthropic_api_key is now part of contextual_chunking_llm_api_key
        video_frame_sample_rate: int = 120, # Default from original VideoParser config
        use_contextual_chunking: bool = False,
        contextual_chunking_llm_model_name: Optional[str] = "claude-3-sonnet-20240229", # Default
        contextual_chunking_llm_api_key: Optional[str] = None,
        contextual_chunking_llm_base_url: Optional[str] = None,
        # Added VideoParser specific vision model details
        video_vision_model_name: Optional[str] = "gpt-4-vision-preview", # Default from original VideoParser config
        video_vision_api_key: Optional[str] = None,
        video_vision_base_url: Optional[str] = None
    ):
        # Initialize basic configuration
        self.use_unstructured_api = use_unstructured_api
        self._unstructured_api_key = unstructured_api_key
        self._assemblyai_api_key = assemblyai_api_key
        self.video_frame_sample_rate = video_frame_sample_rate # Renamed from frame_sample_rate for clarity

        # VideoParser specific vision model details
        self.video_vision_model_name = video_vision_model_name
        self.video_vision_api_key = video_vision_api_key
        self.video_vision_base_url = video_vision_base_url

        # Initialize chunker based on configuration
        if use_contextual_chunking:
            if not contextual_chunking_llm_model_name:
                raise ValueError("contextual_chunking_llm_model_name must be provided if use_contextual_chunking is True")
            self.chunker = ContextualChunker(
                chunk_size,
                chunk_overlap,
                contextual_chunking_llm_model_name,
                contextual_chunking_llm_api_key,
                contextual_chunking_llm_base_url
            )
        else:
            self.chunker = StandardChunker(chunk_size, chunk_overlap)

    def _is_video_file(self, file: bytes, filename: str) -> bool:
        """Check if the file is a video file."""
        try:
            kind = filetype.guess(file)
            return kind is not None and kind.mime.startswith("video/")
        except Exception as e:
            logging.error(f"Error detecting file type: {str(e)}")
            return False

    async def _parse_video(self, file: bytes) -> Tuple[Dict[str, Any], str]:
        """Parse video file to extract transcript and frame descriptions"""
        if not self._assemblyai_api_key:
            raise ValueError("AssemblyAI API key is required for video parsing")

        # Save video to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(file)
            video_path = temp_file.name

        try:
            # frame_sample_rate is now directly available as self.video_frame_sample_rate
            # vision model details are also passed directly to VideoParser constructor

            # Process video
            parser = VideoParser(
                video_path,
                assemblyai_api_key=self._assemblyai_api_key,
                frame_sample_rate=self.video_frame_sample_rate,
                # Pass vision model details
                vision_model_name=self.video_vision_model_name,
                vision_api_key=self.video_vision_api_key,
                vision_base_url=self.video_vision_base_url
            )
            results = await parser.process_video()

            # Combine frame descriptions and transcript
            frame_text = "\n".join(results.frame_descriptions.time_to_content.values())
            transcript_text = "\n".join(results.transcript.time_to_content.values())
            combined_text = f"Frame Descriptions:\n{frame_text}\n\nTranscript:\n{transcript_text}"

            metadata = {
                "video_metadata": results.metadata,
                "frame_timestamps": list(results.frame_descriptions.time_to_content.keys()),
                "transcript_timestamps": list(results.transcript.time_to_content.keys()),
            }

            return metadata, combined_text
        finally:
            os.unlink(video_path)

    async def _parse_document(self, file: bytes, filename: str) -> Tuple[Dict[str, Any], str]:
        """Parse document using unstructured"""
        # Choose a lighter parsing strategy for text-based files. Using
        # `hi_res` on plain PDFs/Word docs invokes OCR which can be 20-30×
        # slower.  A simple extension check covers the majority of cases.
        strategy = "hi_res"
        file_content_type: Optional[str] = None  # Default to None for auto-detection
        if filename.lower().endswith((".pdf", ".doc", ".docx")):
            strategy = "fast"
        elif filename.lower().endswith(".txt"):
            strategy = "fast"
            file_content_type = "text/plain"  # Explicitly set for .txt files
        elif filename.lower().endswith(".json"):
            strategy = "fast"  # or can be omitted if it doesn't apply to json
            file_content_type = "application/json"  # Explicitly set for .json files

        elements = partition(
            file=io.BytesIO(file),
            content_type=file_content_type,  # Use the determined content_type
            metadata_filename=filename,
            strategy=strategy,
            api_key=self._unstructured_api_key if self.use_unstructured_api else None,
        )

        text = "\n\n".join(str(element) for element in elements if str(element).strip())
        return {}, text

    async def parse_file_to_text(self, file: bytes, filename: str) -> Tuple[Dict[str, Any], str]:
        """Parse file content into text based on file type"""
        if self._is_video_file(file, filename):
            return await self._parse_video(file)
        return await self._parse_document(file, filename)

    async def split_text(self, text: str) -> List[Chunk]:
        """Split text into chunks using configured chunking strategy"""
        return self.chunker.split_text(text)
