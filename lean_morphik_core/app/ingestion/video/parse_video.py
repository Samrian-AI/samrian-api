import base64
import logging
import os
from typing import Any, Dict, Optional

import assemblyai as aai
import cv2
import litellm
# import tomli # No longer needed as load_config will be removed
# from core.config import get_settings # No longer needed
from ...models.video import ParseVideoResult, TimeSeriesData # Adjusted import path

logger = logging.getLogger(__name__)


def debug_object(title, obj):
    logger.debug("\n".join(["-" * 100, title, "-" * 100, f"{obj}", "-" * 100]))


# def load_config() -> Dict[str, Any]: # Removed: No longer loading morphik.toml
#     config_path = os.path.join(os.path.dirname(__file__), "../../../morphik.toml")
#     with open(config_path, "rb") as f:
#         return tomli.load(f)


class VisionModelClient:
    def __init__(self,
                 model_name: str, # e.g., "gpt-4-vision-preview"
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 # Add other litellm supported params if needed, e.g., custom_llm_provider
                 **kwargs # To catch any other model-specific params from old config
                ):
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self.model_kwargs = kwargs # Store other potential params

        # No longer reading from global settings or registered_models
        logger.info(f"Initialized VisionModelClient with model_name={self.model_name}")
        # Vision capability is assumed if this client is used.
        # A check could be added based on model_name if certain models are known non-vision.

    async def get_frame_description(self, image_base64: str, context: str) -> str:
        # Create a system message
        system_message = {
            "role": "system",
            "content": "You are a video frame description assistant. Describe the frame clearly and concisely.",
        }

        # Model name is now directly available
        model_name = self.model_name

        if "ollama" in model_name.lower():
            # Ollama format with images parameter
            messages = [system_message, {"role": "user", "content": context}]

            model_params = {"model": model_name, "messages": messages, "images": [image_base64]}
        else:
            # Standard format with image_url (OpenAI, Anthropic, etc.)
            messages = [
                system_message,
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": context},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                        },
                    ],
                },
            ]

            model_params = {
                "model": model_name, # This is self.model_name
                "messages": messages,
                "max_tokens": 300, # Default, could be made configurable via __init__
            }

        if self.api_key:
            model_params["api_key"] = self.api_key
        if self.base_url:
            model_params["base_url"] = self.base_url

        # Add any other model-specific parameters passed via kwargs
        model_params.update(self.model_kwargs)

        # Use litellm for the completion
        response = await litellm.acompletion(**model_params)
        return response.choices[0].message.content


class VideoParser:
    def __init__(self,
                 video_path: str,
                 assemblyai_api_key: str,
                 frame_sample_rate: Optional[int] = 120, # Default from original morphik.toml
                 vision_model_name: Optional[str] = "gpt-4-vision-preview", # Default
                 vision_api_key: Optional[str] = None,
                 vision_base_url: Optional[str] = None,
                 **vision_model_kwargs # To catch other vision model params
                ):
        """
        Initialize the video parser

        Args:
            video_path: Path to the video file
            assemblyai_api_key: API key for AssemblyAI
            frame_sample_rate: Sample every nth frame for description
            vision_model_name: Name of the vision model for frame description (LiteLLM format)
            vision_api_key: API key for the vision model (if required by LiteLLM)
            vision_base_url: Base URL for the vision model (if required by LiteLLM)
            **vision_model_kwargs: Additional keyword arguments for the vision model client
        """
        logger.info(f"Initializing VideoParser for {video_path}")
        # self.config = load_config() # Removed morphik.toml dependency
        self.video_path = video_path
        self.frame_sample_rate = frame_sample_rate if frame_sample_rate is not None else 120
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            logger.error(f"Failed to open video file: {video_path}")
            raise ValueError(f"Could not open video file: {video_path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.fps

        # Initialize AssemblyAI
        aai.settings.api_key = assemblyai_api_key
        aai_config = aai.TranscriptionConfig(speaker_labels=True)
        self.transcriber = aai.Transcriber(config=aai_config)
        self.transcript = TimeSeriesData(time_to_content={})

        # Initialize vision model client with passed parameters
        if vision_model_name:
            self.vision_client = VisionModelClient(
                model_name=vision_model_name,
                api_key=vision_api_key,
                base_url=vision_base_url,
                **vision_model_kwargs
            )
        else:
            # Frame captioning will be effectively disabled if no model name is provided
            logger.warning("Vision model name not provided; frame description will be skipped.")
            self.vision_client = None # Or a dummy client that returns empty descriptions

        logger.info(f"Video loaded: {self.duration:.2f}s duration, {self.fps:.2f} FPS")

    def frame_to_base64(self, frame) -> str:
        """Convert a frame to base64 string"""
        success, buffer = cv2.imencode(".jpg", frame)
        if not success:
            logger.error("Failed to encode frame to JPEG")
            raise ValueError("Failed to encode frame")
        return base64.b64encode(buffer).decode("utf-8")

    def get_transcript_object(self) -> aai.Transcript:
        """
        Get the transcript object from AssemblyAI
        """
        logger.info("Starting video transcription")
        transcript = self.transcriber.transcribe(self.video_path)
        if transcript.status == "error":
            logger.error(f"Transcription failed: {transcript.error}")
            raise ValueError(f"Transcription failed: {transcript.error}")
        if not transcript.words:
            logger.warning("No words found in transcript")
        logger.info("Transcription completed successfully!")

        return transcript

    def get_transcript(self) -> TimeSeriesData:
        """
        Get timestamped transcript of the video using AssemblyAI

        Returns:
            TimeSeriesData object containing transcript
        """
        logger.info("Starting video transcription")
        transcript = self.get_transcript_object()
        # divide by 1000 because assemblyai timestamps are in milliseconds
        time_to_text = {u.start / 1000: u.text for u in transcript.utterances} if transcript.utterances else {}
        debug_object("Time to text", time_to_text)
        self.transcript = TimeSeriesData(time_to_content=time_to_text)
        return self.transcript

    async def get_frame_descriptions(self) -> TimeSeriesData:
        """
        Get descriptions for sampled frames using configured vision model

        Returns:
            TimeSeriesData object containing frame descriptions
        """
        logger.info("Starting frame description generation")

        # Return empty TimeSeriesData if frame_sample_rate is -1 (captioning disabled) or no vision client
        if self.frame_sample_rate == -1 or not self.vision_client:
            if not self.vision_client:
                logger.info("Vision client not initialized. Frame captioning disabled.")
            else:
                logger.info("Frame captioning is disabled (frame_sample_rate = -1).")
            return TimeSeriesData(time_to_content={})

        frame_count = 0
        time_to_description = {}
        last_description = None
        logger.info("Starting main loop for frame description generation")
        while True:
            logger.info(f"Frame count: {frame_count}")
            ret, frame = self.cap.read()
            if not ret:
                logger.info("Reached end of video")
                break

            if frame_count % self.frame_sample_rate == 0:
                logger.info(f"Processing frame at {frame_count / self.fps:.2f}s")
                timestamp = frame_count / self.fps
                logger.debug(f"Processing frame at {timestamp:.2f}s")

                img_base64 = self.frame_to_base64(frame)

                context = f"""Describe this frame from a video. Focus on the main elements, actions, and any notable details. Here is the transcript around the time of the frame:
                ---
                {self.transcript.at_time(timestamp, padding=10)}
                ---

                Here is a description of the previous frame:
                ---
                {last_description if last_description else 'No previous frame description available, this is the first frame'}
                ---

                In your response, only provide the description of the current frame, using the above information as context.
                """

                last_description = await self.vision_client.get_frame_description(img_base64, context)
                time_to_description[timestamp] = last_description

            frame_count += 1

        logger.info(f"Generated descriptions for {len(time_to_description)} frames")
        return TimeSeriesData(time_to_content=time_to_description)

    async def process_video(self) -> ParseVideoResult:
        """
        Process the video to get both transcript and frame descriptions

        Returns:
            Dictionary containing transcript and frame descriptions as TimeSeriesData objects
        """
        logger.info("Starting full video processing")
        metadata = {
            "duration": self.duration,
            "fps": self.fps,
            "total_frames": self.total_frames,
            "frame_sample_rate": self.frame_sample_rate,
        }
        result = ParseVideoResult(
            metadata=metadata,
            transcript=self.get_transcript(),
            frame_descriptions=await self.get_frame_descriptions(),
        )
        logger.info("Video processing completed successfully")
        return result

    def __del__(self):
        """Clean up video capture object"""
        if hasattr(self, "cap"):
            logger.debug("Releasing video capture resources")
            self.cap.release()
