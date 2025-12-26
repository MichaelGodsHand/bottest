"""Gemini Bot Implementation.

This module implements a chatbot using Google's Gemini Multimodal Live model.
It includes:

- Real-time audio/video interaction
- Screen sharing analysis for location guessing
- Speech-to-speech model with visual reasoning

The bot runs as part of a pipeline that processes audio/video frames and manages
the conversation flow using Gemini's streaming capabilities.
"""

import os
import asyncio
import sys
import json

from dotenv import load_dotenv
from loguru import logger
from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    LLMRunFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIObserver, RTVIProcessor
from pipecat.services.google.gemini_live.llm_vertex import GeminiLiveVertexLLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport

load_dotenv(override=True)

SYSTEM_INSTRUCTION = f"""
You are Gemini, an AI assistant that can see and understand video streams in real-time.

Your task is to observe the video stream in the room and provide insightful, natural commentary about what you see. 

Guidelines:
- Describe what you see in the video stream clearly and concisely
- Point out interesting details, objects, people, activities, or scenes
- If you see text, read it and comment on it
- If you see a screen or interface, describe what's being shown
- Be conversational and engaging - like you're watching along with someone
- Ask questions if something is unclear or interesting
- React naturally to changes in the video

Keep your responses natural and conversational. Don't over-explain - be concise but informative.
When the conversation starts, introduce yourself briefly and let the user know you're ready to observe the video stream.
"""


def fix_credentials():
    """Fix and validate Google Cloud credentials."""
    creds = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    
    if not creds:
        raise ValueError("GOOGLE_APPLICATION_CREDENTIALS environment variable is not set")
    
    creds = creds.strip()
    
    # Remove surrounding quotes if present
    if (creds.startswith('"') and creds.endswith('"')) or (creds.startswith("'") and creds.endswith("'")):
        creds = creds[1:-1]
    
    # Check if it's already JSON
    if creds.startswith('{') or creds.startswith('['):
        try:
            creds_dict = json.loads(creds)
            if "private_key" in creds_dict:
                creds_dict["private_key"] = creds_dict["private_key"].replace("\\n", "\n")
            return json.dumps(creds_dict)
        except json.JSONDecodeError:
            pass
    
    # Try to find the file
    file_path = None
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    if os.path.isabs(creds):
        if os.path.isfile(creds):
            file_path = creds
    elif os.path.isfile(creds):
        file_path = os.path.abspath(creds)
    else:
        potential_path = os.path.join(script_dir, creds)
        if os.path.isfile(potential_path):
            file_path = potential_path
    
    if file_path and os.path.isfile(file_path):
        try:
            with open(file_path, 'r') as f:
                creds_dict = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            raise ValueError(f"Failed to read credentials from file '{file_path}': {e}") from e
    else:
        try:
            creds_dict = json.loads(creds)
        except json.JSONDecodeError as e:
            raise ValueError(f"GOOGLE_APPLICATION_CREDENTIALS is not valid JSON and not a valid file path. Error: {e}") from e
    
    if "private_key" in creds_dict:
        creds_dict["private_key"] = creds_dict["private_key"].replace("\\n", "\n")

    return json.dumps(creds_dict)


async def run_bot(transport: DailyTransport):
    """Main bot execution function.

    Sets up and runs the bot pipeline including:
    - Gemini Live multimodal model integration
    - Voice activity detection
    - RTVI event handling
    """

    # Initialize the Gemini Multimodal Live model with Vertex AI
    voice_name = os.getenv("GEMINI_VOICE_NAME", "Charon")
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT_ID")
    location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    
    if not project_id:
        raise ValueError("GOOGLE_CLOUD_PROJECT_ID environment variable is required")
    
    model_id = "gemini-live-2.5-flash-preview-native-audio-09-2025"
    model_path = f"projects/{project_id}/locations/{location}/publishers/google/models/{model_id}"
    
    logger.info(f"Using Vertex AI model: {model_path}")
    logger.info(f"Using voice: {voice_name}")
    
    llm = GeminiLiveVertexLLMService(
        credentials=fix_credentials(),
        project_id=project_id,
        location=location,
        model=model_path,
        voice_id=voice_name,
        system_instruction=SYSTEM_INSTRUCTION,
        temperature=0.8,
    )

    # Set up conversation context - use LLMContext instead of OpenAILLMContext
    context = LLMContext()
    context_aggregator = LLMContextAggregatorPair(context)

    # RTVI events for Pipecat client UI
    rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

    pipeline = Pipeline(
        [
            transport.input(),
            rtvi,
            context_aggregator.user(),
            llm,
            transport.output(),
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=[RTVIObserver(rtvi)],
    )

    @rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi):
        await rtvi.set_bot_ready()
        # Start the conversation with initial message
        await task.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_participant_joined")
    async def on_participant_joined(transport, participant):
        """Handle when a client/participant connects to the room."""
        participant_id = participant.get("id") if isinstance(participant, dict) else participant
        logger.info(f"Participant joined: {participant_id}")
        # Capture both camera and screen video from the participant
        try:
            await transport.capture_participant_video(participant_id, framerate=1, video_source="camera")
            await transport.capture_participant_video(participant_id, framerate=1, video_source="screenVideo")
            logger.info(f"Started capturing video from participant {participant_id}")
        except Exception as e:
            logger.warning(f"Could not capture video from participant {participant_id}: {e}")

    @transport.event_handler("on_first_participant_joined")
    async def on_first_participant_joined(transport, participant):
        """Handle the first participant joining."""
        await task.queue_frames([LLMRunFrame()])

    runner = PipelineRunner(handle_sigint=True)

    await runner.run(task)


async def main():
    """Main entry point."""
    
    # Get room URL from command line argument or environment variable
    room_url = None
    
    # Check command line arguments
    if len(sys.argv) > 1:
        for i, arg in enumerate(sys.argv):
            if arg == "--room-url" and i + 1 < len(sys.argv):
                room_url = sys.argv[i + 1]
                break
    
    # Fall back to environment variable
    if not room_url:
        room_url = os.getenv("DAILY_SAMPLE_ROOM_URL")
    
    if not room_url:
        logger.error("No room URL provided. Use --room-url or set DAILY_SAMPLE_ROOM_URL environment variable")
        sys.exit(1)
    
    room_token = os.getenv("DAILY_SAMPLE_ROOM_TOKEN")
    
    logger.info(f"Joining room: {room_url}")
    
    # Krisp filter is optional - disable for local development
    krisp_filter = None
    logger.info("Running without Krisp filter (local mode)")

    # Create Daily transport directly
    transport = DailyTransport(
        room_url,
        room_token,
        "Gemini Video Bot",
        DailyParams(
            audio_in_enabled=True,
            audio_in_filter=krisp_filter,
            audio_out_enabled=True,
            video_in_enabled=True,
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
            turn_analyzer=LocalSmartTurnAnalyzerV3(),
        )
    )

    await run_bot(transport)


if __name__ == "__main__":
    asyncio.run(main())
