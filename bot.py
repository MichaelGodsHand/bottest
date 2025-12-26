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

from dotenv import load_dotenv
from google.genai.types import ThinkingConfig
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
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIObserver, RTVIProcessor
from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService, InputParams
from pipecat.transports.daily.transport import DailyParams, DailyTransport

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


async def run_bot(transport: DailyTransport):
    """Main bot execution function.

    Sets up and runs the bot pipeline including:
    - Gemini Live multimodal model integration
    - Voice activity detection
    - RTVI event handling
    """

    # Initialize the Gemini Multimodal Live model
    llm = GeminiLiveLLMService(
        api_key=os.getenv("GOOGLE_API_KEY"),
        model="gemini-2.5-flash-native-audio-preview-09-2025",
        voice_id="Charon",  # Aoede, Charon, Fenrir, Kore, Puck
        system_instruction=SYSTEM_INSTRUCTION,
        params=InputParams(thinking=ThinkingConfig(thinking_budget=0)),
    )

    messages = [
        {
            "role": "user",
            "content": "Start by introducing yourself briefly and let everyone know you're ready to observe and describe what you see in the video stream.",
        },
    ]

    # Set up conversation context and management
    # The context aggregator will automatically collect conversation context
    context = OpenAILLMContext(messages)
    context_aggregator = llm.create_context_aggregator(context)

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
    
    # Krisp is available when deployed to Pipecat Cloud
    if os.environ.get("ENV") != "local":
        try:
            from pipecat.audio.filters.krisp_filter import KrispFilter
            krisp_filter = KrispFilter()
        except ImportError:
            krisp_filter = None
    else:
        krisp_filter = None

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
