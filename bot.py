#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

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
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService, InputParams
from pipecat.transports.base_transport import BaseTransport
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


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments):
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

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, participant):
        """Handle when a client/participant connects to the room.
        
        This fires for both existing participants when the bot joins,
        and for new participants who join after the bot.
        """
        participant_id = participant.get("id") if isinstance(participant, dict) else participant
        logger.info(f"Client/Participant connected: {participant_id}")
        # Capture both camera and screen video from the participant
        try:
            await transport.capture_participant_video(participant_id, 1, "camera")
            await transport.capture_participant_video(participant_id, 1, "screenVideo")
            logger.info(f"Started capturing video from participant {participant_id}")
        except Exception as e:
            logger.warning(f"Could not capture video from participant {participant_id}: {e}")

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info(f"Client disconnected")
        # Don't cancel the task when a client disconnects - keep running to observe other participants

    runner = PipelineRunner(handle_sigint=runner_args.handle_sigint)

    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point for the bot starter."""

    # Krisp is available when deployed to Pipecat Cloud
    if os.environ.get("ENV") != "local":
        from pipecat.audio.filters.krisp_filter import KrispFilter

        krisp_filter = KrispFilter()
    else:
        krisp_filter = None

    # Get room URL from environment or command line
    room_url = os.getenv("DAILY_SAMPLE_ROOM_URL")
    room_token = os.getenv("DAILY_SAMPLE_ROOM_TOKEN")
    
    # Check if room URL is provided via command line arguments
    if hasattr(runner_args, 'room_url') and runner_args.room_url:
        room_url = runner_args.room_url
    elif hasattr(runner_args, 'args') and hasattr(runner_args.args, 'room_url'):
        room_url = getattr(runner_args.args, 'room_url', None) or room_url

    transport_params = {
        "daily": lambda: DailyParams(
            audio_in_enabled=True,
            audio_in_filter=krisp_filter,
            audio_out_enabled=True,
            video_in_enabled=True,
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
            turn_analyzer=LocalSmartTurnAnalyzerV3(),
        )
    }

    transport = await create_transport(runner_args, transport_params)
    
    # If room URL is set, try to join the room directly
    if room_url:
        logger.info(f"Room URL found in environment: {room_url}")
        logger.info("Attempting to join room directly...")
        
        # Log transport type and available methods for debugging
        logger.debug(f"Transport type: {type(transport)}")
        logger.debug(f"Transport attributes: {[attr for attr in dir(transport) if not attr.startswith('__')]}")
        
        try:
            # Method 1: Try direct join method on transport
            if hasattr(transport, 'join') and callable(getattr(transport, 'join')):
                logger.info("Trying transport.join()...")
                join_result = await transport.join(room_url, token=room_token) if room_token else await transport.join(room_url)
                logger.info(f"Join result: {join_result}")
            
            # Method 2: Try join_room method
            elif hasattr(transport, 'join_room') and callable(getattr(transport, 'join_room')):
                logger.info("Trying transport.join_room()...")
                join_result = await transport.join_room(room_url, token=room_token) if room_token else await transport.join_room(room_url)
                logger.info(f"Join result: {join_result}")
            
            # Method 3: Access underlying Daily client (for DailyTransport)
            elif isinstance(transport, DailyTransport):
                logger.info("Transport is DailyTransport, accessing internal client...")
                # Try various ways to access the Daily client
                daily_client = None
                for attr_name in ['_client', 'client', '_daily', '_call_client', 'call_client']:
                    if hasattr(transport, attr_name):
                        potential_client = getattr(transport, attr_name)
                        if potential_client and hasattr(potential_client, 'join'):
                            daily_client = potential_client
                            logger.info(f"Found Daily client via {attr_name}")
                            break
                
                if daily_client:
                    # Use the Daily client's join method (synchronous, like in videotest.py)
                    def on_joined(data, error):
                        if error:
                            logger.error(f"Failed to join room: {error}")
                        else:
                            logger.info(f"Successfully joined room: {room_url}")
                    
                    join_kwargs = {"url": room_url, "completion": on_joined}
                    if room_token:
                        join_kwargs["token"] = room_token
                    
                    daily_client.join(**join_kwargs)
                    logger.info("Called join on Daily client")
                    await asyncio.sleep(2)  # Give it time to join
                else:
                    logger.warning("Could not find Daily client in transport")
            
            # Method 4: Set room URL attribute that transport might check
            else:
                logger.info("Trying to set room URL as transport attribute...")
                for attr_name in ['_room_url', 'room_url', '_daily_room_url']:
                    if hasattr(transport, attr_name):
                        setattr(transport, attr_name, room_url)
                        logger.info(f"Set {attr_name} = {room_url}")
                        if room_token and hasattr(transport, '_room_token'):
                            setattr(transport, '_room_token', room_token)
                        break
                
        except Exception as e:
            logger.error(f"Error attempting to join room: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            logger.info("Bot will continue - transport may join room automatically or via web interface")

    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
