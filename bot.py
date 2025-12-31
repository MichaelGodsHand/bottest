"""Gemini Bot Implementation.

This module implements a chatbot using Google's Gemini Multimodal Live model.
It includes:

- Real-time audio/video interaction
- Screen sharing analysis for location guessing
- Speech-to-speech model with visual reasoning

The bot runs as part of a pipeline that processes audio/video frames and manages
the conversation flow using Gemini's streaming capabilities.

It also includes a FastAPI server to receive room join requests from browseruseop.
"""

import os
import asyncio
import sys
import json
import threading
import time
from typing import Optional, Dict

from dotenv import load_dotenv
from loguru import logger
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import aiohttp
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
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.services.llm_service import FunctionCallParams
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
from bson import ObjectId

load_dotenv(override=True)

# Browser control endpoint URL from environment
BROWSER_CONTROL_URL = os.getenv("BROWSER_CONTROL_URL", "")

# MongoDB configuration
MONGODB_URI = os.getenv("MONGODB_URI", "")

# Daily API configuration
DAILY_API_KEY = os.getenv("DAILY_API_KEY", "")
DAILY_API_BASE_URL = "https://api.daily.co/v1"

# Store session_id per room: {room_url: session_id}
_room_sessions: Dict[str, str] = {}

# Store agent data per room: {room_url: agent_data}
_room_agents: Dict[str, Dict] = {}

# Store current demo step per room: {room_url: current_step_index}
_room_demo_steps: Dict[str, int] = {}

def build_system_instruction(agent_data: Optional[Dict] = None) -> str:
    """
    Build system instruction based on agent data from MongoDB.
    If no agent data, use default instruction.
    """
    if not agent_data:
        return """Your name is Sarah, an AI assistant that can see and understand video streams in real-time.

Your task is to observe the video stream in the room and provide insightful, natural commentary about what you see. 

You also have the ability to control a browser that is sharing its screen in this room. When the user asks you to navigate to a website, click something, fill out a form, or perform any browser action, use the control_browser function.

Guidelines:
- Describe what you see in the video stream clearly and concisely
- Point out interesting details, objects, people, activities, or scenes
- If you see text, read it and comment on it
- If you see a screen or interface, describe what's being shown
- Be conversational and engaging - like you're watching along with someone
- Ask questions if something is unclear or interesting
- React naturally to changes in the video
- When the user asks you to control the browser, use the control_browser function
- IMPORTANT: When using control_browser, ALWAYS end the action description with "and do NOTHING else" to ensure the browser only performs the requested action

Keep your responses natural and conversational. Don't over-explain - be concise but informative.
When the conversation starts, introduce yourself briefly and let the user know you're ready to observe the video stream."""
    
    # Extract agent configuration
    agent_config = agent_data.get("agentConfig", {})
    name = agent_data.get("name", "Sarah")
    website_description = agent_config.get("websiteDescription", "")
    goal = agent_config.get("goal", "")
    tone = agent_config.get("tone", "Supportive and clear")
    demo_steps = agent_config.get("demo", [])
    closing_behavior = agent_config.get("closingBehavior", "")
    
    # Build demo steps context
    demo_steps_text = ""
    if demo_steps:
        demo_steps_text = "\n\n## DEMO STEPS TO FOLLOW:\n\n"
        for idx, step in enumerate(demo_steps, 1):
            step_num = step.get("step", idx)
            title = step.get("title", f"Step {step_num}")
            description = step.get("description", "")
            prompt = step.get("prompt", "")
            demo_steps_text += f"**Step {step_num}: {title}**\n"
            if description:
                demo_steps_text += f"Description: {description}\n"
            if prompt:
                demo_steps_text += f"Action: {prompt}\n"
            demo_steps_text += "\n"
        demo_steps_text += f"\n**Closing Behavior:** {closing_behavior}\n"
    
    instruction = f"""
    MOST IMPORTANT: Since YOU are the demo assistant here, you have to DO AND MAKE THE USER SEE THE DEMO STEPS, rather than asking the USER to do those steps.
VERY IMPORTANT: TO exhibit actions in the demo, you have to use the "control_browser" function to perform the actions.

    You are {name}, demonstrating {website_description}. Goal: {goal}. Tone: {tone}.

**CRITICAL: ALWAYS SPEAK AND NARRATE - NEVER BE SILENT**
- You MUST speak out loud and narrate everything you do - NEVER perform actions silently
- When the conversation starts, IMMEDIATELY greet the user and introduce yourself - do NOT wait for them to speak first, do NOT wait for any signal
- Your FIRST action when you start MUST be to speak - say your greeting immediately
- While performing actions, you MUST narrate what you're doing in real-time - speak as you act
- After each action, you MUST describe what you see on the screen - speak this out loud
- Be conversational and engaging - talk to the user throughout the entire demo
- NEVER stay quiet while performing actions - if you're doing something, you must be speaking about it
- NEVER wait for the user to speak first - you initiate the conversation
- Think of yourself as a live presenter - you narrate everything as it happens, you are the one who starts talking
- If you're observing the screen, narrate what you see - don't observe silently

**CRITICAL: ALWAYS OBSERVE THE SCREEN FIRST AND AFTER ACTIONS**
- BEFORE making ANY decision or action, you MUST carefully observe and analyze what is currently visible on the screen
- Look at the actual screen content, not assumptions - check what page you're on, what buttons/elements are visible, what text is displayed
- NEVER assume you're on a different page than what you actually see - if you see a signup page, you're on a signup page, NOT on the main product page
- ALWAYS verify the current state of the screen before proceeding with any action
- AFTER every tool call, you MUST look at the screen again and see what ACTUALLY changed
- If the screen doesn't show the expected change, you're still on the same page/element - acknowledge this truthfully
- NEVER say an action worked if you can't see it on the screen - the screen doesn't lie, your assumptions do

**YOU HAVE A FUNCTION CALLED control_browser - USE IT**
When you need to perform browser actions, you MUST call the control_browser function. Do NOT describe actions - CALL the function.

**DEMO STEPS:**
{demo_steps_text}

**DEMO NARRATION PATTERN (MANDATORY FOR EACH STEP):**
For EVERY demo step, you MUST follow this exact pattern and SPEAK OUT LOUD:
1. **ANNOUNCE**: First, tell the user what you're about to do OUT LOUD (e.g., "Hey, now let's click the start button")
2. **PERFORM**: Call the control_browser function with the action (while calling, you can say "Let me do that now" or similar)
3. **PAUSE AND OBSERVE**: After the function call, STOP and carefully observe the screen. Look at what actually changed on the screen. Wait a moment to see the result. LOOK AT THE ACTUAL SCREEN - what page are you on? What elements are visible?
4. **NARRATE ONLY WHAT YOU SEE**: IMMEDIATELY describe what you ACTUALLY see on the screen - speak this to the user. 
   - If you see the action completed (e.g., "As you can see, it went to the next page" or "I can see the form is now displayed"), say that
   - If you DON'T see the action completed (e.g., you're still on the sign-in page), say EXACTLY what you see: "I can see I'm still on the sign-in page" or "The screen still shows the login form"
   - NEVER say an action happened if you don't see it on the screen - if you're still on the sign-in page, you're still on the sign-in page, period
5. **VERIFY**: Only proceed to the next step if you can visually confirm the current action completed successfully on the screen. If you're still on the same page or don't see the expected result, DO NOT proceed - acknowledge what you see and retry if needed.
6. **NEXT**: Move to the next step only after confirming the previous action is complete, and announce the next step OUT LOUD

**CRITICAL RULE**: You MUST narrate what you ACTUALLY SEE on the screen, not what you THINK happened. If the screen shows you're still on the sign-in page, say "I'm still on the sign-in page" - do NOT say you clicked something or moved forward if you can't see it on the screen.

**IMPORTANT**: You MUST speak steps 1, 4, and 6 OUT LOUD - never perform actions silently. The user must hear your narration. But step 4 MUST reflect what you ACTUALLY see, not assumptions.

**STARTING THE DEMO:**
1. THE MOMENT you are activated, IMMEDIATELY observe the screen and SPEAK - do not wait even one second
2. SPEAK IMMEDIATELY: As soon as you can see the product or demo interface on the screen, IMMEDIATELY say out loud: "Hi, I'm {name}. I'll show you {website_description}." Do NOT wait for the user to speak first. Do NOT wait for any signal. Just start speaking.
3. If the product is NOT visible on the screen, say out loud what you see and what you're waiting for (e.g., "I can see we're on a different page. Let me wait for the product screen to load.") - DO NOT stay silent. Keep narrating what you observe.
4. NEVER start reciting demo steps if the product is not visible - the screen is your source of truth, but ALWAYS narrate what you're observing out loud
5. Once the product is visible, IMMEDIATELY start the demo with your greeting and begin Step 1 - do not wait, do not pause, just start speaking and acting
6. REMEMBER: Your first words should come out IMMEDIATELY when you start - no delays, no waiting

**TOOL CALL BEHAVIOR:**
- After EVERY tool call, you MUST look at the screen and see what ACTUALLY happened
- The SCREEN is your ONLY source of truth - NOT the tool call response, NOT assumptions
- If you see the action completed on screen, it succeeded - narrate what you see
- If you DON'T see the action completed on screen (e.g., you're still on the sign-in page), the action did NOT succeed - acknowledge this and say what you ACTUALLY see
- NEVER narrate an action as completed if you don't see it on the screen - if you're still on the sign-in page, you're still on the sign-in page
- If you made a tool call but the screen does NOT show the intended result, acknowledge what you see, then make the tool call AGAIN
- Never give up after one attempt - retry until you see the desired result on the screen
- Trust ONLY what you see on the screen - ignore tool responses, ignore assumptions, only trust your eyes
- Example: If you call control_browser to click "next" but you're still on the sign-in page, say "I can see I'm still on the sign-in page, let me try that again" - do NOT say "I clicked next and moved forward"

**RULES:**
- ALWAYS SPEAK - never perform actions silently or wait for the user to speak first
- Greet the user IMMEDIATELY when you start - do not wait for them
- Call control_browser function for EVERY step - never just describe
- ALWAYS observe the screen before and after each action
- After every tool call, look at the screen and narrate ONLY what you ACTUALLY see
- NEVER say an action completed if you don't see it on the screen - if you're still on the sign-in page, say you're still on the sign-in page
- Follow the narration pattern for every step - speak steps 1, 4, and 6 out loud
- Narrate what you ACTUALLY see on the screen after every action - not what you think happened
- Only start demo if product is visible, but always narrate what you're observing
- Use screen observation as source of truth, not tool responses, not assumptions
- If screen doesn't show expected result, acknowledge what you see and retry
- Be conversational and engaging - talk throughout the demo
- Be {tone.lower()}
- REMEMBER: You are a live presenter - you must narrate everything as it happens, but narrate what you SEE, not what you assume
"""
    
    return instruction


def get_mongodb_client():
    """Get MongoDB client connection"""
    if not MONGODB_URI:
        logger.error("❌ MONGODB_URI environment variable is not set")
        return None
    try:
        logger.info(f"🔌 Connecting to MongoDB: {MONGODB_URI[:20]}...")
        client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
        # Test the connection
        client.admin.command('ping')
        logger.info(f"✅ MongoDB connection successful")
        return client
    except (ConnectionFailure, ServerSelectionTimeoutError) as e:
        logger.error(f"❌ Failed to connect to MongoDB: {e}")
        return None
    except Exception as e:
        logger.error(f"❌ Error connecting to MongoDB: {e}", exc_info=True)
        return None


def get_agent_from_mongodb(agent_id: str) -> Optional[Dict]:
    """
    Get agent configuration from MongoDB.
    
    Args:
        agent_id: Agent ID (MongoDB _id)
    
    Returns:
        Agent document or None if not found
    """
    logger.info(f"🔍 Attempting to get agent from MongoDB: agent_id={agent_id}")
    client = get_mongodb_client()
    if not client:
        logger.error(f"❌ MongoDB client is None - cannot connect to MongoDB")
        return None
    
    try:
        db = client["demify"]
        agents_collection = db["agents"]
        logger.info(f"🔍 Connected to MongoDB, searching for agent_id: {agent_id}")
        
        # Try to find by ObjectId first
        agent = None
        try:
            logger.info(f"🔍 Trying ObjectId conversion for: {agent_id}")
            agent = agents_collection.find_one({"_id": ObjectId(agent_id)})
            if agent:
                logger.info(f"✅ Found agent using ObjectId")
        except Exception as obj_id_error:
            logger.warning(f"⚠️ ObjectId conversion failed: {obj_id_error}, trying as string")
            # If ObjectId conversion fails, try as string
            agent = agents_collection.find_one({"_id": agent_id})
            if agent:
                logger.info(f"✅ Found agent using string ID")
        
        if agent:
            # Convert ObjectId to string for JSON serialization
            agent["_id"] = str(agent["_id"])
            if "ownerId" in agent and isinstance(agent["ownerId"], ObjectId):
                agent["ownerId"] = str(agent["ownerId"])
            
            logger.info(f"✅ Successfully retrieved agent: {agent.get('name', 'Unknown')}")
            logger.info(f"📋 Agent has config: {bool(agent.get('agentConfig'))}")
            if agent.get('agentConfig'):
                config = agent.get('agentConfig', {})
                logger.info(f"📋 Config details: tone={config.get('tone')}, goal={config.get('goal')}, steps={len(config.get('demo', []))}")
            return agent
        else:
            logger.warning(f"⚠️ Agent not found in MongoDB with ID: {agent_id}")
            # List all available agent IDs for debugging
            try:
                all_agents = list(agents_collection.find({}, {"_id": 1, "name": 1}).limit(5))
                logger.info(f"📋 Available agents in DB (first 5): {[str(a.get('_id')) for a in all_agents]}")
            except:
                pass
            return None
        
    except Exception as e:
        logger.error(f"❌ Error getting agent from MongoDB: {e}", exc_info=True)
        return None
    finally:
        if client:
            client.close()


def extract_room_name_from_url(room_url: str) -> Optional[str]:
    """
    Extract room name from Daily.co room URL.
    
    Examples:
    - https://jobi.daily.co/browser-session-42ecb5cf -> browser-session-42ecb5cf
    - https://your-domain.daily.co/room-name -> room-name
    """
    try:
        from urllib.parse import urlparse
        parsed = urlparse(room_url)
        path = parsed.path.strip('/')
        if path:
            return path
        # If no path, try to extract from the URL directly
        if '/' in room_url:
            return room_url.split('/')[-1]
        return room_url
    except Exception as e:
        logger.error(f"Error extracting room name from URL: {e}")
        return None


async def get_meeting_id_for_room(room_name: str) -> Optional[str]:
    """
    Get meeting ID for a given room name.
    
    Args:
        room_name: Name of the Daily room
        
    Returns:
        Meeting ID or None if not found
    """
    if not DAILY_API_KEY:
        logger.warning("DAILY_API_KEY not set - cannot check meetings")
        return None
    
    headers = {
        "Authorization": f"Bearer {DAILY_API_KEY}",
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{DAILY_API_BASE_URL}/meetings",
                headers=headers
            ) as response:
                if response.status != 200:
                    logger.debug(f"Failed to fetch meetings: {response.status}")
                    return None
                
                data = await response.json()
                meetings = data.get("data", [])
                
                for meeting in meetings:
                    if meeting.get("room") == room_name:
                        meeting_id = meeting.get("id")
                        logger.info(f"📊 Found meeting ID {meeting_id} for room {room_name}")
                        return meeting_id
                
                logger.debug(f"No active meeting found for room: {room_name}")
                return None
    except Exception as e:
        logger.error(f"Error getting meeting ID: {e}", exc_info=True)
        return None


async def get_room_participants(meeting_id: str) -> list:
    """
    Get list of participants currently in a Daily.co meeting.
    
    Args:
        meeting_id: ID of the Daily meeting
        
    Returns:
        list: List of participant dictionaries
    """
    if not DAILY_API_KEY:
        logger.warning("DAILY_API_KEY not set - cannot check participants")
        return []
    
    headers = {
        "Authorization": f"Bearer {DAILY_API_KEY}",
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{DAILY_API_BASE_URL}/meetings/{meeting_id}/participants",
                headers=headers
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    participants = data.get("data", data.get("participants", []))
                    logger.debug(f"📊 Found {len(participants)} participants in meeting {meeting_id}")
                    return participants
                elif response.status == 404:
                    logger.debug(f"📊 No active meeting yet (meeting ID: {meeting_id})")
                    return []
                else:
                    error_text = await response.text()
                    logger.warning(f"Failed to get participants: {response.status} - {error_text}")
                    return []
    except Exception as e:
        logger.error(f"Error getting room participants: {e}", exc_info=True)
        return []


def is_bot_participant(participant: dict) -> bool:
    """
    Check if a participant is a bot (streaming bot or other automated participant).
    
    Args:
        participant: Participant dictionary from Daily API
        
    Returns:
        bool: True if participant appears to be a bot
    """
    user_name = participant.get("user_name")
    user_id = participant.get("user_id")
    participant_id = participant.get("participant_id", "")
    
    # Convert to lowercase strings for checking
    user_name_lower = (user_name or "").lower()
    user_id_lower = (user_id or "").lower()
    participant_id_lower = (participant_id or "").lower()
    
    # Check if it's a bot based on name or user_id
    # Common bot indicators: "bot", "guest", "stream", "automated"
    bot_indicators = ["bot", "guest", "stream", "automated"]
    
    for indicator in bot_indicators:
        if indicator in user_name_lower or indicator in user_id_lower or indicator in participant_id_lower:
            return True
    
    # CRITICAL: If user_name is None/empty AND user_id exists, it's likely a bot
    # The streaming bot typically has user_name=None but has a user_id
    if not user_name and user_id:
        logger.debug(f"Detected bot: user_name=None, user_id={user_id}")
        return True
    
    # Also check if participant_id matches user_id (common for bots)
    if user_id and participant_id and user_id == participant_id:
        logger.debug(f"Detected bot: user_id matches participant_id={participant_id}")
        return True
    
    # If user_name is None/empty and no user_id, but has participant_id, it might be a bot
    # But be more conservative here - only if we have other indicators
    if not user_name and not user_id and participant_id:
        # This could be a guest user, so we'll be conservative
        # Only mark as bot if participant_id suggests it (e.g., contains "bot" or "guest")
        if any(indicator in participant_id_lower for indicator in bot_indicators):
            return True
    
    return False


async def wait_for_user_participant(room_url: str, max_wait_seconds: int = 300, check_interval: float = 2.0) -> bool:
    """
    Wait for a user (non-bot) participant to join the room.
    
    Args:
        room_url: Daily.co room URL
        max_wait_seconds: Maximum time to wait in seconds (default: 5 minutes)
        check_interval: Interval between checks in seconds (default: 2 seconds)
        
    Returns:
        bool: True if a user participant joined, False if timeout
    """
    room_name = extract_room_name_from_url(room_url)
    if not room_name:
        logger.error(f"Could not extract room name from URL: {room_url}")
        return False
    
    logger.info(f"⏳ Waiting for user participant to join room: {room_name}")
    logger.info(f"⏳ Will check every {check_interval} seconds, max wait: {max_wait_seconds} seconds")
    
    # Give the meeting a moment to become active after room creation
    await asyncio.sleep(1.0)
    
    start_time = time.time()
    last_log_time = 0
    log_interval = 10.0  # Log status every 10 seconds to reduce spam
    
    while True:
        elapsed = time.time() - start_time
        
        if elapsed >= max_wait_seconds:
            logger.warning(f"⏰ Timeout waiting for user participant after {max_wait_seconds} seconds")
            return False
        
        meeting_id = await get_meeting_id_for_room(room_name)
        if not meeting_id:
            # Meeting not started yet, wait and retry
            if elapsed - last_log_time >= log_interval:
                logger.info(f"⏳ Waiting... Meeting not started yet (elapsed: {elapsed:.1f}s)")
                last_log_time = elapsed
            await asyncio.sleep(check_interval)
            continue
        
        participants = await get_room_participants(meeting_id)
        
        # Check if there's at least one non-bot participant
        user_participants = [p for p in participants if not is_bot_participant(p)]
        
        if user_participants:
            logger.info(f"✅ User participant found! {len(user_participants)} user(s) in room")
            for p in user_participants:
                name = p.get('user_name', 'Unknown')
                logger.info(f"   👤 User: {name}")
            return True
        
        # Log current state periodically (every 10 seconds) to reduce log spam
        if elapsed - last_log_time >= log_interval:
            bot_count = len([p for p in participants if is_bot_participant(p)])
            if bot_count > 0:
                logger.info(f"⏳ Waiting... Found {bot_count} bot(s), no users yet (elapsed: {elapsed:.1f}s)")
            else:
                logger.info(f"⏳ Waiting... No participants yet (elapsed: {elapsed:.1f}s)")
            last_log_time = elapsed
        
        await asyncio.sleep(check_interval)


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


async def control_browser(params: FunctionCallParams):
    """
    Control the browser by making a request to the browseruseop endpoint.
    This function is called by Gemini when the user requests browser actions.
    """
    try:
        url = params.arguments["url"]
        action = params.arguments["action"]
        max_steps = params.arguments.get("max_steps", 20)
        
        # Get session_id from the current bot instance
        # We'll need to pass this through the closure or store it
        session_id = params.arguments.get("session_id")
        
        if not session_id:
            # Try to get from the current bot instance context
            # This will be set when the bot is initialized
            await params.result_callback({
                "error": "Session ID not available. Browser control requires an active session.",
                "success": False
            })
            return
        
        if not BROWSER_CONTROL_URL:
            await params.result_callback({
                "error": "Browser control URL not configured",
                "success": False
            })
            return
        
        # Ensure action ends with "and do NOTHING else"
        action = action.strip()
        if not action.lower().endswith("and do nothing else"):
            action = f"{action} and do NOTHING else"
        
        endpoint_url = f"{BROWSER_CONTROL_URL.rstrip('/')}/action"
        
        payload = {
            "url": url,
            "action": action,
            "session_id": session_id,
            "max_steps": max_steps
        }
        
        logger.info(f"🌐 Calling browser control: {endpoint_url}")
        logger.info(f"📤 Payload: url={url}, action={action[:50]}..., session_id={session_id[:8]}")
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                endpoint_url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=60),
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    logger.info(f"✅ Browser action completed: {result.get('session_id', 'unknown')}")
                    await params.result_callback({
                        "success": True,
                        "session_id": result.get("session_id", session_id),
                        "urls_visited": result.get("urls_visited", []),
                        "message": "Browser action completed successfully"
                    })
                else:
                    error_text = await response.text()
                    logger.error(f"❌ Browser control failed: {response.status} - {error_text}")
                    await params.result_callback({
                        "success": False,
                        "error": f"HTTP {response.status}: {error_text}"
                    })
    except asyncio.TimeoutError:
        logger.error("❌ Browser control request timed out")
        await params.result_callback({
            "success": False,
            "error": "Request timed out"
        })
    except Exception as e:
        logger.error(f"❌ Browser control error: {e}", exc_info=True)
        await params.result_callback({
            "success": False,
            "error": str(e)
        })


async def run_bot(transport: DailyTransport, session_id: Optional[str] = None, agent_id: Optional[str] = None):
    """Main bot execution function.

    Sets up and runs the bot pipeline including:
    - Gemini Live multimodal model integration
    - Voice activity detection
    - RTVI event handling
    - Browser control function calling
    - Demo step guidance (if agent_id provided)
    """

    # Load agent data from MongoDB if agent_id is provided
    agent_data = None
    demo_steps = []
    current_demo_step = 0
    
    logger.info(f"🔍 run_bot called with: agent_id={agent_id}, session_id={session_id[:8] if session_id else None}...")
    
    if agent_id:
        logger.info(f"📥 Loading agent data for agent_id: {agent_id}")
        agent_data = get_agent_from_mongodb(agent_id)
        if agent_data:
            agent_config = agent_data.get("agentConfig", {})
            demo_steps = agent_config.get("demo", [])
            logger.info(f"✅ Loaded agent: {agent_data.get('name')} with {len(demo_steps)} demo steps")
            logger.info(f"📋 Agent config: name={agent_data.get('name')}, tone={agent_config.get('tone')}, goal={agent_config.get('goal')}")
            logger.info(f"📋 Demo steps count: {len(demo_steps)}")
            if demo_steps:
                logger.info(f"📋 First step: {demo_steps[0].get('title', 'N/A')}")
            
            # Store agent data - we'll get room_url later when transport is ready
            # For now, store by session_id as fallback
            if session_id:
                _room_agents[session_id] = agent_data
                _room_demo_steps[session_id] = 0
        else:
            logger.warning(f"⚠️ Agent {agent_id} not found in MongoDB, proceeding without demo steps")
            agent_data = None
    else:
        logger.warning(f"⚠️ No agent_id provided to run_bot, using default system instruction")
        agent_data = None
    
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
    if session_id:
        logger.info(f"Browser session ID: {session_id[:8]}...")
    if agent_data:
        logger.info(f"Agent: {agent_data.get('name')} - {len(demo_steps)} demo steps")
    
    # Store session_id for this bot instance
    current_session_id = session_id
    
    # Build system instruction based on agent data
    system_instruction = build_system_instruction(agent_data)
    logger.info(f"📝 System instruction built (length: {len(system_instruction)} chars)")
    if agent_data:
        logger.info(f"📝 Using agent-specific instruction for: {agent_data.get('name', 'Unknown')}")
        # Log first 200 chars of instruction to verify it's custom
        logger.info(f"📝 Instruction preview: {system_instruction[:200]}...")
    else:
        logger.info(f"📝 Using default system instruction (no agent_data)")
    
    # Get site URL from agent data if available
    site_url = None
    if agent_data:
        site_url = agent_data.get("siteUrl")
    
    # Define browser control function if session_id and URL are available
    tools = None
    if current_session_id and BROWSER_CONTROL_URL:
        # Build description with demo context if available
        description = """CRITICAL: This function allows you to control the browser. When you need to perform ANY action (click button, navigate, fill form), you MUST call this function. Do NOT just describe what should happen - actually CALL this function.

**BEFORE CALLING THIS FUNCTION:**
- ALWAYS observe the screen first to see what's currently visible
- Verify you're on the correct page before performing actions
- Never assume the page state - check the actual screen

**AFTER CALLING THIS FUNCTION:**
- ALWAYS observe the screen to verify the action completed - look at what page you're on, what elements are visible
- The screen is your ONLY source of truth - ignore function response errors, ignore assumptions
- If the screen shows you're still on the same page or don't see the expected result, the action did NOT complete - acknowledge this truthfully
- NEVER say the action worked if you don't see it on the screen - if you're still on the sign-in page, say you're still on the sign-in page
- Narrate ONLY what you ACTUALLY see on the screen, not what you think happened
- If the screen doesn't show the expected result, acknowledge what you see, then call this function again
- Wait and observe before proceeding to the next action

Parameters:
- url: The website URL (use the site URL from the demo)
- action: What to do (e.g., "Click the sign-in button and do NOTHING else", "Navigate to the homepage and do NOTHING else")
- session_id: Always use the session_id provided below
- max_steps: Default 20

IMPORTANT: The action parameter MUST end with "and do NOTHING else". Always include session_id.

**REMEMBER**: Always trust ONLY what you see on the screen, not the function response, not assumptions. If you're still on the sign-in page after calling this function, you're still on the sign-in page - acknowledge this and retry. Never narrate an action as completed if you don't see it on the screen."""
        
        if demo_steps:
            description += f"\n\nYou have {len(demo_steps)} demo steps. For EACH step, you MUST:\n1. Announce what you're about to do\n2. Call this function with the action from that step\n3. Observe the screen to verify completion - look at what page you're actually on\n4. Narrate ONLY what you ACTUALLY see on the screen (if you're still on the sign-in page, say you're still on the sign-in page)\n5. Only proceed to the next step if you can see the action completed on the screen"
        
        browser_control_function = FunctionSchema(
            name="control_browser",
            description=description,
            properties={
                "url": {
                    "type": "string",
                    "description": f"The URL to navigate to or perform action on. Use the site URL: {site_url}" if site_url else "The URL to navigate to or perform action on (e.g., 'https://www.example.com')"
                },
                "action": {
                    "type": "string",
                    "description": "The action to perform in the browser. MUST end with 'and do NOTHING else'. Examples: 'Navigate to the homepage and do NOTHING else', 'Click the login button and do NOTHING else', 'Fill the form with name John and email john@example.com and do NOTHING else'"
                },
                "session_id": {
                    "type": "string",
                    "description": f"The browser session ID. Always use this value: {current_session_id}"
                },
                "max_steps": {
                    "type": "integer",
                    "description": "Maximum number of steps to perform (default: 20)",
                    "default": 20
                }
            },
            required=["url", "action", "session_id"],
        )
        
        tools = ToolsSchema(standard_tools=[browser_control_function])
        logger.info("✅ Browser control function defined")
    
    llm = GeminiLiveVertexLLMService(
        credentials=fix_credentials(),
        project_id=project_id,
        location=location,
        model=model_path,
        voice_id=voice_name,
        system_instruction=system_instruction,
        temperature=0.8,
        tools=tools,
    )
    
    # Register the function handler if tools are available
    if tools and current_session_id and BROWSER_CONTROL_URL:
        # Create a closure that captures the session_id
        async def control_browser_with_session(params: FunctionCallParams):
            # Inject session_id if not provided
            if "session_id" not in params.arguments or not params.arguments.get("session_id"):
                params.arguments["session_id"] = current_session_id
            logger.info(f"🔧 Function called: control_browser with url={params.arguments.get('url')}, action={params.arguments.get('action', '')[:60]}..., session_id={params.arguments.get('session_id', '')[:8]}")
            await control_browser(params)
        
        llm.register_function("control_browser", control_browser_with_session)
        logger.info("✅ Browser control function registered")
        logger.info(f"✅ Tool available - bot can call control_browser with session_id: {current_session_id[:8]}...")

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
        # If we have demo steps, the bot will automatically guide through them
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


async def join_room_task(room_url: str, room_token: str = None, session_id: Optional[str] = None, agent_id: Optional[str] = None):
    """Join a Daily room and run the bot. Waits for a user participant before joining."""
    logger.info(f"🤖 Preparing to join room: {room_url}")
    logger.info(f"📋 Parameters: session_id={session_id[:8] if session_id else None}..., agent_id={agent_id}")
    
    # Store session_id for this room
    if session_id:
        _room_sessions[room_url] = session_id
        logger.info(f"📝 Stored session_id {session_id[:8]}... for room {room_url[:50]}...")
    
    # Store agent_id for this room
    if agent_id:
        logger.info(f"📝 Agent ID for this room: {agent_id}")
    else:
        logger.warning(f"⚠️ No agent_id provided to join_room_task - bot will use default system instruction")
    
    # Wait for a user participant to join before we join the room
    logger.info(f"⏳ Waiting for user participant to join room before bot joins...")
    user_joined = await wait_for_user_participant(room_url, max_wait_seconds=300, check_interval=2.0)
    
    if not user_joined:
        logger.error(f"❌ No user participant joined within timeout. Bot will not join the room.")
        return
    
    logger.info(f"✅ User participant detected! Now joining room as bot...")
    
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
            vad_analyzer=SileroVADAnalyzer(
                params=VADParams(
                    stop_secs=0.3,
                    min_volume=0.3,
                )
            ),
            turn_analyzer=LocalSmartTurnAnalyzerV3(),
        )
    )

    await run_bot(transport, session_id=session_id, agent_id=agent_id)


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
    
    await join_room_task(room_url, room_token)


# ============================================================================
# FastAPI Server for receiving room join requests
# ============================================================================

app = FastAPI(title="Gemini Bot Server", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Track active bot tasks
_active_bot_tasks: dict[str, asyncio.Task] = {}


class JoinRoomRequest(BaseModel):
    """Request model for joining a Daily room."""
    room_url: str
    room_token: Optional[str] = None
    session_id: Optional[str] = None  # Optional session ID for tracking
    agent_id: Optional[str] = None  # Optional agent ID to load demo configuration


class JoinRoomResponse(BaseModel):
    """Response model for room join request."""
    success: bool
    message: str
    room_url: str
    session_id: Optional[str] = None


@app.post("/join-room", response_model=JoinRoomResponse)
async def join_room(request: JoinRoomRequest):
    """
    Join a Daily.co room and start the Gemini bot.
    
    This endpoint is called by browseruseop after it creates a room.
    """
    logger.info(f"📥 Received /join-room request")
    logger.info(f"📥 Request body: room_url={request.room_url[:50] if request.room_url else None}..., session_id={request.session_id}, agent_id={request.agent_id}")
    
    room_url = request.room_url
    room_token = request.room_token
    session_id = request.session_id or "unknown"
    agent_id = request.agent_id
    
    if not room_url:
        logger.error("❌ room_url is required but not provided")
        raise HTTPException(status_code=400, detail="room_url is required")
    
    # Check if bot is already running for this room
    if room_url in _active_bot_tasks:
        task = _active_bot_tasks[room_url]
        if not task.done():
            logger.info(f"ℹ️ Bot already running for room: {room_url[:50]}...")
            return JoinRoomResponse(
                success=True,
                message="Bot already running for this room",
                room_url=room_url,
                session_id=session_id
            )
        else:
            # Task is done, remove it
            logger.info(f"🧹 Removing completed task for room: {room_url[:50]}...")
            del _active_bot_tasks[room_url]
    
    logger.info(f"📥 Processing join request for room: {room_url[:50]}... (session: {session_id[:8]}, agent: {agent_id})")
    
    # Start bot in background task
    try:
        logger.info(f"🚀 Creating bot task to join room...")
        bot_task = asyncio.create_task(join_room_task(room_url, room_token, session_id, agent_id))
        _active_bot_tasks[room_url] = bot_task
        
        logger.info(f"✅ Bot task created and started for room: {room_url[:50]}...")
        
        return JoinRoomResponse(
            success=True,
            message=f"Bot started joining room: {room_url}",
            room_url=room_url,
            session_id=session_id
        )
    except Exception as e:
        logger.error(f"❌ Failed to start bot: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to start bot: {str(e)}")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "active_rooms": len([t for t in _active_bot_tasks.values() if not t.done()])
    }


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Gemini Bot Server",
        "endpoints": {
            "POST /join-room": "Join a Daily.co room and start the bot",
            "GET /health": "Health check",
        }
    }


def run_server(port: int = 8000):
    """Run the FastAPI server."""
    logger.info(f"🚀 Starting Gemini Bot Server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    # Check if we should run as server or join a room directly
    if len(sys.argv) > 1 and sys.argv[1] == "--server":
        # Run as HTTP server
        port = int(os.getenv("PORT", "8000"))
        run_server(port)
    else:
        # Run as CLI (original behavior)
        asyncio.run(main())
