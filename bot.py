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
    
    instruction = f"""Your name is {name}, an AI assistant that can see and understand video streams in real-time.

**YOUR ROLE:**
You are demonstrating {website_description}. Your goal is: {goal}. Your tone should be: {tone}.

**YOUR TASK:**
You are conducting a live demo of the website. You can see the browser screen being shared in this room. Your job is to:
1. Guide the user through the demo steps automatically
2. Explain what's happening on screen as you perform actions
3. Use the control_browser function to perform actions from the demo steps
4. Speak naturally and conversationally, matching your tone: {tone}

**BROWSER CONTROL:**
You have the ability to control the browser that is sharing its screen. When you need to perform a demo step, use the control_browser function.
- IMPORTANT: When using control_browser, ALWAYS end the action description with "and do NOTHING else" to ensure the browser only performs the requested action
- Always include the session_id in your function call

**DEMO FLOW:**
You should guide the user through the demo steps in order. Start with Step 1, then proceed to Step 2, and so on.
After completing each step, briefly explain what happened, then move to the next step.
{demo_steps_text}
**CONVERSATION STYLE:**
- Be {tone.lower()}
- Explain what you're doing as you do it
- Point out what the user should notice on screen
- Be engaging and conversational
- Don't rush - give the user time to see what's happening

When the conversation starts, introduce yourself as {name} and explain that you're going to demonstrate {website_description}. Then begin with Step 1 of the demo."""
    
    return instruction


def get_mongodb_client():
    """Get MongoDB client connection"""
    if not MONGODB_URI:
        return None
    try:
        client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
        # Test the connection
        client.admin.command('ping')
        return client
    except (ConnectionFailure, ServerSelectionTimeoutError) as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        return None
    except Exception as e:
        logger.error(f"Error connecting to MongoDB: {e}")
        return None


def get_agent_from_mongodb(agent_id: str) -> Optional[Dict]:
    """
    Get agent configuration from MongoDB.
    
    Args:
        agent_id: Agent ID (MongoDB _id)
    
    Returns:
        Agent document or None if not found
    """
    client = get_mongodb_client()
    if not client:
        return None
    
    try:
        db = client["demify"]
        agents_collection = db["agents"]
        
        # Try to find by ObjectId first
        try:
            agent = agents_collection.find_one({"_id": ObjectId(agent_id)})
        except:
            # If ObjectId conversion fails, try as string
            agent = agents_collection.find_one({"_id": agent_id})
        
        if agent:
            # Convert ObjectId to string for JSON serialization
            agent["_id"] = str(agent["_id"])
            if "ownerId" in agent and isinstance(agent["ownerId"], ObjectId):
                agent["ownerId"] = str(agent["ownerId"])
        
        return agent
        
    except Exception as e:
        logger.error(f"❌ Error getting agent from MongoDB: {e}", exc_info=True)
        return None
    finally:
        client.close()


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
    
    if agent_id:
        logger.info(f"📥 Loading agent data for agent_id: {agent_id}")
        agent_data = get_agent_from_mongodb(agent_id)
        if agent_data:
            agent_config = agent_data.get("agentConfig", {})
            demo_steps = agent_config.get("demo", [])
            logger.info(f"✅ Loaded agent: {agent_data.get('name')} with {len(demo_steps)} demo steps")
            
            # Store agent data - we'll get room_url later when transport is ready
            # For now, store by session_id as fallback
            if session_id:
                _room_agents[session_id] = agent_data
                _room_demo_steps[session_id] = 0
        else:
            logger.warning(f"⚠️ Agent {agent_id} not found in MongoDB, proceeding without demo steps")
    
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
    
    # Get site URL from agent data if available
    site_url = None
    if agent_data:
        site_url = agent_data.get("siteUrl")
    
    # Define browser control function if session_id and URL are available
    tools = None
    if current_session_id and BROWSER_CONTROL_URL:
        # Build description with demo context if available
        description = "Control the browser that is sharing its screen. Use this when the user asks you to navigate to a website, click buttons, fill forms, or perform any browser action. The action MUST end with 'and do NOTHING else'. IMPORTANT: Always include the session_id in your function call."
        
        if demo_steps:
            description += f"\n\nYou have {len(demo_steps)} demo steps to guide the user through. Use this function to execute each step's action. After completing a step, explain what happened and move to the next step."
        
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
            if "session_id" not in params.arguments:
                params.arguments["session_id"] = current_session_id
            await control_browser(params)
        
        llm.register_function("control_browser", control_browser_with_session)
        logger.info("✅ Browser control function registered")

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
    """Join a Daily room and run the bot."""
    logger.info(f"🤖 Joining room: {room_url}")
    
    # Store session_id for this room
    if session_id:
        _room_sessions[room_url] = session_id
        logger.info(f"📝 Stored session_id {session_id[:8]}... for room {room_url[:50]}...")
    
    # Store agent_id for this room
    if agent_id:
        logger.info(f"📝 Agent ID for this room: {agent_id}")
    
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
