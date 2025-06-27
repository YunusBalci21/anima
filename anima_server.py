"""
ANIMA Real-time Server - FIXED VERSION with proper imports
Connects the enhanced simulation to web interface via WebSocket
"""

import asyncio
import json
import logging
import time
from typing import Dict, Optional
from datetime import datetime
from enum import Enum
import os

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from world_sim import SimulationWorld, SimulationConfig
from anima_deepseek_agent import initialize_deepseek, Emotion, Action, ResourceType

def safe_json_serializer(obj):
    """Safe JSON serializer that handles enums and other problematic objects"""
    if isinstance(obj, Enum):
        return obj.value
    elif hasattr(obj, 'value') and not isinstance(obj, (str, int, float, bool)):
        return obj.value
    elif isinstance(obj, tuple):
        return list(obj)
    elif isinstance(obj, set):
        return list(obj)
    elif hasattr(obj, 'to_dict'):
        return obj.to_dict()
    elif hasattr(obj, '__dict__'):
        result = {}
        for key, value in obj.__dict__.items():
            try:
                json.dumps(value)
                result[key] = value
            except (TypeError, ValueError):
                if isinstance(value, Enum):
                    result[key] = value.value
                elif hasattr(value, 'value'):
                    result[key] = str(value)
                else:
                    result[key] = str(value)
        return result
    else:
        return str(obj)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ANIMA-Server")

# Create FastAPI app
app = FastAPI(title="ANIMA - The Emergent Self Engine")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
class SimulationManager:
    def __init__(self):
        self.world: Optional[SimulationWorld] = None
        self.config: Optional[SimulationConfig] = None
        self.is_running: bool = False
        self.is_paused: bool = False
        self.speed_multiplier: float = 1.0
        self.connected_clients: list = []
        self.simulation_task: Optional[asyncio.Task] = None
        self.state_history: list = []
        self.max_history: int = 100
        self.model_loading: bool = False
        self.model_loaded: bool = False
        self.model_error: Optional[str] = None

    async def initialize_simulation(self, config: SimulationConfig):
        """Initialize a new simulation"""
        self.config = config

        # Create world without LLM first for faster startup
        temp_config = SimulationConfig(
            world_size=config.world_size,
            initial_agents=config.initial_agents,
            resource_spawn_rate=config.resource_spawn_rate,
            time_per_tick=config.time_per_tick,
            use_llm=False  # Start without LLM
        )

        self.world = SimulationWorld(temp_config)
        self.is_running = False
        self.is_paused = False
        self.state_history = []

        logger.info(f"Initialized simulation with {config.initial_agents} agents (LLM loading separately)")

        # Load LLM in background if requested
        if config.use_llm:
            asyncio.create_task(self._load_llm_async(config))

    async def _load_llm_async(self, config: SimulationConfig):
        """Load LLM model asynchronously"""
        self.model_loading = True
        self.model_error = None

        try:
            logger.info("🧠 Starting DeepSeek model loading in background...")
            await self.broadcast_message({
                "type": "status",
                "message": "Loading DeepSeek consciousness engine...",
                "loading": True
            })

            # Load model in thread to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._load_model_sync, config)

            # Update world config to use LLM
            if self.world:
                self.world.config.use_llm = True
                self.world.config.model_name = config.model_name
                self.world.config.temperature = config.temperature
                self.world.config.llm_awakening_age = config.llm_awakening_age
                self.world.config.llm_awakening_wisdom = config.llm_awakening_wisdom

                # Update existing agents to potentially use LLM
                for agent in self.world.agents.values():
                    agent.config = self.world.config
                    agent.check_consciousness_evolution()

            self.model_loaded = True
            self.model_loading = False

            logger.info("✅ DeepSeek model loaded successfully!")
            await self.broadcast_message({
                "type": "status",
                "message": "DeepSeek consciousness engine ready!",
                "model_loaded": True,
                "loading": False
            })

        except Exception as e:
            self.model_error = str(e)
            self.model_loading = False
            logger.error(f"❌ Failed to load DeepSeek model: {e}")
            await self.broadcast_message({
                "type": "status",
                "message": f"LLM loading failed: {str(e)[:100]}...",
                "error": True,
                "loading": False
            })

    def _load_model_sync(self, config: SimulationConfig):
        """Synchronous model loading"""
        try:
            initialize_deepseek(config)
        except Exception as e:
            logger.error(f"Model loading error: {e}")
            raise

    async def start_simulation(self):
        """Start the simulation loop"""
        if not self.world:
            raise ValueError("Simulation not initialized")

        self.is_running = True
        self.is_paused = False
        self.simulation_task = asyncio.create_task(self._run_simulation_loop())
        logger.info("Simulation started")

    async def pause_simulation(self):
        """Pause the simulation"""
        self.is_paused = True
        logger.info("Simulation paused")

    async def resume_simulation(self):
        """Resume the simulation"""
        self.is_paused = False
        logger.info("Simulation resumed")

    async def stop_simulation(self):
        """Stop the simulation"""
        self.is_running = False
        if self.simulation_task:
            self.simulation_task.cancel()
        logger.info("Simulation stopped")

    async def _run_simulation_loop(self):
        """Main simulation loop"""
        while self.is_running:
            if not self.is_paused and self.world:
                try:
                    # Run one tick
                    await self.world.tick()

                    # Get current state
                    state = self.world.get_world_state()

                    # Add model status to state
                    state["model_status"] = {
                        "loading": self.model_loading,
                        "loaded": self.model_loaded,
                        "error": self.model_error
                    }

                    # Add to history
                    self.state_history.append(state)
                    if len(self.state_history) > self.max_history:
                        self.state_history.pop(0)

                    # Broadcast to all connected clients
                    await self.broadcast_state(state)

                    # Sleep based on speed multiplier
                    await asyncio.sleep(self.config.time_per_tick / self.speed_multiplier)

                except Exception as e:
                    logger.error(f"Error in simulation loop: {e}")
                    import traceback
                    traceback.print_exc()
                    await asyncio.sleep(1)
            else:
                await asyncio.sleep(0.1)

    async def broadcast_state(self, state: Dict):
        """Broadcast world state to all connected clients"""
        message = {
            "type": "state",
            "data": state,
            "timestamp": time.time()
        }
        await self.broadcast_message(message)

    async def broadcast_message(self, message: Dict):
        """Broadcast any message to all connected clients"""
        disconnected = []

        for client in self.connected_clients:
            try:
                safe_message = json.loads(json.dumps(message, default=safe_json_serializer))
                await client.send_text(json.dumps(safe_message))
            except Exception as e:
                logger.error(f"Error broadcasting to client: {e}")
                disconnected.append(client)

        # Remove disconnected clients
        for client in disconnected:
            if client in self.connected_clients:
                self.connected_clients.remove(client)

    def add_client(self, websocket: WebSocket):
        """Add a new connected client"""
        self.connected_clients.append(websocket)
        logger.info(f"Client connected. Total clients: {len(self.connected_clients)}")

    def remove_client(self, websocket: WebSocket):
        """Remove a disconnected client"""
        if websocket in self.connected_clients:
            self.connected_clients.remove(websocket)
        logger.info(f"Client disconnected. Total clients: {len(self.connected_clients)}")

# Create simulation manager
sim_manager = SimulationManager()

# Mount static files
static_dir = "static"
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# API Routes
@app.get("/")
async def root():
    """Serve the main HTML page"""
    html_path = os.path.join(static_dir, "index.html")
    if os.path.exists(html_path):
        return FileResponse(html_path)
    else:
        return {"message": "ANIMA - The Emergent Self Engine", "status": "ready"}

@app.post("/api/simulation/create")
async def create_simulation(config: dict):
    """Create a new simulation with given configuration"""
    try:
        sim_config = SimulationConfig(
            world_size=tuple(config.get("world_size", [50, 50])),
            initial_agents=config.get("initial_agents", 20),
            resource_spawn_rate=config.get("resource_spawn_rate", 0.1),
            time_per_tick=config.get("time_per_tick", 0.1),
            use_llm=config.get("use_llm", True),
            llm_awakening_age=config.get("llm_awakening_age", 100),
            llm_awakening_wisdom=config.get("llm_awakening_wisdom", 5),
            batch_size=config.get("batch_size", 4)
        )

        await sim_manager.initialize_simulation(sim_config)

        return {
            "status": "success",
            "message": "Simulation created",
            "config": {
                "world_size": sim_config.world_size,
                "initial_agents": sim_config.initial_agents,
                "use_llm": sim_config.use_llm
            }
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/simulation/start")
async def start_simulation():
    """Start the simulation"""
    if not sim_manager.world:
        raise HTTPException(status_code=400, detail="No simulation created")

    await sim_manager.start_simulation()
    return {"status": "success", "message": "Simulation started"}

@app.post("/api/simulation/pause")
async def pause_simulation():
    """Pause the simulation"""
    await sim_manager.pause_simulation()
    return {"status": "success", "message": "Simulation paused"}

@app.post("/api/simulation/resume")
async def resume_simulation():
    """Resume the simulation"""
    await sim_manager.resume_simulation()
    return {"status": "success", "message": "Simulation resumed"}

@app.post("/api/simulation/stop")
async def stop_simulation():
    """Stop the simulation"""
    await sim_manager.stop_simulation()
    return {"status": "success", "message": "Simulation stopped"}

@app.post("/api/simulation/speed")
async def set_simulation_speed(speed: dict):
    """Set simulation speed multiplier"""
    multiplier = speed.get("multiplier", 1.0)
    sim_manager.speed_multiplier = max(0.1, min(10.0, multiplier))
    return {"status": "success", "speed": sim_manager.speed_multiplier}

@app.get("/api/simulation/state")
async def get_simulation_state():
    """Get current simulation state"""
    if not sim_manager.world:
        raise HTTPException(status_code=400, detail="No simulation running")

    state = sim_manager.world.get_world_state()
    state["model_status"] = {
        "loading": sim_manager.model_loading,
        "loaded": sim_manager.model_loaded,
        "error": sim_manager.model_error
    }
    return state

@app.get("/api/experiments")
async def get_experiments():
    """Get predefined experiment configurations"""
    experiments = {
        "genesis": {
            "name": "Genesis - The First Awakening",
            "description": "Standard initial conditions",
            "config": {
                "world_size": [50, 50],
                "initial_agents": 20,
                "resource_spawn_rate": 0.1,
                "use_llm": True
            }
        },
        "scarcity": {
            "name": "Scarcity - The Hungry World",
            "description": "Limited resources test survival",
            "config": {
                "world_size": [40, 40],
                "initial_agents": 30,
                "resource_spawn_rate": 0.03,
                "use_llm": True
            }
        },
        "abundance": {
            "name": "Abundance - The Garden Paradise",
            "description": "Plentiful resources enable creativity",
            "config": {
                "world_size": [60, 60],
                "initial_agents": 40,
                "resource_spawn_rate": 0.2,
                "use_llm": True
            }
        },
        "enlightenment": {
            "name": "Enlightenment - The Awakening",
            "description": "Accelerated consciousness evolution",
            "config": {
                "world_size": [50, 50],
                "initial_agents": 25,
                "resource_spawn_rate": 0.1,
                "llm_awakening_age": 50,
                "llm_awakening_wisdom": 3,
                "use_llm": True
            }
        },
        "babel": {
            "name": "Babel - The Tower of Tongues",
            "description": "Focus on language evolution",
            "config": {
                "world_size": [50, 50],
                "initial_agents": 50,
                "resource_spawn_rate": 0.1,
                "language_mutation_rate": 0.15,
                "use_llm": True
            }
        }
    }
    return experiments

@app.get("/api/agents/{agent_id}")
async def get_agent_details(agent_id: str):
    """Get detailed information about a specific agent"""
    if not sim_manager.world:
        raise HTTPException(status_code=400, detail="No simulation running")

    if agent_id not in sim_manager.world.agents:
        raise HTTPException(status_code=404, detail="Agent not found")

    agent = sim_manager.world.agents[agent_id]

    # Get detailed info
    details = agent.to_dict()
    details.update({
        "memory_recent": [str(m) for m in agent.memory.get_recent(10)],
        "creative_works_details": [
            {
                "type": w.work_type,
                "created_at": w.created_at,
                "appreciation": w.appreciation_count,
                "content_preview": str(w.content)[:200]
            }
            for w in agent.creative_works[-5:]
        ],
        "full_beliefs": dict(agent.beliefs.beliefs),
        "myths": agent.beliefs.myths[-3:],
        "personality_traits": agent._describe_personality()
    })

    return details

# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket connection for real-time updates"""
    await websocket.accept()
    sim_manager.add_client(websocket)

    try:
        # Send initial state if simulation exists
        if sim_manager.world:
            try:
                state = sim_manager.world.get_world_state()
                state["model_status"] = {
                    "loading": sim_manager.model_loading,
                    "loaded": sim_manager.model_loaded,
                    "error": sim_manager.model_error
                }

                clean_state = json.loads(json.dumps(state, default=safe_json_serializer))

                message = {
                    "type": "state",
                    "data": clean_state,
                    "timestamp": time.time()
                }

                await websocket.send_text(json.dumps(message, default=safe_json_serializer))

            except Exception as e:
                logger.error(f"Error sending initial state: {e}")
                # Send minimal state on error
                minimal_state = {
                    "time": sim_manager.world.time if sim_manager.world else 0,
                    "agents": [],
                    "resources": [],
                    "cultures": {},
                    "languages": {},
                    "myths": [],
                    "events": [],
                    "model_status": {
                        "loading": sim_manager.model_loading,
                        "loaded": sim_manager.model_loaded,
                        "error": sim_manager.model_error
                    }
                }

                await websocket.send_text(json.dumps({
                    "type": "state",
                    "data": minimal_state,
                    "timestamp": time.time()
                }))

        # Keep connection alive
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                if data == "ping":
                    await websocket.send_text("pong")
            except asyncio.TimeoutError:
                # Send ping if no message received
                await websocket.send_text("ping")

    except WebSocketDisconnect:
        sim_manager.remove_client(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        sim_manager.remove_client(websocket)

# Health check
@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "simulation_running": sim_manager.is_running,
        "connected_clients": len(sim_manager.connected_clients),
        "current_time": sim_manager.world.time if sim_manager.world else 0,
        "model_status": {
            "loading": sim_manager.model_loading,
            "loaded": sim_manager.model_loaded,
            "error": sim_manager.model_error
        }
    }

# Startup event - SIMPLIFIED
@app.on_event("startup")
async def startup_event():
    """Initialize on startup - without heavy model loading"""
    logger.info("🚀 ANIMA Server starting up...")

    # Create default simulation WITHOUT LLM for fast startup
    default_config = SimulationConfig(
        world_size=(50, 50),
        initial_agents=20,
        resource_spawn_rate=0.1,
        use_llm=True  # Will load in background
    )

    await sim_manager.initialize_simulation(default_config)
    logger.info("✅ Server ready! LLM loading in background...")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("ANIMA Server shutting down...")
    if sim_manager.is_running:
        await sim_manager.stop_simulation()

# Run the server
if __name__ == "__main__":
    # Create static directory if it doesn't exist
    os.makedirs("static", exist_ok=True)

    # Check for index.html
    if not os.path.exists("static/index.html"):
        logger.warning("⚠️  static/index.html not found! Web interface may not work.")

    logger.info("🌟 Starting ANIMA Server...")
    logger.info("📱 Open http://localhost:8000 in your browser")
    logger.info("🧠 DeepSeek model will load in background after startup")

    # Run with uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=False  # Disable reload for stability
    )