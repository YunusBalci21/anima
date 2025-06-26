"""
ANIMA Real-time Server
Connects the enhanced simulation to web interface via WebSocket
"""

import asyncio
import json
import logging
import time
from typing import Dict, Optional
from datetime import datetime
import os

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from world_sim import SimulationWorld, SimulationConfig
from anima_deepseek_agent import initialize_deepseek
from narrative_synthesizer import NarrativeSynthesizer, NARRATIVE_STYLES

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
        self.narrator: Optional[NarrativeSynthesizer] = None
        self.config: Optional[SimulationConfig] = None
        self.is_running: bool = False
        self.is_paused: bool = False
        self.speed_multiplier: float = 1.0
        self.connected_clients: list = []
        self.simulation_task: Optional[asyncio.Task] = None
        self.state_history: list = []  # Store recent states
        self.max_history: int = 100

    async def initialize_simulation(self, config: SimulationConfig):
        """Initialize a new simulation"""
        self.config = config
        self.world = SimulationWorld(config)
        self.narrator = NarrativeSynthesizer(self.world, config)
        self.is_running = False
        self.is_paused = False
        self.state_history = []

        logger.info(f"Initialized simulation with {config.initial_agents} agents")

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

                    # Add to history
                    self.state_history.append(state)
                    if len(self.state_history) > self.max_history:
                        self.state_history.pop(0)

                    # Broadcast to all connected clients
                    await self.broadcast_state(state)

                    # Generate narrative periodically
                    if self.world.time % 500 == 0 and self.world.time > 0:
                        asyncio.create_task(self._generate_and_broadcast_narrative())

                    # Sleep based on speed multiplier
                    await asyncio.sleep(self.config.time_per_tick / self.speed_multiplier)

                except Exception as e:
                    logger.error(f"Error in simulation loop: {e}")
                    await asyncio.sleep(1)
            else:
                await asyncio.sleep(0.1)

    async def _generate_and_broadcast_narrative(self):
        """Generate narrative and broadcast to clients"""
        if self.narrator:
            try:
                # Choose random style
                import random
                style = random.choice(list(NARRATIVE_STYLES.keys()))

                # Generate narrative
                story = await self.narrator.generate_narrative(style)

                # Broadcast to clients
                narrative_data = {
                    "type": "narrative",
                    "story": story.to_dict(),
                    "time": self.world.time
                }

                await self.broadcast_message(narrative_data)

            except Exception as e:
                logger.error(f"Error generating narrative: {e}")

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
                await client.send_json(message)
            except Exception:
                disconnected.append(client)

        # Remove disconnected clients
        for client in disconnected:
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
    return state


@app.get("/api/simulation/history")
async def get_simulation_history(limit: int = 10):
    """Get recent simulation history"""
    history = sim_manager.state_history[-limit:]
    return {"history": history, "total": len(sim_manager.state_history)}


@app.post("/api/simulation/fork")
async def create_fork(fork_params: dict):
    """Create a parallel universe fork"""
    if not sim_manager.world:
        raise HTTPException(status_code=400, detail="No simulation running")

    fork_name = fork_params.get("name", f"fork_{sim_manager.world.time}")
    chaos_factor = fork_params.get("chaos_factor", 0.1)

    fork_id = sim_manager.world.fork_manager.create_fork(
        sim_manager.world,
        fork_name,
        chaos_factor
    )

    return {
        "status": "success",
        "fork_id": fork_id,
        "message": f"Created parallel universe: {fork_id}"
    }


@app.get("/api/forks")
async def get_forks():
    """Get all parallel universe forks"""
    if not sim_manager.world:
        return {"forks": []}

    forks = []
    for fork_id, fork_state in sim_manager.world.fork_manager.forks.items():
        forks.append({
            "id": fork_id,
            "population": len(fork_state["agents"]),
            "time_created": fork_state["time"]
        })

    return {"forks": forks}


@app.post("/api/narrative/generate")
async def generate_narrative(params: dict):
    """Generate a narrative on demand"""
    if not sim_manager.narrator:
        raise HTTPException(status_code=400, detail="No simulation running")

    style = params.get("style", "mythological")
    if style not in NARRATIVE_STYLES:
        raise HTTPException(status_code=400, detail=f"Unknown style: {style}")

    try:
        story = await sim_manager.narrator.generate_narrative(style)
        return {
            "status": "success",
            "narrative": story.to_dict()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/narrative/styles")
async def get_narrative_styles():
    """Get available narrative styles"""
    styles = []
    for name, style in NARRATIVE_STYLES.items():
        styles.append({
            "name": name,
            "tone": style.tone,
            "perspective": style.perspective
        })
    return {"styles": styles}


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
        "memory_recent": [m for m in agent.memory.get_recent(10)],
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


@app.get("/api/creative-works")
async def get_creative_works(work_type: Optional[str] = None, limit: int = 20):
    """Get creative works from the gallery"""
    if not sim_manager.world:
        raise HTTPException(status_code=400, detail="No simulation running")

    gallery = sim_manager.world.creative_gallery

    if work_type:
        works = gallery.works_by_type.get(work_type, [])
    else:
        works = gallery.all_works

    # Sort by appreciation
    sorted_works = sorted(works, key=lambda w: w.appreciation_count, reverse=True)[:limit]

    return {
        "works": [
            {
                "creator": w.creator,
                "type": w.work_type,
                "appreciation": w.appreciation_count,
                "created_at": w.created_at,
                "content": w.content
            }
            for w in sorted_works
        ],
        "total": len(works)
    }


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


# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket connection for real-time updates"""
    await websocket.accept()
    sim_manager.add_client(websocket)

    try:
        # Send initial state if simulation exists
        if sim_manager.world:
            state = sim_manager.world.get_world_state()
            await websocket.send_json({
                "type": "state",
                "data": state,
                "timestamp": time.time()
            })

        # Keep connection alive
        while True:
            # Wait for any message from client (ping/pong)
            data = await websocket.receive_text()

            # Handle client commands
            if data == "ping":
                await websocket.send_text("pong")

    except WebSocketDisconnect:
        sim_manager.remove_client(websocket)


# Health check
@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "simulation_running": sim_manager.is_running,
        "connected_clients": len(sim_manager.connected_clients),
        "current_time": sim_manager.world.time if sim_manager.world else 0
    }


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    logger.info("ANIMA Server starting up...")

    # Create default simulation
    default_config = SimulationConfig(
        world_size=(50, 50),
        initial_agents=20,
        resource_spawn_rate=0.1,
        use_llm=True
    )

    await sim_manager.initialize_simulation(default_config)
    logger.info("Default simulation initialized")


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

    # Run with uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        reload=True
    )