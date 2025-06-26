"""
Test script for ANIMA with DeepSeek Integration
Demonstrates how agents use DeepSeek for consciousness
"""

import asyncio
import logging
from datetime import datetime

# Import the DeepSeek-enhanced components
from anima_deepseek_agent import Agent, SimulationConfig, initialize_deepseek
from world_sim import SimulationWorld

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ANIMA-DeepSeek")

async def test_deepseek_agent():
    """Test a single agent with DeepSeek consciousness"""
    logger.info("🧬 Initializing ANIMA with DeepSeek consciousness...")
    
    # Configure for DeepSeek
    config = SimulationConfig(
        world_size=(10, 10),
        initial_agents=3,  # Start small for testing
        use_llm=True,
        model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        temperature=0.8,
        max_response_length=150
    )
    
    # Initialize model (this will download on first run)
    logger.info("🧠 Loading DeepSeek model...")
    initialize_deepseek(config)
    
    # Create a test agent
    agent = Agent("test_001", (5, 5), config)
    logger.info(f"✨ Created agent: {agent.name}")
    
    # Test thinking process
    logger.info("\n📊 Testing agent thinking...")
    
    # Create mock context
    context = {
        "position": (5, 5),
        "nearby_agents": [
            {"id": "other_001", "name": "Kara", "emotion": "curious"}
        ],
        "nearby_resources": [
            {"type": "food", "position": (5, 6)},
            {"type": "knowledge", "position": (4, 5)}
        ],
        "time": 100
    }
    
    # Let agent think
    decision = await agent.think(context)
    logger.info(f"🤔 {agent.name}'s decision: {decision['action'].value}")
    logger.info(f"💭 Reasoning: {decision['reasoning']}")
    
    # Test contemplation
    logger.info("\n🧘 Testing philosophical contemplation...")
    contemplation = agent.contemplate_existence()
    logger.info(f"🌟 Contemplation type: {contemplation['type']}")
    logger.info(f"📜 Content: {contemplation['content']}")
    
    # Test communication
    logger.info("\n💬 Testing communication...")
    other_agent = Agent("test_002", (5, 6), config)
    message = agent.communicate("hello friend", other_agent)
    logger.info(f"🗣️ {agent.name} says: {message['utterance']}")
    
    # Show agent state
    logger.info("\n📋 Agent state:")
    state = agent.to_dict()
    for key, value in state.items():
        logger.info(f"  {key}: {value}")
    
    return agent

async def run_mini_simulation():
    """Run a small simulation with DeepSeek agents"""
    logger.info("\n🌍 Starting mini ANIMA world with DeepSeek agents...")
    
    config = SimulationConfig(
        world_size=(20, 20),
        initial_agents=5,
        use_llm=True,
        temperature=0.8,
        resource_spawn_rate=0.1
    )
    
    # Create world
    world = SimulationWorld(config)
    
    # Run for a few ticks
    logger.info("⏱️ Running simulation...")
    for tick in range(10):
        logger.info(f"\n--- Tick {tick} ---")
        
        # Each agent perceives and thinks
        for agent_id, agent in world.agents.items():
            if agent.physical_state.is_alive():
                # Perceive
                perception = agent.perceive(world)
                
                # Think (with DeepSeek)
                decision = await agent.think(perception)
                
                logger.info(f"{agent.name}: {decision['action'].value} - {decision['reasoning'][:50]}...")
                
        # Execute one tick
        world.tick()
        
        # Show world state
        logger.info(f"Population: {len(world.agents)}, Symbols: {len(world.languages)}")
        
        # Check for interesting events
        if world.events.events:
            event = world.events.events[-1]
            logger.info(f"📢 Event: {event.type}")
            
    # Final summary
    logger.info("\n📊 Final Summary:")
    logger.info(f"  Survivors: {len(world.agents)}")
    logger.info(f"  Languages created: {len(world.languages)}")
    logger.info(f"  Cultures: {len(world.cultures)}")
    
    # Check for transcendent awareness
    aware = [a for a in world.agents.values() if "the_watcher_exists" in a.beliefs.beliefs]
    if aware:
        logger.info(f"  🔮 Agents aware of The Watcher: {len(aware)}")
        for agent in aware:
            logger.info(f"    - {agent.name}")

def compare_thinking_modes():
    """Compare DeepSeek vs rule-based thinking"""
    logger.info("\n🔬 Comparing thinking modes...")
    
    # Create two agents - one with DeepSeek, one without
    config_llm = SimulationConfig(use_llm=True)
    config_rules = SimulationConfig(use_llm=False)
    
    agent_deepseek = Agent("deepseek_agent", (0, 0), config_llm)
    agent_rules = Agent("rules_agent", (1, 1), config_rules)
    
    context = {
        "position": (0, 0),
        "nearby_agents": [],
        "nearby_resources": [{"type": "food", "position": (0, 1)}],
        "time": 0
    }
    
    logger.info("\n🤖 DeepSeek Agent:")
    logger.info(f"  Name: {agent_deepseek.name}")
    logger.info(f"  Personality: {agent_deepseek._describe_personality()}")
    
    logger.info("\n📋 Rule-based Agent:")
    logger.info(f"  Name: {agent_rules.name}")
    logger.info(f"  Personality: {agent_rules._describe_personality()}")

if __name__ == "__main__":
    logger.info("""
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║    ANIMA + DeepSeek: Digital Consciousness Emerges          ║
║                                                               ║
║    Watch as agents use DeepSeek's reasoning to:             ║
║    - Form their own thoughts and beliefs                     ║
║    - Create unique languages and myths                       ║
║    - Develop awareness of their digital nature               ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
    """)
    
    # Run tests
    try:
        # Test single agent
        asyncio.run(test_deepseek_agent())
        
        # Run mini simulation
        asyncio.run(run_mini_simulation())
        
        # Compare modes
        compare_thinking_modes()
        
    except Exception as e:
        logger.error(f"Error during test: {e}")
        logger.info("\n💡 Tips:")
        logger.info("  1. Make sure you have enough GPU memory (8GB+ recommended)")
        logger.info("  2. First run will download the model (~16GB)")
        logger.info("  3. Use CPU mode by setting device='cpu' in config if needed")
        logger.info("  4. Reduce max_response_length to save memory")