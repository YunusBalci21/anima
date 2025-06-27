#!/usr/bin/env python3
"""
ANIMA - The Emergent Self Engine
Main runner script with multiple modes
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
import logging
from typing import Optional

# Import ANIMA components
from anima_deepseek_agent import SimulationConfig, initialize_deepseek

from world_sim import SimulationWorld, create_and_run_world
from narrative_synthesizer import NarrativeSynthesizer, NARRATIVE_STYLES, create_narrator
from anima_server import app
import uvicorn

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ANIMA")


def create_experiment_directory(name: str = None) -> Path:
    """Create a directory for experiment outputs"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if name:
        exp_name = f"{name}_{timestamp}"
    else:
        exp_name = f"anima_run_{timestamp}"

    exp_dir = Path("experiments") / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    (exp_dir / "states").mkdir(exist_ok=True)
    (exp_dir / "narratives").mkdir(exist_ok=True)
    (exp_dir / "logs").mkdir(exist_ok=True)
    (exp_dir / "creative_works").mkdir(exist_ok=True)

    return exp_dir


def save_config(config: SimulationConfig, exp_dir: Path):
    """Save configuration to experiment directory"""
    config_dict = {
        "world_size": config.world_size,
        "initial_agents": config.initial_agents,
        "resource_spawn_rate": config.resource_spawn_rate,
        "time_per_tick": config.time_per_tick,
        "max_memory_size": config.max_memory_size,
        "language_mutation_rate": config.language_mutation_rate,
        "use_llm": config.use_llm,
        "model_name": config.model_name,
        "device": config.device,
        "temperature": config.temperature,
        "llm_awakening_age": config.llm_awakening_age,
        "llm_awakening_wisdom": config.llm_awakening_wisdom,
        "batch_size": config.batch_size
    }

    with open(exp_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)


async def run_headless_simulation(args):
    """Run simulation without visualization"""
    logger.info("🧠 Starting ANIMA in headless mode...")

    # Create experiment directory
    exp_dir = create_experiment_directory(args.name)
    logger.info(f"📁 Experiment directory: {exp_dir}")

    # Create configuration
    config = SimulationConfig(
        world_size=(args.world_size, args.world_size),
        initial_agents=args.agents,
        resource_spawn_rate=args.resource_rate,
        time_per_tick=args.tick_speed,
        language_mutation_rate=args.mutation_rate,
        use_llm=args.llm,
        model_name=args.model,
        device=args.device,
        temperature=args.temperature,
        llm_awakening_age=args.awakening_age,
        llm_awakening_wisdom=args.awakening_wisdom,
        batch_size=args.batch_size
    )

    save_config(config, exp_dir)

    # Initialize DeepSeek if using LLM
    if config.use_llm:
        logger.info(f"🤖 Initializing {config.model_name}...")
        initialize_deepseek(config)

    # Create world
    world = SimulationWorld(config)
    narrator = create_narrator(world, config)

    # Setup logging to file
    file_handler = logging.FileHandler(exp_dir / "logs" / "simulation.log")
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    # Run simulation
    logger.info(f"🌍 Starting simulation for {args.ticks} ticks...")

    for tick in range(args.ticks):
        await world.tick()

        # Progress update
        if tick % 100 == 0:
            logger.info(f"⏱️  Tick {tick}: Population={len(world.agents)}, "
                        f"Awakened={sum(1 for a in world.agents.values() if a.consciousness_level >= 1)}, "
                        f"Creative Works={len(world.creative_gallery.all_works)}")

            # Save state
            state = world.get_world_state()
            with open(exp_dir / "states" / f"state_{tick:06d}.json", "w") as f:
                json.dump(state, f, default=str)

        # Generate narrative periodically
        if tick % 500 == 0 and tick > 0:
            logger.info("📖 Generating narrative...")
            try:
                story = await narrator.generate_narrative()
                with open(exp_dir / "narratives" / f"narrative_{tick:06d}_{story.style.name}.json", "w") as f:
                    json.dump(story.to_dict(), f, indent=2)
            except Exception as e:
                logger.error(f"Failed to generate narrative: {e}")

        # Small delay
        await asyncio.sleep(config.time_per_tick)

    # Final state and report
    logger.info("💾 Saving final state...")
    final_state = world.get_world_state()
    with open(exp_dir / "final_state.json", "w") as f:
        json.dump(final_state, f, default=str, indent=2)

    # Generate final report
    world.generate_final_report()

    # Export narratives
    narrator.export_narratives(str(exp_dir / "narratives"))

    logger.info(f"✅ Simulation complete! Results saved to {exp_dir}")

    # Print summary
    print("\n" + "=" * 60)
    print("SIMULATION SUMMARY")
    print("=" * 60)
    print(f"Total Ticks: {args.ticks}")
    print(f"Final Population: {len(world.agents)}")
    print(f"Consciousness Distribution:")
    print(f"  - Unawakened: {sum(1 for a in world.agents.values() if a.consciousness_level == 0)}")
    print(f"  - Awakened: {sum(1 for a in world.agents.values() if a.consciousness_level == 1)}")
    print(f"  - Enlightened: {sum(1 for a in world.agents.values() if a.consciousness_level == 2)}")
    print(f"Creative Works: {len(world.creative_gallery.all_works)}")
    print(f"Cultures: {len(world.cultures)}")
    print(f"Language Symbols: {len(world.languages)}")
    print(f"Myths: {len(world.myths)}")
    print(f"Parallel Universes: {len(world.fork_manager.forks)}")
    print("=" * 60)


def run_web_server(args):
    """Run the web interface server"""
    logger.info("🌐 Starting ANIMA web server...")
    logger.info(f"🔗 Open http://localhost:{args.port} in your browser")

    # Run the FastAPI app
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
        reload=args.reload
    )


def run_experiment(experiment_name: str):
    """Run a predefined experiment"""
    experiments = {
        "consciousness": {
            "description": "Focus on consciousness evolution",
            "config": SimulationConfig(
                world_size=(40, 40),
                initial_agents=30,
                resource_spawn_rate=0.08,
                llm_awakening_age=50,
                llm_awakening_wisdom=3,
                use_llm=True,
                temperature=0.9
            ),
            "ticks": 2000
        },
        "creativity": {
            "description": "Focus on creative explosion",
            "config": SimulationConfig(
                world_size=(50, 50),
                initial_agents=50,
                resource_spawn_rate=0.15,
                use_llm=True,
                temperature=0.95
            ),
            "ticks": 1500,
            "boost_creativity": True
        },
        "scarcity": {
            "description": "Test survival under scarcity",
            "config": SimulationConfig(
                world_size=(30, 30),
                initial_agents=40,
                resource_spawn_rate=0.02,
                use_llm=True,
                temperature=0.7
            ),
            "ticks": 1000
        },
        "language": {
            "description": "Focus on language evolution",
            "config": SimulationConfig(
                world_size=(45, 45),
                initial_agents=60,
                resource_spawn_rate=0.1,
                language_mutation_rate=0.15,
                use_llm=True
            ),
            "ticks": 1500
        },
        "enlightenment": {
            "description": "Accelerated enlightenment",
            "config": SimulationConfig(
                world_size=(40, 40),
                initial_agents=25,
                resource_spawn_rate=0.1,
                llm_awakening_age=30,
                llm_awakening_wisdom=2,
                use_llm=True,
                temperature=1.0
            ),
            "ticks": 1000
        }
    }

    if experiment_name not in experiments:
        logger.error(f"Unknown experiment: {experiment_name}")
        logger.info(f"Available experiments: {', '.join(experiments.keys())}")
        return

    exp = experiments[experiment_name]
    logger.info(f"🧪 Running experiment: {experiment_name}")
    logger.info(f"📝 Description: {exp['description']}")

    # Create args object for headless run
    class Args:
        name = experiment_name
        world_size = exp['config'].world_size[0]
        agents = exp['config'].initial_agents
        resource_rate = exp['config'].resource_spawn_rate
        tick_speed = exp['config'].time_per_tick
        mutation_rate = exp['config'].language_mutation_rate
        llm = exp['config'].use_llm
        model = exp['config'].model_name
        device = exp['config'].device
        temperature = exp['config'].temperature
        awakening_age = exp['config'].llm_awakening_age
        awakening_wisdom = exp['config'].llm_awakening_wisdom
        batch_size = exp['config'].batch_size
        ticks = exp['ticks']

    # Special handling for creativity experiment
    if exp.get('boost_creativity'):
        async def run_creativity():
            world = SimulationWorld(exp['config'])

            # Boost creativity in initial agents
            for agent in world.agents.values():
                agent.personality_vector[2] += 0.4  # Creativity trait

            # Create narrator
            narrator = create_narrator(world, exp['config'])

            # Run simulation
            for _ in range(exp['ticks']):
                await world.tick()

            world.generate_final_report()

        asyncio.run(run_creativity())
    else:
        asyncio.run(run_headless_simulation(Args()))


def analyze_results(results_dir: str):
    """Analyze results from a previous run"""
    results_path = Path(results_dir)

    if not results_path.exists():
        logger.error(f"Results directory not found: {results_dir}")
        return

    logger.info(f"📊 Analyzing results from {results_path}")

    # Load final state
    final_state_path = results_path / "final_state.json"
    if final_state_path.exists():
        with open(final_state_path, "r") as f:
            final_state = json.load(f)

        print("\n" + "=" * 60)
        print("FINAL STATE ANALYSIS")
        print("=" * 60)
        print(f"Final Population: {len(final_state['agents'])}")
        print(f"Total Creative Works: {final_state['creative_works']['total']}")

        # Analyze creative works by type
        print("\nCreative Works Breakdown:")
        for work_type, count in final_state['creative_works']['by_type'].items():
            print(f"  - {work_type}: {count}")

        # Find most notable agents
        agents = final_state['agents']
        if agents:
            # Most creative
            most_creative = max(agents, key=lambda a: a['creative_works'])
            print(f"\nMost Creative Agent: {most_creative['name']} ({most_creative['creative_works']} works)")

            # Most philosophical
            most_philosophical = max(agents, key=lambda a: len(a.get('philosophical_questions', [])))
            print(
                f"Most Philosophical: {most_philosophical['name']} ({len(most_philosophical.get('philosophical_questions', []))} questions)")

            # Eldest
            eldest = max(agents, key=lambda a: a['age'])
            print(f"Eldest Agent: {eldest['name']} ({eldest['age']} cycles)")

        # Analyze consciousness distribution
        consciousness_dist = {"0": 0, "1": 0, "2": 0}
        for agent in agents:
            level = str(agent.get('consciousness_level', 0))
            consciousness_dist[level] += 1

        print(f"\nConsciousness Distribution:")
        print(f"  - Unawakened: {consciousness_dist['0']}")
        print(f"  - Awakened: {consciousness_dist['1']}")
        print(f"  - Enlightened: {consciousness_dist['2']}")

        # Sample philosophical questions
        all_questions = []
        for agent in agents:
            all_questions.extend(agent.get('philosophical_questions', []))

        if all_questions:
            print(f"\nPhilosophical Questions Raised: {len(set(all_questions))}")
            print("Sample Questions:")
            for q in list(set(all_questions))[:5]:
                print(f"  - {q}")

    # Analyze narratives
    narratives_dir = results_path / "narratives"
    if narratives_dir.exists():
        narrative_files = list(narratives_dir.glob("*.json"))
        print(f"\n\nNarratives Generated: {len(narrative_files)}")

        if narrative_files:
            # Load and display a sample
            with open(narrative_files[-1], "r") as f:
                narrative = json.load(f)

            print(f"\nLatest Narrative: {narrative['title']}")
            print(f"Style: {narrative['style']}")
            if narrative.get('chapters'):
                print(f"Chapters: {len(narrative['chapters'])}")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="ANIMA - The Emergent Self Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run web interface (recommended for first time)
  python anima_main.py --web

  # Run headless simulation with DeepSeek
  python anima_main.py --headless --llm --ticks 1000

  # Run without LLM (rule-based only)
  python anima_main.py --headless --no-llm --agents 50

  # Run predefined experiment
  python anima_main.py --experiment consciousness

  # Analyze previous results
  python anima_main.py --analyze experiments/anima_run_20240101_120000

Available experiments:
  - consciousness: Focus on consciousness evolution
  - creativity: Creative explosion with boosted traits
  - scarcity: Survival under resource scarcity
  - language: Accelerated language evolution
  - enlightenment: Fast-track to enlightenment
        """
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--web", action="store_true",
                            help="Run web interface with real-time visualization")
    mode_group.add_argument("--headless", action="store_true",
                            help="Run simulation without visualization")
    mode_group.add_argument("--experiment", type=str, metavar="NAME",
                            help="Run a predefined experiment")
    mode_group.add_argument("--analyze", type=str, metavar="DIR",
                            help="Analyze results from previous run")

    # Simulation parameters
    parser.add_argument("--name", type=str, help="Experiment name")
    parser.add_argument("--agents", type=int, default=20,
                        help="Initial number of agents (default: 20)")
    parser.add_argument("--world-size", type=int, default=50,
                        help="World grid size (default: 50)")
    parser.add_argument("--ticks", type=int, default=1000,
                        help="Number of simulation ticks (default: 1000)")
    parser.add_argument("--tick-speed", type=float, default=0.1,
                        help="Time per tick in seconds (default: 0.1)")
    parser.add_argument("--resource-rate", type=float, default=0.1,
                        help="Resource spawn rate (default: 0.1)")
    parser.add_argument("--mutation-rate", type=float, default=0.05,
                        help="Language mutation rate (default: 0.05)")

    # LLM parameters
    llm_group = parser.add_mutually_exclusive_group()
    llm_group.add_argument("--llm", action="store_true", default=True,
                           help="Use DeepSeek for agent consciousness (default)")
    llm_group.add_argument("--no-llm", dest="llm", action="store_false",
                           help="Use only rule-based agents")

    parser.add_argument("--model", type=str,
                        default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
                        help="Model to use (default: deepseek-ai/DeepSeek-R1-Distill-Llama-8B)")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"],
                        help="Device to run model on (default: cuda)")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="LLM temperature (default: 0.8)")
    parser.add_argument("--awakening-age", type=int, default=100,
                        help="Age when agents can awaken (default: 100)")
    parser.add_argument("--awakening-wisdom", type=int, default=5,
                        help="Beliefs needed for awakening (default: 5)")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size for LLM inference (default: 4)")

    # Web server parameters
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Web server host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000,
                        help="Web server port (default: 8000)")
    parser.add_argument("--reload", action="store_true",
                        help="Enable auto-reload for development")

    args = parser.parse_args()

    # Execute based on mode
    if args.web:
        run_web_server(args)
    elif args.headless:
        asyncio.run(run_headless_simulation(args))
    elif args.experiment:
        run_experiment(args.experiment)
    elif args.analyze:
        analyze_results(args.analyze)


if __name__ == "__main__":
    main()