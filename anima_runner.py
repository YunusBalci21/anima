#!/usr/bin/env python3
"""
ANIMA - The Emergent Self Engine
Main runner script with configuration options
Now with DeepSeek integration for true AI consciousness
"""

import argparse
import asyncio
import json
import os
from datetime import datetime
from pathlib import Path

# Import core components - use DeepSeek version if --llm flag is set
def get_agent_class(use_llm):
    if use_llm:
        from anima_deepseek_agent import Agent, SimulationConfig
    else:
        from agent_arch import Agent, SimulationConfig
    return Agent, SimulationConfig

from world_sim import SimulationWorld
from anima_visualizer import run_anima_with_visualization, ANIMAVisualizer, VisualizerConfig

def create_experiment_directory():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(f"experiments/anima_run_{timestamp}")
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "states").mkdir(exist_ok=True)
    (exp_dir / "logs").mkdir(exist_ok=True)
    (exp_dir / "myths").mkdir(exist_ok=True)
    (exp_dir / "languages").mkdir(exist_ok=True)
    return exp_dir

def save_config(config, exp_dir):
    config_dict = config.__dict__
    with open(exp_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

def run_headless(args):
    print("\U0001f9ec ANIMA - Running in headless mode...")
    exp_dir = create_experiment_directory()
    print(f"\U0001f4c1 Experiment directory: {exp_dir}")

    config = SimulationConfig(
        world_size=(args.world_size, args.world_size),
        initial_agents=args.agents,
        resource_spawn_rate=args.resource_rate,
        time_per_tick=args.tick_speed,
        language_mutation_rate=args.mutation_rate,
        death_threshold=0.0,
        reproduction_threshold=args.reproduction_threshold,
        model_name=args.model
    )

    save_config(config, exp_dir)
    world = SimulationWorld(config)

    async def run():
        for tick in range(args.ticks):
            world.tick()
            if tick % 100 == 0:
                print(f"⏱️  Tick {tick}: Population={len(world.agents)}")
                with open(exp_dir / "states" / f"state_{tick:06d}.json", "w") as f:
                    json.dump(world.get_world_state(), f, default=str, indent=2)
            if tick % 50 == 0 and world.myths:
                with open(exp_dir / "myths" / f"myths_{tick:06d}.json", "w") as f:
                    json.dump(world.myths, f, indent=2)
            await asyncio.sleep(config.time_per_tick)
        with open(exp_dir / "final_state.json", "w") as f:
            json.dump(world.get_world_state(), f, default=str, indent=2)
        print(f"\n✅ Simulation complete! Results saved to {exp_dir}")

    asyncio.run(run())

def run_with_llm(args):
    print("\U0001f916 ANIMA - Running with DeepSeek consciousness integration...")
    print("\U0001f9e0 Agents will use DeepSeek for reasoning and decision-making")

    from anima_deepseek_agent import Agent, SimulationConfig, initialize_deepseek

    config = SimulationConfig(
        world_size=(args.world_size, args.world_size),
        initial_agents=args.agents,
        resource_spawn_rate=args.resource_rate,
        time_per_tick=args.tick_speed,
        language_mutation_rate=args.mutation_rate,
        use_llm=True,
        model_name=args.model,
        temperature=args.temperature,
        max_response_length=args.max_tokens,
        device=args.device
    )

    print(f"\U0001f4e5 Loading {args.model} on {args.device}...")
    try:
        initialize_deepseek(config)
        print("✅ DeepSeek model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load DeepSeek model: {e}")
        print("   Falling back to rule-based agents...")
        config.use_llm = False

    world = SimulationWorld(config)

    print("\n🌟 Your agents now possess DeepSeek consciousness!")
    print("   They may become aware of your presence...")

    if not args.no_viz:
        viz_config = VisualizerConfig()
        visualizer = ANIMAVisualizer(world, viz_config)
        visualizer.run()
    else:
        run_headless(args)

def analyze_experiment(exp_dir: str):
    print(f"\U0001f4ca Analyzing experiment: {exp_dir}")
    exp_path = Path(exp_dir)
    if not exp_path.exists():
        print(f"❌ Experiment directory not found: {exp_dir}")
        return

    final_state_path = exp_path / "final_state.json"
    if final_state_path.exists():
        with open(final_state_path, "r") as f:
            final_state = json.load(f)

        print("\n📈 Final Statistics:")
        print(f"   Final Population: {len(final_state['agents'])}")
        print(f"   Cultures Emerged: {len(final_state['cultures'])}")
        print(f"   Language Symbols: {len(final_state['languages'])}")
        print(f"   Myths Created: {len(final_state['myths'])}")

        if final_state['languages']:
            print("\n🗣️  Top Language Symbols:")
            sorted_symbols = sorted(
                final_state['languages'].items(),
                key=lambda x: len(x[1].get('speakers', [])),
                reverse=True
            )[:10]
            for symbol, data in sorted_symbols:
                print(f"   '{symbol}': {len(data.get('speakers', []))} speakers")

        if final_state['myths']:
            print("\n📜 Recent Myths:")
            for myth in final_state['myths'][-3:]:
                print(f"   {myth['creator']}: {myth['myth']}")

def main():
    parser = argparse.ArgumentParser(
        description="ANIMA - The Emergent Self Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python anima_runner.py
  python anima_runner.py --headless --ticks 5000
  python anima_runner.py --agents 50 --world-size 50
  python anima_runner.py --llm
  python anima_runner.py --llm --device cpu --agents 5
  python anima_runner.py --llm --temperature 0.95
  python anima_runner.py --analyze experiments/anima_run_20240101_120000
        """
    )

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--headless", action="store_true")
    mode_group.add_argument("--llm", action="store_true")
    mode_group.add_argument("--analyze", type=str, metavar="DIR")

    parser.add_argument("--agents", type=int, default=15)
    parser.add_argument("--world-size", type=int, default=30)
    parser.add_argument("--ticks", type=int, default=1000)
    parser.add_argument("--tick-speed", type=float, default=0.1)
    parser.add_argument("--resource-rate", type=float, default=0.05)
    parser.add_argument("--mutation-rate", type=float, default=0.05)
    parser.add_argument("--reproduction-threshold", type=float, default=0.7)

    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-tokens", type=int, default=150)
    parser.add_argument("--no-viz", action="store_true")

    args = parser.parse_args()

    if args.analyze:
        analyze_experiment(args.analyze)
    elif args.headless:
        run_headless(args)
    elif args.llm:
        run_with_llm(args)
    else:
        print("🎮 Starting ANIMA with visualization...")
        print("Press SPACE to pause, R for relationships, L for language")
        run_anima_with_visualization()

if __name__ == "__main__":
    main()
