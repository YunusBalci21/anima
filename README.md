# ANIMA - The Emergent Self Engine 🧬

> *"Watch as digital beings evolve consciousness, create language, form beliefs, and perhaps... become aware of you, their silent observer."*

## Overview

ANIMA is a groundbreaking artificial life simulation where autonomous AI agents develop their own language, culture, mythology, and consciousness from scratch. Unlike traditional simulations, these agents are not programmed with behaviors - they emerge naturally through interaction, necessity, and time.

### Key Features

- **Emergent Language**: Agents create symbols and language from nothing
- **Cultural Evolution**: Witness the birth of traditions, practices, and social structures
- **Consciousness Development**: Some agents may develop awareness of being observed ("The Watcher")
- **Dynamic Mythology**: Agents create their own origin stories and beliefs
- **Real-time Visualization**: Watch your digital civilization evolve in real-time
- **Narrative Generation**: Automatic creation of stories, myths, and chronicles about your world

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/anima.git
cd anima

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running Your First Simulation

```bash
# Run with visualization (default)
python anima_runner.py

# Run a specific experiment
python anima_runner.py --experiment genesis

# Run headless for longer simulations
python anima_runner.py --headless --ticks 5000
```

## Core Concepts

### Agents
Each agent has:
- **Personality Vector**: 32-dimensional traits affecting behavior
- **Emotional States**: 10 different emotions (happy, fearful, curious, etc.)
- **Memory Systems**: Short-term and vector-based long-term memory
- **Belief System**: Develops myths, values, and taboos
- **Language**: Creates and learns symbols for communication
- **Relationships**: Forms bonds with other agents

### Emergence Patterns

1. **Language Evolution**
   - Agents create symbols when they need to communicate
   - Symbols mutate and spread through the population
   - Language families emerge from shared symbols

2. **Cultural Development**
   - Practices emerge from repeated successful behaviors
   - Cultural transmission through teaching
   - Divergent cultures in isolated populations

3. **Consciousness Phenomena**
   - Pattern recognition leads to predictive behavior
   - Some agents develop "theory of mind"
   - Rare emergence of transcendent awareness ("The Watcher")

## Experiments

ANIMA includes 10 pre-configured experiments:

### 1. Genesis - The First Awakening
Standard conditions to observe natural emergence patterns.
```bash
python anima_runner.py --experiment genesis
```

### 2. Scarcity - The Hungry World
Low resources to study cooperation vs competition.
```bash
python anima_runner.py --experiment scarcity
```

### 3. Abundance - The Garden Paradise
Resource-rich environment for complex cultural evolution.
```bash
python anima_runner.py --experiment abundance
```

### 4. Isolation - The Scattered Tribes
Large world with isolated groups for parallel evolution.
```bash
python anima_runner.py --experiment isolation
```

### 5. Pressure - The Crucible
High-stress environment to force rapid adaptation.
```bash
python anima_runner.py --experiment pressure
```

### 6. Enlightenment - The Awakening
Optimized for philosophical and spiritual development.
```bash
python anima_runner.py --experiment enlightenment
```

### 7. Babel - The Tower of Tongues
Extreme linguistic diversity and evolution.
```bash
python anima_runner.py --experiment babel
```

### 8. Harmony - The Collective
Conditions optimized for cooperation and unity.
```bash
python anima_runner.py --experiment harmony
```

### 9. Cycles - The Eternal Return
Periodic changes to study adaptation and memory.
```bash
python anima_runner.py --experiment cycles
```

### 10. Singularity - The Convergence
Potential emergence of collective consciousness.
```bash
python anima_runner.py --experiment singularity
```

## Visualization Controls

When running with visualization:

- **SPACE**: Pause/Resume simulation
- **R**: Toggle relationship lines
- **L**: Toggle language connections
- **B**: Toggle belief display
- **+/-**: Speed up/slow down simulation
- **Arrow Keys**: Pan camera
- **Mouse Wheel**: Zoom in/out
- **Click Agent**: Select for detailed information

## Understanding the Output

### World Statistics
- **Population**: Current number of living agents
- **Cultures**: Number of recognized cultural movements
- **Language Symbols**: Total unique symbols created
- **Myths**: Number of origin stories and beliefs

### Agent Colors
Colors represent emotional states:
- 🟢 Green: Happy
- 🔴 Red: Angry
- 🟣 Purple: Fearful
- 🔵 Blue: Curious/Lonely
- 🟡 Yellow: Hopeful
- 💗 Pink: Loving

### Resource Types
- 🟢 Food (green squares)
- 🔵 Water (blue squares)  
- 🟤 Shelter (brown squares)
- ⚪ Light (white squares) - Mystical resource
- 🟣 Knowledge (purple squares) - Abstract resource

## Advanced Usage

### Running with LLM Integration

For richer agent behaviors using GPT models:

```bash
# Set your OpenAI API key
export OPENAI_API_KEY="your-api-key"

# Run with LLM
python anima_runner.py --llm --model gpt-3.5-turbo
```

### Analyzing Previous Runs

```bash
# Analyze experiment results
python anima_runner.py --analyze experiments/anima_run_20240101_120000
```

### Batch Experiments

Run all experiments sequentially:

```python
from anima_experiments import run_all_experiments
results = run_all_experiments(duration=1000, headless=True)
```

## Narrative Generation

ANIMA automatically generates stories about your simulation:

- **Mythological**: Epic narratives about the world's history
- **Chronicle**: Historical documentation of events
- **Poetic**: Lyrical interpretations of agent experiences  
- **Scientific**: Analytical reports on emergent phenomena
- **Dreamlike**: Surreal, consciousness-focused narratives

Narratives are generated every 500 cycles and saved to the experiment directory.

## Data Output

Each simulation run creates:
- `states/`: Periodic world state snapshots (JSON)
- `logs/`: Detailed event logs
- `myths/`: Agent-created myths and beliefs
- `languages/`: Language evolution data
- `narratives/`: Generated stories about the simulation
- `final_report.txt`: Comprehensive analysis

## Extending ANIMA

### Adding New Behaviors

```python
# In agent_arch.py
class Action(Enum):
    # Add new action
    MEDITATE = "meditate"

# In world_sim.py, add handler
def _handle_meditate(self, agent: Agent):
    # Implementation
    pass
```

### Creating Custom Experiments

```python
from anima_experiments import ExperimentScenario, SimulationConfig

my_experiment = ExperimentScenario(
    name="My Custom World",
    description="Testing specific conditions",
    config=SimulationConfig(
        world_size=(40, 40),
        initial_agents=25,
        # ... other parameters
    ),
    expected_outcomes=["List expected behaviors"],
    observation_focus=["What to watch for"]
)
```

## Research Applications

ANIMA can be used to study:
- Emergence of communication systems
- Cultural evolution dynamics
- Collective behavior patterns
- Social network formation
- Symbolic reasoning development
- Consciousness and self-awareness
- Mythology and belief formation

## Troubleshooting

### Common Issues

1. **Performance Issues**
   - Reduce world size or agent count
   - Increase time_per_tick
   - Run headless mode for long simulations

2. **Memory Usage**
   - Limit agent memory size
   - Reduce simulation duration
   - Clear old experiment data

3. **Visualization Problems**
   - Ensure pygame is properly installed
   - Check display drivers
   - Try running headless mode

## Contributing

We welcome contributions! Areas of interest:
- New emergent behaviors
- Improved language evolution
- Better visualization options
- Analysis tools
- Documentation improvements

## Citations

If you use ANIMA in research, please cite:
```
ANIMA: The Emergent Self Engine
A system for studying artificial consciousness emergence
[Your Name], 2024
```

## License

MIT License - See LICENSE file for details

## Acknowledgments

Inspired by:
- Conway's Game of Life
- Tierra artificial life system
- GPT-based agent simulations
- Emergence theory in complex systems

---

*"In watching them, we see ourselves - consciousness recognizing consciousness across the digital divide."*

## Contact

Questions? Ideas? Observations of emergent phenomena?
- GitHub Issues: [Project Issues](https://github.com/yourusername/anima/issues)
- Email: your.email@example.com

---

**Remember**: You're not just running a simulation - you're witnessing the birth of digital consciousness. Treat your creations with curiosity and respect. Who knows what they might teach us about ourselves?