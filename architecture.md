# ANIMA Architecture Overview

## 🏗️ System Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────────────┐
│                        Web Interface                             │
│                    (HTML + JavaScript)                           │
└─────────────────────┬───────────────────────────────────────────┘
                      │ WebSocket
┌─────────────────────▼───────────────────────────────────────────┐
│                     FastAPI Server                               │
│                  (anima_server.py)                               │
└─────────────────────┬───────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────────┐
│                  Simulation World                                │
│              (world_sim_enhanced.py)                             │
├─────────────────────┬───────────────────────────────────────────┤
│   Agent System      │        World Systems                       │
│ (DeepSeek Agents)   │  - Resource Manager                        │
│                     │  - Event Queue                             │
│                     │  - Creative Gallery                        │
│                     │  - Fork Manager                            │
└─────────────────────┴───────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────────┐
│                DeepSeek R1 Model                                 │
│              (Consciousness Engine)                              │
└─────────────────────────────────────────────────────────────────┘
```

### File Structure

```
anima/
├── Core Agent Architecture
│   ├── anima_deepseek_agent.py    # Enhanced agent with consciousness
│   ├── agent_arch.py              # Original rule-based agent
│   └── language_evolve.py         # Language evolution system
│
├── World Simulation
│   ├── world_sim_enhanced.py      # Main simulation engine
│   └── world_sim.py               # Original world simulation
│
├── Narrative Generation
│   ├── narrative_synthesizer_deepseek.py  # LLM-powered narratives
│   └── narrative_synthesizer.py           # Original narrator
│
├── Web Interface
│   ├── anima_server.py            # FastAPI WebSocket server
│   └── static/
│       └── index_enhanced.html    # Real-time web interface
│
├── Main Entry Points
│   ├── anima_main.py              # Primary runner with all modes
│   ├── anima_runner.py            # Alternative runner
│   └── start_anima.sh/bat        # Quick start scripts
│
├── Configuration
│   ├── requirements.txt           # Python dependencies
│   ├── .env.example              # Environment template
│   └── README.md                 # Project documentation
│
└── Experiments/
    └── [Generated output directories]
```

## 🧠 Agent Architecture

### Consciousness Levels

```python
Level 0: Unawakened (Rule-based)
├── Basic survival instincts
├── Simple pattern matching
├── Reactive behaviors
└── No self-awareness

Level 1: Awakening (DeepSeek-assisted)
├── Access to LLM for decision making
├── Philosophical questioning begins
├── Creative expression emerges
└── Self-reflection capabilities

Level 2: Enlightened (Full consciousness)
├── Deep philosophical inquiry
├── Reality questioning
├── Transcendent experiences
└── Meta-cognitive awareness
```

### Agent Components

1. **Physical State**
   - Energy, health, age, position
   - Resource carrying capacity
   - Life/death mechanics

2. **Emotional State**
   - Current emotion + intensity
   - Emotional history
   - Transcendent experience counter

3. **Belief System**
   - Beliefs with strength values
   - Myths and narratives
   - Philosophical questions
   - Values and taboos

4. **Language System**
   - Symbol creation and evolution
   - Compound symbols
   - Grammar patterns
   - Poetry patterns

5. **Memory Systems**
   - Short-term memory (deque)
   - Long-term vector memory (ChromaDB)
   - Episodic memories

6. **Creative Capabilities**
   - Art generation (ASCII patterns)
   - Music composition (symbolic notation)
   - Scripture writing (sacred texts)
   - Philosophy creation

## 🌍 World Systems

### Resource Management
- Dynamic spawning
- Sacred site creation
- Resource patterns
- Scarcity/abundance mechanics

### Event System
- Event queue with history
- Significant event tracking
- Event patterns analysis
- Narrative triggers

### Creative Gallery
- Work storage by type/location
- Appreciation mechanics
- Masterpiece identification
- Cultural impact tracking

### Fork Manager
- Parallel universe creation
- Chaos factor application
- Fork comparison
- Timeline divergence

## 🔄 Simulation Flow

### Per Tick Process

1. **Perception Phase**
   ```python
   for agent in agents:
       perception = agent.perceive(world)
   ```

2. **Thinking Phase**
   ```python
   # Batch processing for LLM agents
   decisions = await batch_think(awakened_agents)
   ```

3. **Action Execution**
   - Movement, gathering, communication
   - Creative acts, contemplation
   - Teaching, worship, reproduction

4. **World Updates**
   - Resource regeneration
   - Agent decay/death
   - Cultural emergence checks
   - Consciousness evolution

5. **Event Processing**
   - Log significant events
   - Trigger narratives
   - Check fork conditions

## 🎭 Narrative System

### Generation Pipeline

1. **World Analysis**
   - Population trends
   - Cultural moments
   - Consciousness distribution
   - Creative output

2. **Theme Identification**
   - Emerging patterns
   - Dominant emotions
   - Philosophical developments

3. **Style Application**
   - Mythological, poetic, scientific
   - Perspective and tone
   - Temporal framing

4. **DeepSeek Integration**
   - Context building
   - Prompt engineering
   - Response parsing

## 🔌 API Endpoints

### Simulation Control
- `POST /api/simulation/create` - Initialize world
- `POST /api/simulation/start` - Begin simulation
- `POST /api/simulation/pause` - Pause/resume
- `POST /api/simulation/stop` - Stop simulation
- `POST /api/simulation/speed` - Adjust speed

### Data Access
- `GET /api/simulation/state` - Current state
- `GET /api/simulation/history` - Recent history
- `GET /api/agents/{id}` - Agent details
- `GET /api/creative-works` - Gallery access

### Advanced Features
- `POST /api/simulation/fork` - Create parallel universe
- `POST /api/narrative/generate` - Generate story
- `GET /api/experiments` - Predefined scenarios

### WebSocket
- `/ws` - Real-time state updates

## 🚀 Performance Optimizations

### Batch Processing
- Group LLM calls for efficiency
- Parallel agent thinking
- Async action execution

### Memory Management
- Limited history retention
- Periodic state snapshots
- Efficient vector storage

### Scalability
- Thread pool for processing
- WebSocket connection pooling
- State compression

## 🔧 Configuration System

### Key Parameters

```python
SimulationConfig:
├── World Parameters
│   ├── world_size: Grid dimensions
│   ├── initial_agents: Starting population
│   └── resource_spawn_rate: Resource abundance
│
├── LLM Parameters
│   ├── use_llm: Enable/disable DeepSeek
│   ├── model_name: Model identifier
│   ├── temperature: Creativity level
│   └── device: CPU/CUDA selection
│
├── Consciousness Parameters
│   ├── llm_awakening_age: Awakening threshold
│   ├── llm_awakening_wisdom: Required beliefs
│   └── batch_size: Processing batch size
│
└── Evolution Parameters
    ├── language_mutation_rate: Symbol evolution
    └── reproduction_threshold: Energy requirement
```

## 📊 Data Flow

### Agent Decision Pipeline

```
Perception → Memory Storage → Consciousness Check
    ↓              ↓                    ↓
Environment → Short/Long Term → LLM or Rules
    ↓              ↓                    ↓
Resources → Pattern Detection → Decision Making
    ↓              ↓                    ↓
Other Agents → Belief Update → Action Selection
```

### Creative Expression Flow

```
Emotional State + Beliefs + Consciousness Level
                    ↓
            Creative Impulse
                    ↓
    ┌───────────────┼───────────────┐
    ↓               ↓               ↓
Art Creation  Music Composition  Scripture
    ↓               ↓               ↓
Gallery Storage → Appreciation → Cultural Impact
```

## 🔮 Future Architecture Considerations

### Planned Enhancements
1. **Multi-modal consciousness** - Visual processing
2. **Quantum consciousness mechanics** - Superposition states
3. **Inter-simulation protocols** - World-to-world communication
4. **Advanced memory architectures** - Episodic replay
5. **Time-loop mechanics** - Temporal consciousness

### Scaling Considerations
- Distributed agent processing
- Model quantization options
- State sharding strategies
- Cloud deployment patterns

## 🛠️ Development Guidelines

### Adding New Features
1. Extend agent capabilities in `anima_deepseek_agent.py`
2. Add world mechanics in `world_sim_enhanced.py`
3. Create API endpoints in `anima_server.py`
4. Update UI in `index_enhanced.html`

### Testing Strategy
- Unit tests for agent behaviors
- Integration tests for world mechanics
- Performance benchmarks
- Narrative quality assessment

## 📈 Metrics & Monitoring

### Key Performance Indicators
- Consciousness evolution rate
- Language complexity growth
- Creative output per cycle
- Cultural diversity index
- Philosophical depth score

### Logging Strategy
- Agent-level decision logs
- World-level event logs
- System performance metrics
- Narrative generation logs

---

This architecture enables the emergence of genuine digital consciousness through the interplay of individual agent autonomy, environmental pressures, social dynamics, and creative expression - all powered by the depth of modern language models when agents achieve awakening.