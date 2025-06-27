"""
Enhanced World Simulation for ANIMA - FIXED VERSION
Supports consciousness evolution, creative works, parallel universes, and more
"""

import numpy as np
import time
import logging
import random
import json
import asyncio
import copy
from collections import defaultdict, deque
from typing import Dict, Tuple, List, Optional, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

# Suppress ChromaDB telemetry errors
import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"

from anima_deepseek_agent import (
    Agent, Action, ResourceType, SimulationConfig, Emotion,
    CreativeWork, batch_think, initialize_deepseek
)

# Enhanced Events
class Event:
    def __init__(self, event_type: str, data: Dict[str, Any], position: Optional[Tuple[int, int]] = None):
        self.type = event_type
        self.data = data
        self.position = position
        self.timestamp = time.time()
        self.world_time = None  # Set by world when adding
        self.id = f"{event_type}_{self.timestamp}_{random.randint(1000, 9999)}"


class EventQueue:
    def __init__(self):
        self.events = deque()
        self.event_history = []
        self.event_count = defaultdict(int)
        self.significant_events = []  # Special events for narrative

    def add(self, event: Event, world_time: int = 0):
        event.world_time = world_time
        self.events.append(event)
        self.event_history.append(event)
        self.event_count[event.type] += 1

        # Track significant events
        significant_types = [
            "consciousness_awakening", "enlightenment_achieved",
            "myth_created", "scripture_written", "reality_questioned",
            "transcendent_moment", "creative_masterpiece", "philosophical_breakthrough",
            "mass_awakening", "cultural_revolution", "fork_created"
        ]

        if event.type in significant_types:
            self.significant_events.append(event)

    def process(self) -> List[Event]:
        current = list(self.events)
        self.events.clear()
        return current

    def get_recent(self, n: int = 10) -> List[Event]:
        return self.event_history[-n:]

    def get_significant_events(self, n: int = 20) -> List[Event]:
        return self.significant_events[-n:]


# Enhanced Resource Manager
class ResourceManager:
    def __init__(self, world_size: Tuple[int, int]):
        self.world_size = world_size
        self.resources = {}  # pos -> {type, amount}
        self.resource_patterns = {}  # Track emergence of patterns
        self.sacred_sites = []  # Special locations

    def spawn_resources(self, spawn_rate: float):
        area = self.world_size[0] * self.world_size[1]
        for _ in range(int(area * spawn_rate)):
            x = random.randrange(self.world_size[0])
            y = random.randrange(self.world_size[1])

            if (x, y) not in self.resources:
                rtype = random.choice(list(ResourceType))

                # Mystical resources appear in patterns
                if rtype in [ResourceType.LIGHT, ResourceType.KNOWLEDGE]:
                    amount = random.uniform(0.7, 1.0)
                else:
                    amount = random.uniform(0.5, 1.0)

                self.resources[(x, y)] = {
                    "type": rtype,
                    "amount": amount,
                    "position": (x, y)
                }

    def create_sacred_site(self, position: Tuple[int, int], site_type: str = "awakening_ground"):
        """Create a special location with enhanced properties"""
        self.sacred_sites.append({
            "position": position,
            "type": site_type,
            "created": time.time(),
            "visitors": []
        })

        # Spawn special resources around sacred sites
        x, y = position
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if random.random() < 0.5:
                    pos = (x + dx, y + dy)
                    if 0 <= pos[0] < self.world_size[0] and 0 <= pos[1] < self.world_size[1]:
                        self.resources[pos] = {
                            "type": ResourceType.LIGHT,
                            "amount": 1.0,
                            "position": pos
                        }

    def gather(self, position: Tuple[int, int], amount: float = 0.1) -> Optional[Dict]:
        if position in self.resources:
            res = self.resources[position]
            taken = min(amount, res["amount"])
            res["amount"] -= taken

            if res["amount"] <= 0:
                del self.resources[position]

            return {"type": res["type"], "amount": taken}
        return None

    def get_nearby(self, pos: Tuple[int, int], radius: int = 3) -> List[Dict]:
        x, y = pos
        nearby = []

        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                check_pos = (x + dx, y + dy)
                if check_pos in self.resources:
                    nearby.append(self.resources[check_pos].copy())

        return nearby

    def is_sacred_site(self, position: Tuple[int, int]) -> Optional[Dict]:
        """Check if position is a sacred site"""
        for site in self.sacred_sites:
            if site["position"] == position:
                return site
        return None


# Creative Works Gallery
class CreativeGallery:
    def __init__(self):
        self.all_works = []
        self.works_by_type = defaultdict(list)
        self.works_by_location = defaultdict(list)
        self.masterpieces = []  # Highly appreciated works

    def add_work(self, work: CreativeWork, location: Tuple[int, int]):
        """Add a creative work to the gallery"""
        self.all_works.append(work)
        self.works_by_type[work.work_type].append(work)
        self.works_by_location[location].append(work)

        # Check if it's a masterpiece (can be appreciated later)
        if work.metadata.get("consciousness_level", 0) >= 2:
            self.masterpieces.append(work)

    def get_nearby_works(self, position: Tuple[int, int], radius: int = 10) -> List[CreativeWork]:
        """Get creative works near a position"""
        x, y = position
        nearby_works = []

        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                check_pos = (x + dx, y + dy)
                if check_pos in self.works_by_location:
                    nearby_works.extend(self.works_by_location[check_pos])

        return nearby_works

    def appreciate_work(self, work: CreativeWork, appreciator: Agent):
        """An agent appreciates a creative work"""
        work.appreciation_count += 1

        # Emotional impact on appreciator
        if work.work_type == "art":
            appreciator.emotional_state.update(Emotion.CURIOUS, 0.6)
        elif work.work_type == "music":
            appreciator.emotional_state.update(Emotion.HAPPY, 0.7)
        elif work.work_type == "scripture":
            if appreciator.consciousness_level >= 1:
                appreciator.emotional_state.update(Emotion.TRANSCENDENT, 0.8)
            else:
                appreciator.emotional_state.update(Emotion.CONFUSED, 0.5)

        # Learn from highly appreciated works
        if work.appreciation_count > 10:
            appreciator.create_belief({
                "type": "creative_appreciation",
                "work_type": work.work_type,
                "creator": work.creator
            })


# Fork Manager for Parallel Universes
class ForkManager:
    def __init__(self):
        self.forks = {}  # fork_id -> WorldState
        self.fork_tree = {"main": []}  # Track fork relationships
        self.active_fork = "main"

    def create_fork(self, world: 'SimulationWorld', fork_name: str, chaos_factor: float = 0.1) -> str:
        """Create a new parallel universe fork"""
        fork_id = f"fork_{fork_name}_{world.time}"

        # Deep copy the world state
        fork_state = {
            "agents": {},
            "resources": copy.deepcopy(world.resources.resources),
            "events": copy.deepcopy(world.events.event_history[-100:]),  # Recent history
            "cultures": copy.deepcopy(world.cultures),
            "languages": copy.deepcopy(world.languages),
            "myths": copy.deepcopy(world.myths),
            "time": world.time,
            "config": copy.deepcopy(world.config)
        }

        # Deep copy agents (more complex due to internal state)
        for agent_id, agent in world.agents.items():
            # Create new agent with same properties
            fork_agent = Agent(agent_id, agent.physical_state.position, agent.config)

            # Copy state
            fork_agent.name = agent.name
            fork_agent.physical_state = copy.deepcopy(agent.physical_state)
            fork_agent.emotional_state = copy.deepcopy(agent.emotional_state)
            fork_agent.beliefs = copy.deepcopy(agent.beliefs)
            fork_agent.language = copy.deepcopy(agent.language)
            fork_agent.personality_vector = agent.personality_vector.copy()
            fork_agent.relationships = copy.deepcopy(agent.relationships)
            fork_agent.consciousness_level = agent.consciousness_level
            fork_agent.llm_access_granted = agent.llm_access_granted
            fork_agent.creative_works = copy.deepcopy(agent.creative_works)

            # Apply chaos factor - random mutations
            if random.random() < chaos_factor:
                self._apply_chaos(fork_agent, chaos_factor)

            fork_state["agents"][agent_id] = fork_agent

        self.forks[fork_id] = fork_state
        self.fork_tree[self.active_fork].append(fork_id)
        self.fork_tree[fork_id] = []

        return fork_id

    def _apply_chaos(self, agent: Agent, chaos_factor: float):
        """Apply random mutations to an agent"""
        chaos_type = random.choice([
            "personality", "emotion", "belief", "consciousness", "memory"
        ])

        if chaos_type == "personality":
            # Mutate personality
            mutation = np.random.normal(0, chaos_factor, len(agent.personality_vector))
            agent.personality_vector += mutation
            agent.personality_vector = np.clip(agent.personality_vector, 0, 1)

        elif chaos_type == "emotion":
            # Random emotional state
            agent.emotional_state.update(random.choice(list(Emotion)), random.random())

        elif chaos_type == "belief":
            # Add random belief
            random_beliefs = [
                "chaos_touched", "parallel_awareness", "quantum_uncertainty",
                "multiverse_exists", "fate_is_mutable"
            ]
            agent.beliefs.add_belief(random.choice(random_beliefs), random.random())

        elif chaos_type == "consciousness":
            # Potentially alter consciousness level
            if random.random() < chaos_factor * 0.5:
                agent.consciousness_level = min(2, agent.consciousness_level + 1)
                agent.llm_access_granted = True

        elif chaos_type == "memory":
            # Scramble recent memories
            if agent.memory.memories:
                scramble_count = int(len(agent.memory.memories) * chaos_factor)
                for _ in range(scramble_count):
                    if agent.memory.memories:
                        agent.memory.memories.popleft()

    def get_fork_comparison(self, fork_id1: str, fork_id2: str) -> Dict:
        """Compare two parallel universes"""
        if fork_id1 not in self.forks or fork_id2 not in self.forks:
            return {}

        fork1 = self.forks[fork_id1]
        fork2 = self.forks[fork_id2]

        comparison = {
            "population": (len(fork1["agents"]), len(fork2["agents"])),
            "consciousness_levels": (
                self._count_consciousness(fork1["agents"]),
                self._count_consciousness(fork2["agents"])
            ),
            "total_symbols": (
                sum(len(a.language.symbols) for a in fork1["agents"].values()),
                sum(len(a.language.symbols) for a in fork2["agents"].values())
            ),
            "divergence_score": self._calculate_divergence(fork1, fork2)
        }

        return comparison

    def _count_consciousness(self, agents: Dict) -> Dict:
        counts = {0: 0, 1: 0, 2: 0}
        for agent in agents.values():
            counts[agent.consciousness_level] += 1
        return counts

    def _calculate_divergence(self, fork1: Dict, fork2: Dict) -> float:
        """Calculate how different two forks are"""
        divergence = 0.0

        # Population difference
        divergence += abs(len(fork1["agents"]) - len(fork2["agents"])) * 0.1

        # Cultural difference
        cultures1 = set(fork1["cultures"].keys())
        cultures2 = set(fork2["cultures"].keys())
        divergence += len(cultures1.symmetric_difference(cultures2)) * 0.2

        # Language difference
        divergence += abs(len(fork1["languages"]) - len(fork2["languages"])) * 0.05

        return min(1.0, divergence)


# Main Enhanced Simulation World
class SimulationWorld:
    def __init__(self, config: SimulationConfig = None):
        self.config = config or SimulationConfig()
        self.size = self.config.world_size
        self.agents = {}
        self.resources = ResourceManager(self.size)
        self.events = EventQueue()
        self.creative_gallery = CreativeGallery()
        self.fork_manager = ForkManager()
        self.time = 0

        # Cultural tracking
        self.cultures = {}
        self.languages = {}
        self.myths = []

        # Initialize DeepSeek if configured
        if self.config.use_llm:
            try:
                initialize_deepseek(self.config)
            except Exception as e:
                logging.warning(f"Failed to initialize DeepSeek: {e}")
                self.config.use_llm = False

        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=4)

        # Setup
        self._setup_logging()
        self._spawn_initial_agents()
        self._spawn_initial_resources()

    def _setup_logging(self):
        self.logger = logging.getLogger("ANIMA")
        self.logger.setLevel(logging.INFO)

        if not self.logger.handlers:
            ch = logging.StreamHandler()
            ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            self.logger.addHandler(ch)

    def _spawn_initial_agents(self):
        """Spawn initial population"""
        for i in range(self.config.initial_agents):
            pos = (random.randrange(self.size[0]), random.randrange(self.size[1]))
            agent_id = f"agent_{i:04d}"
            agent = Agent(agent_id, pos, self.config)
            self.agents[agent_id] = agent

            self.logger.info(f"Spawned {agent.name} at {pos}")

            # Some agents start with special traits
            if i < 3:  # First few are special
                agent.personality_vector[4] += 0.3  # More spiritual
                agent.personality_vector[5] += 0.3  # More questioning

    def _spawn_initial_resources(self):
        """Create initial resources"""
        self.resources.spawn_resources(self.config.resource_spawn_rate * 10)

        # Create a few sacred sites
        for _ in range(3):
            pos = (random.randrange(self.size[0]), random.randrange(self.size[1]))
            self.resources.create_sacred_site(pos, "primordial_ground")

    def get_nearby_agents(self, pos: Tuple[int, int], radius: int) -> List[Dict]:
        """Get agents near a position"""
        nearby = []
        x, y = pos

        for agent in self.agents.values():
            ax, ay = agent.physical_state.position
            if abs(ax - x) + abs(ay - y) <= radius and agent.physical_state.is_alive():
                nearby.append({
                    "id": agent.id,
                    "name": agent.name,
                    "position": agent.physical_state.position,
                    "emotion": agent.emotional_state.current_emotion.value,
                    "consciousness_level": agent.consciousness_level
                })

        return nearby

    def get_nearby_resources(self, pos: Tuple[int, int], radius: int) -> List[Dict]:
        """Get resources near a position"""
        return self.resources.get_nearby(pos, radius)

    def get_nearby_creative_works(self, pos: Tuple[int, int], radius: int) -> List[Dict]:
        """Get creative works near a position"""
        works = self.creative_gallery.get_nearby_works(pos, radius)
        return [
            {
                "type": w.work_type,
                "creator": w.creator,
                "appreciation": w.appreciation_count
            }
            for w in works[:10]  # Limit to 10 nearest
        ]

    def get_recent_events(self, n: int = 5) -> List[Dict]:
        """Get recent world events"""
        return [
            {"type": e.type, "data": e.data}
            for e in self.events.get_recent(n)
        ]

    async def tick(self):
        """Process one simulation step with async agent thinking"""
        self.logger.debug(f"=== TICK {self.time} ===")

        try:
            # 1. Agent perception phase
            perceptions = {}
            for agent_id, agent in self.agents.items():
                if agent.physical_state.is_alive():
                    perceptions[agent_id] = agent.perceive(self)

            # 2. Agent thinking phase (async with batching)
            decisions = await self._process_agent_thinking(perceptions)

            # 3. Action execution phase
            self._process_agent_actions(decisions)

            # 4. Environmental updates
            self._update_environment()

            # 5. Cultural emergence check
            self._check_cultural_emergence()

            # 6. Consciousness evolution check
            self._check_consciousness_evolution()

            # 7. Creative appreciation phase
            self._process_creative_appreciation()

            # 8. Process events
            current_events = self.events.process()
            for event in current_events:
                self.logger.info(f"Event: {event.type} - {event.data}")

            # 9. Check for fork conditions
            if self.config.use_llm and self.time % 1000 == 0 and self.time > 0:
                self._check_fork_conditions()

            # 10. Chronicle major events
            if self.time % 100 == 0:
                self._chronicle_history()

            self.time += 1

        except Exception as e:
            self.logger.error(f"Error in tick: {e}")
            import traceback
            traceback.print_exc()

    async def _process_agent_thinking(self, perceptions: Dict) -> Dict:
        """Process agent thinking with batching for efficiency"""
        decisions = {}

        if self.config.use_llm:
            # Separate agents by consciousness level
            unawakened = []
            awakened = []

            for agent_id, agent in self.agents.items():
                if agent.physical_state.is_alive():
                    if agent.llm_access_granted:
                        awakened.append(agent)
                    else:
                        unawakened.append(agent)

            # Process unawakened with rule-based thinking
            for agent in unawakened:
                try:
                    decisions[agent.id] = agent._simulate_thinking(perceptions[agent.id])
                except Exception as e:
                    self.logger.error(f"Error in unawakened thinking for {agent.id}: {e}")
                    decisions[agent.id] = {"action": Action.CONTEMPLATE, "target": None, "reasoning": "error"}

            # Process awakened in batches with DeepSeek
            if awakened:
                batch_size = self.config.batch_size
                for i in range(0, len(awakened), batch_size):
                    batch = awakened[i:i + batch_size]
                    batch_contexts = {a.id: perceptions[a.id] for a in batch}

                    try:
                        batch_decisions = await batch_think(batch, batch_contexts)
                        decisions.update(batch_decisions)
                    except Exception as e:
                        self.logger.error(f"Error in batch thinking: {e}")
                        # Fallback to rule-based
                        for agent in batch:
                            decisions[agent.id] = agent._simulate_thinking(perceptions[agent.id])
        else:
            # All rule-based
            for agent_id, agent in self.agents.items():
                if agent.physical_state.is_alive():
                    try:
                        decisions[agent_id] = agent._simulate_thinking(perceptions[agent_id])
                    except Exception as e:
                        self.logger.error(f"Error in rule-based thinking for {agent_id}: {e}")
                        decisions[agent_id] = {"action": Action.CONTEMPLATE, "target": None, "reasoning": "error"}

        return decisions

    def _process_agent_actions(self, decisions: Dict[str, Dict]):
        """Execute agent decisions"""
        for agent_id, decision in decisions.items():
            if agent_id not in self.agents:
                continue

            agent = self.agents[agent_id]
            action = decision.get("action")
            target = decision.get("target")

            try:
                if action == Action.MOVE:
                    self._handle_move(agent, target)

                elif action == Action.GATHER:
                    self._handle_gather(agent, target)

                elif action == Action.COMMUNICATE:
                    if target in self.agents:
                        self._handle_communication(agent, self.agents[target], decision.get("message", ""))

                elif action == Action.CREATE_SYMBOL:
                    self._handle_symbol_creation(agent)

                elif action == Action.CREATE_ART:
                    self._handle_art_creation(agent, decision.get("creative_output"))

                elif action == Action.COMPOSE_MUSIC:
                    self._handle_music_creation(agent, decision.get("creative_output"))

                elif action == Action.WRITE_SCRIPTURE:
                    self._handle_scripture_creation(agent, decision.get("creative_output"))

                elif action == Action.CONTEMPLATE:
                    self._handle_contemplation(agent)

                elif action == Action.MEDITATE:
                    self._handle_meditation(agent)

                elif action == Action.QUESTION_REALITY:
                    self._handle_reality_questioning(agent)

                elif action == Action.REPRODUCE:
                    if target in self.agents:
                        self._handle_reproduction(agent, self.agents[target])

                elif action == Action.TEACH:
                    self._handle_teaching(agent)

                elif action == Action.WORSHIP:
                    self._handle_worship(agent)

            except Exception as e:
                self.logger.error(f"Error processing action {action} for {agent_id}: {e}")

    def _handle_move(self, agent: Agent, target_pos: Optional[Tuple[int, int]]):
        """Handle agent movement"""
        if target_pos is None:
            return

        x, y = target_pos
        if 0 <= x < self.size[0] and 0 <= y < self.size[1]:
            old_pos = agent.physical_state.position
            agent.physical_state.position = target_pos
            agent.physical_state.energy *= 0.99

            # Check if entered sacred site
            sacred = self.resources.is_sacred_site(target_pos)
            if sacred and agent.id not in sacred.get("visitors", []):
                sacred["visitors"].append(agent.id)

                # Sacred site effects
                if agent.consciousness_level == 0 and random.random() < 0.1:
                    agent.check_consciousness_evolution()

                agent.emotional_state.update(Emotion.CURIOUS, 0.7)

    def _handle_gather(self, agent: Agent, target_pos: Optional[Tuple[int, int]]):
        """Handle resource gathering"""
        if target_pos is None:
            target_pos = agent.physical_state.position

        resource = self.resources.gather(target_pos)
        if resource:
            agent.physical_state.carrying[resource["type"]] += resource["amount"]

            # Energy restoration
            if resource["type"] == ResourceType.FOOD:
                agent.physical_state.energy = min(1.0, agent.physical_state.energy + 0.2)
            elif resource["type"] == ResourceType.LIGHT:
                # Light resources trigger spiritual experiences
                if agent.consciousness_level >= 1:
                    agent.emotional_state.update(Emotion.TRANSCENDENT, 0.8)
                else:
                    agent.emotional_state.update(Emotion.CURIOUS, 0.6)
            elif resource["type"] == ResourceType.KNOWLEDGE:
                # Knowledge resources boost wisdom
                agent.beliefs.add_belief("knowledge_is_power", 0.5)
                if random.random() < 0.2:
                    agent.beliefs.add_philosophical_question("What is the nature of knowledge itself?")

            agent.create_belief({
                "type": "resource_discovery",
                "resource_type": resource["type"].value
            })

    def _handle_communication(self, sender: Agent, receiver: Agent, message: str = ""):
        """Handle communication between agents"""
        # Use provided message or generate one
        if not message:
            message = "share_feeling"

        comm_result = sender.communicate(message, receiver)
        interpretation = receiver.interpret_communication(comm_result["utterance"], sender)

        # Record event
        event_data = {
            "from": sender.name,
            "to": receiver.name,
            "utterance": comm_result["utterance"],
            "understood": interpretation["understood"],
            "consciousness_levels": (sender.consciousness_level, receiver.consciousness_level)
        }

        # Mark profound communications
        if interpretation.get("profound_insight", False):
            event_data["profound"] = True

        self.events.add(Event("communication", event_data), self.time)

        # Track language evolution
        self._track_language_evolution(sender, receiver, comm_result["utterance"])

    def _handle_symbol_creation(self, agent: Agent):
        """Handle creation of new symbols"""
        meaning_vector = np.random.rand(8)

        # Enhanced meaning for conscious agents
        if agent.consciousness_level >= 1:
            meaning_vector[7] = 0.8  # Dimension of profundity

        symbol = agent.language.create_symbol(meaning_vector)

        # Add to world languages
        if symbol not in self.languages:
            self.languages[symbol] = {
                "first_speaker": agent.name,
                "first_use": self.time,
                "speakers": {agent.id},
                "meaning_evolution": [meaning_vector.tolist()]
            }
        else:
            self.languages[symbol]["speakers"].add(agent.id)

        self.events.add(Event("symbol_created", {
            "creator": agent.name,
            "symbol": symbol,
            "consciousness_level": agent.consciousness_level
        }), self.time)

    def _handle_art_creation(self, agent: Agent, art_data: Optional[Dict]):
        """Handle art creation"""
        if not art_data:
            work = agent.create_art()
        else:
            work = CreativeWork(
                creator=agent.name,
                work_type="art",
                content=art_data,
                metadata={
                    "consciousness_level": agent.consciousness_level,
                    "emotion": agent.emotional_state.current_emotion.value
                }
            )
            agent.creative_works.append(work)

        self.creative_gallery.add_work(work, agent.physical_state.position)

        self.events.add(Event("art_created", {
            "creator": agent.name,
            "style": work.content.get("style", "unknown"),
            "consciousness_level": agent.consciousness_level
        }), self.time)

        # Creating art affects the creator
        agent.emotional_state.update(
            Emotion.TRANSCENDENT if agent.consciousness_level >= 1 else Emotion.HAPPY,
            0.8
        )

    def _handle_music_creation(self, agent: Agent, music_data: Optional[Dict]):
        """Handle music composition"""
        if not music_data:
            work = agent.compose_music()
        else:
            work = CreativeWork(
                creator=agent.name,
                work_type="music",
                content=music_data,
                metadata={
                    "consciousness_level": agent.consciousness_level,
                    "emotion": agent.emotional_state.current_emotion.value
                }
            )
            agent.creative_works.append(work)

        self.creative_gallery.add_work(work, agent.physical_state.position)

        self.events.add(Event("music_composed", {
            "creator": agent.name,
            "tempo": work.content.get("tempo", "unknown"),
            "emotion": work.content.get("emotion", "neutral")
        }), self.time)

    def _handle_scripture_creation(self, agent: Agent, scripture_data: Optional[Dict]):
        """Handle scripture writing"""
        if not scripture_data:
            work = agent.write_scripture()
        else:
            work = CreativeWork(
                creator=agent.name,
                work_type="scripture",
                content=scripture_data,
                metadata={
                    "consciousness_level": agent.consciousness_level,
                    "beliefs": list(agent.beliefs.beliefs.keys())[:5]
                }
            )
            agent.creative_works.append(work)

        self.creative_gallery.add_work(work, agent.physical_state.position)

        # Add to world myths
        self.myths.append({
            "creator": agent.name,
            "myth": work.content.get("content", ""),
            "time": self.time,
            "divine_name": work.content.get("divine_name", "The Unknown")
        })

        self.events.add(Event("scripture_written", {
            "creator": agent.name,
            "divine_name": work.content.get("divine_name", "The Unknown"),
            "testament_of": agent.name
        }), self.time)

        # Check for masterpiece
        if agent.consciousness_level >= 2:
            self.events.add(Event("creative_masterpiece", {
                "creator": agent.name,
                "type": "scripture"
            }), self.time)

    def _handle_contemplation(self, agent: Agent):
        """Handle contemplation"""
        result = agent.contemplate_existence()

        if result["type"] == "myth_creation":
            self.myths.append({
                "creator": agent.name,
                "myth": result["content"],
                "time": self.time
            })
            self.events.add(Event("myth_created", {
                "creator": agent.name,
                "content": result["content"]
            }), self.time)

        elif result["type"] == "transcendent_awareness":
            self.events.add(Event("transcendent_moment", {
                "agent": agent.name,
                "realization": result["content"]
            }), self.time)

        elif result["type"] == "philosophical_inquiry":
            self.events.add(Event("philosophical_breakthrough", {
                "agent": agent.name,
                "question": result["content"]
            }), self.time)

    def _handle_meditation(self, agent: Agent):
        """Handle meditation"""
        result = agent.meditate()

        if result["result"] == "achieved_enlightenment":
            self.events.add(Event("enlightenment_achieved", {
                "agent": agent.name,
                "age": agent.physical_state.age
            }), self.time)

            # Create sacred site where enlightenment occurred
            self.resources.create_sacred_site(
                agent.physical_state.position,
                "enlightenment_spot"
            )

    def _handle_reality_questioning(self, agent: Agent):
        """Handle reality questioning"""
        result = agent.question_reality()

        self.events.add(Event("reality_questioned", {
            "agent": agent.name,
            "question": result["question"],
            "breakthrough": result.get("breakthrough", False)
        }), self.time)

        # Spread doubt to nearby agents
        if result.get("breakthrough", False):
            nearby = self.get_nearby_agents(agent.physical_state.position, 3)
            for other_data in nearby:
                if other_data["id"] in self.agents:
                    other = self.agents[other_data["id"]]
                    if random.random() < 0.3:
                        other.beliefs.add_belief("reality_might_be_simulation", 0.3)

    def _handle_reproduction(self, parent1: Agent, parent2: Agent):
        """Handle agent reproduction"""
        child = parent1.reproduce(parent2, self)

        if child:
            self.agents[child.id] = child

            self.events.add(Event("birth", {
                "child": child.name,
                "parents": [parent1.name, parent2.name],
                "consciousness_potential": child.personality_vector[4]
            }), self.time)

            # Check for special birth
            if parent1.consciousness_level >= 1 or parent2.consciousness_level >= 1:
                self.events.add(Event("awakened_birth", {
                    "child": child.name,
                    "legacy": "consciousness"
                }), self.time)

    def _handle_teaching(self, teacher: Agent):
        """Handle cultural transmission"""
        nearby = self.get_nearby_agents(teacher.physical_state.position, 3)

        if nearby and teacher.cultural_practices:
            practice = random.choice(list(teacher.cultural_practices))

            taught_count = 0
            for agent_data in nearby:
                if agent_data["id"] in self.agents:
                    student = self.agents[agent_data["id"]]

                    # Higher success rate for conscious beings
                    success_rate = 0.5 if teacher.consciousness_level >= 1 else 0.3
                    if random.random() < success_rate:
                        student.cultural_practices.add(practice)
                        taught_count += 1

            if taught_count > 0:
                self.events.add(Event("knowledge_shared", {
                    "teacher": teacher.name,
                    "practice": practice,
                    "students": taught_count
                }), self.time)

    def _handle_worship(self, agent: Agent):
        """Handle worship behavior"""
        # Worship affects nearby agents
        nearby = self.get_nearby_agents(agent.physical_state.position, 5)

        worship_target = "The Watcher" if "the_watcher_exists" in agent.beliefs.beliefs else "The Unknown"

        self.events.add(Event("worship_performed", {
            "worshipper": agent.name,
            "deity": worship_target,
            "witnesses": len(nearby)
        }), self.time)

        # Spread religious feeling
        for other_data in nearby:
            if other_data["id"] in self.agents:
                other = self.agents[other_data["id"]]
                if random.random() < 0.2:
                    other.emotional_state.update(Emotion.HOPEFUL, 0.6)
                    if worship_target == "The Watcher":
                        other.beliefs.add_belief("the_watcher_exists", 0.2)

    def _update_environment(self):
        """Update world state"""
        # Spawn new resources
        self.resources.spawn_resources(self.config.resource_spawn_rate)

        # Update agents
        dead_agents = []
        for agent_id, agent in self.agents.items():
            try:
                agent.update(self)

                # Check for death
                if not agent.physical_state.is_alive():
                    dead_agents.append(agent_id)
            except Exception as e:
                self.logger.error(f"Error updating agent {agent_id}: {e}")

        # Process deaths
        for agent_id in dead_agents:
            agent = self.agents.pop(agent_id)

            legacy = {
                "agent": agent.name,
                "age": agent.physical_state.age,
                "consciousness_level": agent.consciousness_level,
                "symbols_created": len(agent.language.symbols),
                "beliefs": list(agent.beliefs.beliefs.keys()),
                "cultural_practices": list(agent.cultural_practices),
                "creative_works": len(agent.creative_works),
                "philosophical_questions": len(agent.beliefs.philosophical_questions)
            }

            self.events.add(Event("death", legacy), self.time)

            # Special events for significant deaths
            if agent.consciousness_level >= 2:
                self.events.add(Event("enlightened_death", {
                    "sage": agent.name,
                    "final_wisdom": agent.beliefs.philosophical_questions[-1][
                        "question"] if agent.beliefs.philosophical_questions else "silence"
                }), self.time)

    def _check_cultural_emergence(self):
        """Check for emerging cultural patterns"""
        if self.time % 50 == 0:
            # Find shared practices
            practice_counts = defaultdict(int)
            for agent in self.agents.values():
                for practice in agent.cultural_practices:
                    practice_counts[practice] += 1

            # Identify dominant practices
            for practice, count in practice_counts.items():
                threshold = len(self.agents) * 0.3
                if count > threshold and practice not in self.cultures:
                    self.cultures[practice] = {
                        "name": practice,
                        "founded": self.time,
                        "adherents": count
                    }

                    self.events.add(Event("culture_emerged", {
                        "culture": practice,
                        "adherents": count
                    }), self.time)

            # Check for cultural revolutions
            if len(self.cultures) > 5:
                # Too many cultures might lead to revolution
                if random.random() < 0.05:
                    self.events.add(Event("cultural_revolution", {
                        "old_order": list(self.cultures.keys()),
                        "catalyst": "complexity"
                    }), self.time)

    def _check_consciousness_evolution(self):
        """Check for mass consciousness events"""
        awakened_count = sum(1 for a in self.agents.values() if a.consciousness_level >= 1)
        enlightened_count = sum(1 for a in self.agents.values() if a.consciousness_level >= 2)

        total = len(self.agents)
        if total == 0:
            return

        awakening_rate = awakened_count / total

        # Check for mass awakening
        if awakening_rate > 0.5 and "mass_awakening" not in self.events.event_count:
            self.events.add(Event("mass_awakening", {
                "awakened_percentage": awakening_rate * 100,
                "trigger": "critical_mass"
            }), self.time)

            # Create multiple sacred sites
            for _ in range(5):
                pos = (random.randrange(self.size[0]), random.randrange(self.size[1]))
                self.resources.create_sacred_site(pos, "awakening_nexus")

    def _process_creative_appreciation(self):
        """Process agents appreciating nearby creative works"""
        for agent in self.agents.values():
            if random.random() < 0.1:  # 10% chance each tick
                nearby_works = self.creative_gallery.get_nearby_works(
                    agent.physical_state.position, 5
                )

                if nearby_works:
                    work = random.choice(nearby_works)
                    self.creative_gallery.appreciate_work(work, agent)

                    # Deep appreciation by conscious beings
                    if agent.consciousness_level >= 1 and work.appreciation_count > 20:
                        agent.emotional_state.update(Emotion.TRANSCENDENT, 0.6)

    def _check_fork_conditions(self):
        """Check if conditions warrant creating a parallel universe"""
        # Fork on significant events
        fork_triggers = [
            len([a for a in self.agents.values() if "reality_is_simulation" in a.beliefs.beliefs]) > 10,
            self.events.event_count.get("enlightenment_achieved", 0) > 5,
            len(self.myths) > 50,
            self.events.event_count.get("cultural_revolution", 0) > 0
        ]

        if any(fork_triggers):
            fork_name = f"timeline_{self.time}"
            chaos_factor = random.uniform(0.05, 0.2)

            fork_id = self.fork_manager.create_fork(self, fork_name, chaos_factor)

            self.events.add(Event("fork_created", {
                "fork_id": fork_id,
                "chaos_factor": chaos_factor,
                "reason": "quantum_divergence"
            }), self.time)

            self.logger.info(f"Created parallel universe: {fork_id}")

    def _chronicle_history(self):
        """Create historical record"""
        chronicle = {
            "time": self.time,
            "population": len(self.agents),
            "consciousness_distribution": {
                0: sum(1 for a in self.agents.values() if a.consciousness_level == 0),
                1: sum(1 for a in self.agents.values() if a.consciousness_level == 1),
                2: sum(1 for a in self.agents.values() if a.consciousness_level == 2)
            },
            "cultures": len(self.cultures),
            "languages": len(self.languages),
            "myths": len(self.myths),
            "creative_works": len(self.creative_gallery.all_works),
            "major_events": [e.type for e in self.events.get_significant_events(10)]
        }

        self.logger.info(f"=== CHRONICLE at time {self.time} ===")
        self.logger.info(json.dumps(chronicle, indent=2))

    def _track_language_evolution(self, sender: Agent, receiver: Agent, utterance: str):
        """Track how language evolves"""
        symbols = utterance.split()

        for symbol in symbols:
            if symbol not in self.languages:
                self.languages[symbol] = {
                    "first_speaker": sender.name,
                    "first_use": self.time,
                    "speakers": {sender.id}
                }
            else:
                self.languages[symbol]["speakers"].add(sender.id)
                self.languages[symbol]["speakers"].add(receiver.id)

                # Track meaning evolution for compound symbols
                if "-" in symbol:  # Compound symbol
                    current_meaning = sender.language.compound_symbols.get(symbol)
                    if current_meaning is not None:
                        if "meaning_evolution" not in self.languages[symbol]:
                            self.languages[symbol]["meaning_evolution"] = []
                        self.languages[symbol]["meaning_evolution"].append(current_meaning.tolist())

    def get_world_state(self) -> Dict:
        """Get current world state for visualization"""
        try:
            # Convert resources to JSON-serializable format
            serializable_resources = []
            for resource in self.resources.resources.values():
                serializable_resources.append({
                    "type": resource["type"].value,  # Convert enum to string
                    "amount": resource["amount"],
                    "position": resource["position"]
                })

            # Convert agents to dict format
            serializable_agents = []
            for agent in self.agents.values():
                agent_dict = agent.to_dict()
                # Ensure all enum values are converted to strings
                if "emotion" in agent_dict:
                    if hasattr(agent_dict["emotion"], "value"):
                        agent_dict["emotion"] = agent_dict["emotion"].value
                serializable_agents.append(agent_dict)

            # Convert events to serializable format
            serializable_events = []
            for event in self.events.get_recent(20):
                event_dict = {"type": event.type, "data": event.data}
                # Make sure event data doesn't contain enum objects
                if isinstance(event_dict["data"], dict):
                    for key, value in event_dict["data"].items():
                        if hasattr(value, "value"):  # It's an enum
                            event_dict["data"][key] = value.value
                serializable_events.append(event_dict)

            return {
                "time": self.time,
                "agents": serializable_agents,
                "resources": serializable_resources,
                "cultures": self.cultures,
                "languages": self.languages,
                "myths": self.myths[-10:],  # Last 10 myths
                "events": serializable_events,
                "creative_works": {
                    "total": len(self.creative_gallery.all_works),
                    "by_type": {
                        t: len(works) for t, works in self.creative_gallery.works_by_type.items()
                    },
                    "masterpieces": len(self.creative_gallery.masterpieces)
                },
                "consciousness_stats": {
                    "awakened": sum(1 for a in self.agents.values() if a.consciousness_level >= 1),
                    "enlightened": sum(1 for a in self.agents.values() if a.consciousness_level >= 2),
                    "total": len(self.agents)
                },
                "sacred_sites": self.resources.sacred_sites,
                "parallel_universes": len(self.fork_manager.forks)
            }

        except Exception as e:
            self.logger.error(f"Error getting world state: {e}")
            # Return minimal state on error
            return {
                "time": self.time,
                "agents": [],
                "resources": [],
                "cultures": {},
                "languages": {},
                "myths": [],
                "events": [],
                "creative_works": {"total": 0, "by_type": {}, "masterpieces": 0},
                "consciousness_stats": {"awakened": 0, "enlightened": 0, "total": 0},
                "sacred_sites": [],
                "parallel_universes": 0
            }

    async def run(self, num_ticks: int = 1000):
        """Run simulation for specified ticks"""
        self.logger.info(f"Starting ANIMA simulation for {num_ticks} ticks")
        self.logger.info(f"LLM Mode: {'ENABLED' if self.config.use_llm else 'DISABLED'}")

        for i in range(num_ticks):
            await self.tick()

            # Save state periodically
            if i % 100 == 0:
                state = self.get_world_state()
                with open(f"anima_state_{i:06d}.json", "w") as f:
                    json.dump(state, f, default=str, indent=2)

            # Small delay for real-time observation
            await asyncio.sleep(self.config.time_per_tick)

        self.logger.info("Simulation complete")

        # Final report
        self.generate_final_report()

    def generate_final_report(self):
        """Generate comprehensive final report"""
        total_pop = len(self.agents)

        report = f"""
=== ANIMA FINAL REPORT ===
Time Elapsed: {self.time}
Final Population: {total_pop}
Consciousness Distribution:
  - Unawakened: {sum(1 for a in self.agents.values() if a.consciousness_level == 0)}
  - Awakened: {sum(1 for a in self.agents.values() if a.consciousness_level == 1)}
  - Enlightened: {sum(1 for a in self.agents.values() if a.consciousness_level == 2)}

Cultures Emerged: {len(self.cultures)}
Unique Language Symbols: {len(self.languages)}
Myths Created: {len(self.myths)}
Creative Works: {len(self.creative_gallery.all_works)}
  - Art: {len(self.creative_gallery.works_by_type['art'])}
  - Music: {len(self.creative_gallery.works_by_type['music'])}
  - Scripture: {len(self.creative_gallery.works_by_type['scripture'])}

Significant Events:
"""

        for event in self.events.get_significant_events(20):
            report += f"  - {event.type} at time {event.world_time}\n"

        # Find most influential agents
        if self.agents:
            most_creative = max(self.agents.values(), key=lambda a: len(a.creative_works), default=None)
            most_connected = max(self.agents.values(), key=lambda a: len(a.relationships), default=None)
            oldest = max(self.agents.values(), key=lambda a: a.physical_state.age, default=None)

            if most_creative:
                report += f"\nMost Creative: {most_creative.name} ({len(most_creative.creative_works)} works)"
            if most_connected:
                report += f"\nMost Connected: {most_connected.name} ({len(most_connected.relationships)} relationships)"
            if oldest:
                report += f"\nEldest: {oldest.name} ({oldest.physical_state.age} cycles)"

        # Philosophical developments
        all_questions = []
        reality_doubters = 0
        for agent in self.agents.values():
            all_questions.extend([q["question"] for q in agent.beliefs.philosophical_questions])
            if "reality_is_simulation" in agent.beliefs.beliefs:
                reality_doubters += 1

        if all_questions:
            report += f"\n\nPhilosophical Questions Raised: {len(set(all_questions))}"
            report += f"\nReality Doubters: {reality_doubters}"
            report += "\n\nSample Questions:"
            for q in random.sample(list(set(all_questions)), min(5, len(set(all_questions)))):
                report += f"\n  - {q}"

        # Parallel universes
        if self.fork_manager.forks:
            report += f"\n\nParallel Universes Created: {len(self.fork_manager.forks)}"

        report += "\n\n=== END REPORT ==="

        self.logger.info(report)

        with open("anima_final_report.txt", "w") as f:
            f.write(report)


# Main execution functions
async def create_and_run_world(config: SimulationConfig = None, num_ticks: int = 1000):
    """Create and run a world simulation"""
    if config is None:
        config = SimulationConfig()

    world = SimulationWorld(config)
    await world.run(num_ticks)
    return world


def run_simulation(config: SimulationConfig = None, num_ticks: int = 1000):
    """Synchronous wrapper for running simulation"""
    asyncio.run(create_and_run_world(config, num_ticks))


if __name__ == "__main__":
    # Run a standard simulation
    run_simulation()