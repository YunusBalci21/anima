import numpy as np
import random
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, deque
import time
from enum import Enum
import chromadb
from chromadb.utils import embedding_functions
import asyncio
import logging

# Configuration
@dataclass
class SimulationConfig:
    world_size: Tuple[int, int] = (50, 50)
    initial_agents: int = 20
    resource_spawn_rate: float = 0.1
    time_per_tick: float = 0.1
    max_memory_size: int = 100
    language_mutation_rate: float = 0.05
    death_threshold: float = 0.0
    reproduction_threshold: float = 0.8
    model_name: str = "gpt-3.5-turbo"  # Can switch to DeepSeek or local models
    
# Emotional States
class Emotion(Enum):
    NEUTRAL = "neutral"
    HAPPY = "happy"
    ANGRY = "angry"
    FEARFUL = "fearful"
    CURIOUS = "curious"
    LONELY = "lonely"
    LOVING = "loving"
    CONFUSED = "confused"
    HOPEFUL = "hopeful"
    DESPERATE = "desperate"

# Resource Types
class ResourceType(Enum):
    FOOD = "food"
    WATER = "water"
    SHELTER = "shelter"
    LIGHT = "light"  # Mystical resource
    KNOWLEDGE = "knowledge"  # Abstract resource

# Actions
class Action(Enum):
    MOVE = "move"
    GATHER = "gather"
    COMMUNICATE = "communicate"
    SHARE = "share"
    ATTACK = "attack"
    REPRODUCE = "reproduce"
    CONTEMPLATE = "contemplate"
    CREATE_SYMBOL = "create_symbol"
    TEACH = "teach"
    WORSHIP = "worship"

# Physical State
@dataclass
class PhysicalState:
    energy: float = 1.0
    health: float = 1.0
    age: int = 0
    position: Tuple[int, int] = (0, 0)
    carrying: Dict[ResourceType, float] = None
    
    def __post_init__(self):
        if self.carrying is None:
            self.carrying = {r: 0.0 for r in ResourceType}
    
    def decay(self):
        self.energy *= 0.98
        self.health *= 0.995
        self.age += 1
        
    def is_alive(self) -> bool:
        return self.energy > 0 and self.health > 0

# Emotional State
class EmotionalState:
    def __init__(self):
        self.current_emotion = Emotion.NEUTRAL
        self.emotion_intensity = 0.5
        self.emotion_history = deque(maxlen=20)
        self.emotional_memory = {}  # agent_id -> last_emotion_caused
        
    def update(self, new_emotion: Emotion, intensity: float = 0.5):
        self.emotion_history.append((self.current_emotion, self.emotion_intensity))
        self.current_emotion = new_emotion
        self.emotion_intensity = min(1.0, intensity)
        
    def get_emotional_state(self) -> Dict:
        return {
            "current": self.current_emotion.value,
            "intensity": self.emotion_intensity,
            "history": list(self.emotion_history)[-5:]
        }

# Belief System
class BeliefSystem:
    def __init__(self):
        self.beliefs = {}  # concept -> belief_strength
        self.myths = []  # List of narrative beliefs
        self.values = {}  # moral_concept -> importance
        self.taboos = set()  # Forbidden actions/concepts
        self.sacred_symbols = set()
        
    def add_belief(self, concept: str, strength: float = 0.5):
        if concept in self.beliefs:
            # Reinforce existing belief
            self.beliefs[concept] = min(1.0, self.beliefs[concept] + 0.1)
        else:
            self.beliefs[concept] = strength
            
    def add_myth(self, narrative: str):
        self.myths.append({
            "story": narrative,
            "created_at": time.time(),
            "belief_strength": 0.5
        })
        
    def add_value(self, moral_concept: str, importance: float):
        self.values[moral_concept] = importance
        
    def is_taboo(self, action: str) -> bool:
        return action in self.taboos

# Language System
class Language:
    def __init__(self):
        self.symbols = {}  # symbol -> meaning_vector
        self.grammar_patterns = []  # List of valid symbol combinations
        self.utterances = []  # History of all utterances
        self.symbol_counter = 0
        
    def create_symbol(self, meaning_vector: np.ndarray) -> str:
        """Create a new symbol for a concept"""
        # Generate pseudo-random symbol based on meaning
        vowels = "aeiou"
        consonants = "bcdfghjklmnpqrstvwxyz"
        length = random.randint(2, 6)
        symbol = ""
        
        for i in range(length):
            if i % 2 == 0:
                symbol += random.choice(consonants)
            else:
                symbol += random.choice(vowels)
                
        # Add unique identifier if collision
        if symbol in self.symbols:
            symbol += str(self.symbol_counter)
            self.symbol_counter += 1
            
        self.symbols[symbol] = meaning_vector
        return symbol
        
    def find_symbol(self, meaning_vector: np.ndarray, threshold: float = 0.8) -> Optional[str]:
        """Find existing symbol for similar meaning"""
        for symbol, vec in self.symbols.items():
            similarity = np.dot(meaning_vector, vec) / (np.linalg.norm(meaning_vector) * np.linalg.norm(vec))
            if similarity > threshold:
                return symbol
        return None
        
    def mutate_symbol(self, symbol: str) -> str:
        """Create variation of existing symbol"""
        if symbol not in self.symbols:
            return symbol
            
        chars = list(symbol)
        if random.random() < 0.5 and len(chars) > 2:
            # Swap two characters
            i, j = random.sample(range(len(chars)), 2)
            chars[i], chars[j] = chars[j], chars[i]
        else:
            # Change one character
            i = random.randint(0, len(chars) - 1)
            if chars[i] in "aeiou":
                chars[i] = random.choice("aeiou")
            else:
                chars[i] = random.choice("bcdfghjklmnpqrstvwxyz")
                
        new_symbol = "".join(chars)
        # Slightly mutate meaning too
        self.symbols[new_symbol] = self.symbols[symbol] + np.random.normal(0, 0.1, len(self.symbols[symbol]))
        return new_symbol

# Memory Systems
class ShortTermMemory:
    def __init__(self, capacity: int = 100):
        self.capacity = capacity
        self.memories = deque(maxlen=capacity)

    def add(self, memory: Dict):
        memory["timestamp"] = time.time()
        self.memories.append(memory)

    def get_recent(self, n: int = 10) -> List[Dict]:
        return list(self.memories)[-n:]

    def search(self, query: str) -> List[Dict]:
        results = []
        for memory in self.memories:
            if query.lower() in str(memory).lower():
                results.append(memory)
        return results

class VectorMemory:
    def __init__(self):
        self.client = chromadb.Client()
        self.collection = self.client.create_collection(
            name=f"agent_memory_{random.randint(1000, 9999)}",
            embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction()
        )
        
    def store(self, text: str, metadata: Dict):
        self.collection.add(
            documents=[text],
            metadatas=[metadata],
            ids=[f"memory_{time.time()}_{random.randint(1000, 9999)}"]
        )
        
    def retrieve(self, query: str, n_results: int = 5) -> List[Dict]:
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        return results

# The Agent
class Agent:
    def __init__(self, agent_id: str, position: Tuple[int, int], model_config: Dict):
        self.id = agent_id
        self.name = self._generate_name()
        self.model_config = model_config
        
        # Core systems
        self.memory = ShortTermMemory(capacity=100)
        self.long_term_memory = VectorMemory()
        self.physical_state = PhysicalState(position=position)
        self.emotional_state = EmotionalState()
        self.beliefs = BeliefSystem()
        self.language = Language()
        
        # Personality and relationships
        self.personality_vector = np.random.rand(32)
        self.relationships = {}  # agent_id -> relationship_score
        self.reputation = 0.0
        
        # Learning and culture
        self.skills = {}  # skill -> proficiency
        self.cultural_practices = set()
        self.inventions = []
        
        # Internal state
        self.current_goal = None
        self.pending_actions = []
        self.last_action_time = time.time()
        
    def _generate_name(self) -> str:
        """Generate a unique name for the agent"""
        syllables = ["ka", "ra", "mi", "to", "na", "be", "lu", "so", "chi", "wa"]
        return "".join(random.sample(syllables, random.randint(2, 3))).capitalize()
        
    def perceive(self, environment: 'SimulationWorld') -> Dict:
        """Process sensory input from environment"""
        perception = {
            "position": self.physical_state.position,
            "nearby_agents": environment.get_nearby_agents(self.physical_state.position, radius=5),
            "nearby_resources": environment.get_nearby_resources(self.physical_state.position, radius=3),
            "time": environment.time,
            "events": environment.get_recent_events()
        }
        
        # Store in memory
        self.memory.add({
            "type": "perception",
            "data": perception,
            "emotion": self.emotional_state.current_emotion.value
        })
        
        return perception
        
    async def think(self, context: Dict) -> Dict:
        """Use LLM to process context and decide on actions"""
        # Prepare prompt with agent's unique perspective
        prompt = self._build_thinking_prompt(context)
        
        # Simulate constrained thinking (or use actual LLM)
        if self.model_config.get("use_llm", False):
            # Use actual LLM
            response = await self._call_llm(prompt)
            decision = self._parse_llm_response(response)
        else:
            # Simulate decision-making
            decision = self._simulate_thinking(context)
            
        # Update internal state based on decision
        self._update_internal_state(decision)
        
        return decision
        
    def _build_thinking_prompt(self, context: Dict) -> str:
        """Build a prompt that reflects agent's unique perspective"""
        prompt = f"""You are {self.name}, a being in a world you're trying to understand.
        
Your emotional state: {self.emotional_state.current_emotion.value} (intensity: {self.emotional_state.emotion_intensity})
Your beliefs: {json.dumps(list(self.beliefs.beliefs.keys())[:5])}
Your values: {json.dumps(list(self.beliefs.values.keys())[:3])}

Current situation:
- Location: {context['position']}
- Nearby beings: {len(context['nearby_agents'])}
- Available resources: {[r['type'] for r in context['nearby_resources']]}
- Recent memories: {[m['type'] for m in self.memory.get_recent(3)]}

Your personality traits guide you to value: {self._describe_personality()}

What do you want to do? Consider:
1. Your immediate needs (energy: {self.physical_state.energy}, health: {self.physical_state.health})
2. Your relationships and community
3. Your understanding of the world
4. Your creative or spiritual impulses

Respond with your intention and reasoning."""
        
        return prompt
        
    def _describe_personality(self) -> str:
        """Generate personality description from vector"""
        traits = []
        if self.personality_vector[0] > 0.7:
            traits.append("curiosity and exploration")
        if self.personality_vector[1] > 0.7:
            traits.append("social connection")
        if self.personality_vector[2] > 0.7:
            traits.append("creativity and expression")
        if self.personality_vector[3] > 0.7:
            traits.append("order and structure")
        if self.personality_vector[4] > 0.7:
            traits.append("spiritual contemplation")
            
        return ", ".join(traits) if traits else "balance and survival"
        
    def _simulate_thinking(self, context: Dict) -> Dict:
        """Simulate decision-making without LLM"""
        # Priority-based decision making
        decision = {"action": None, "target": None, "reasoning": ""}
        
        # Check survival needs
        if self.physical_state.energy < 0.3:
            # Need food
            if context['nearby_resources']:
                food_resources = [r for r in context['nearby_resources'] if r['type'] == ResourceType.FOOD]
                if food_resources:
                    decision['action'] = Action.GATHER
                    decision['target'] = food_resources[0]['position']
                    decision['reasoning'] = "hungy_need_food"
            else:
                decision['action'] = Action.MOVE
                decision['target'] = self._random_direction()
                decision['reasoning'] = "search_food"
                
        # Social needs
        elif self.emotional_state.current_emotion == Emotion.LONELY and context['nearby_agents']:
            target_agent = random.choice(context['nearby_agents'])
            decision['action'] = Action.COMMUNICATE
            decision['target'] = target_agent['id']
            decision['reasoning'] = "lonely_seek_companionship"
            
        # Creative impulse
        elif random.random() < 0.1 * self.personality_vector[2]:  # Creativity trait
            decision['action'] = Action.CREATE_SYMBOL
            decision['reasoning'] = "inspired_create"
            
        # Exploration
        elif random.random() < 0.2 * self.personality_vector[0]:  # Curiosity trait
            decision['action'] = Action.MOVE
            decision['target'] = self._random_direction()
            decision['reasoning'] = "explore_world"
            
        # Default: contemplate
        else:
            decision['action'] = Action.CONTEMPLATE
            decision['reasoning'] = "think_existence"
            
        return decision
        
    def _random_direction(self) -> Tuple[int, int]:
        """Get random adjacent position"""
        x, y = self.physical_state.position
        dx, dy = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
        return (x + dx, y + dy)
        
    def communicate(self, message: str, target_agent: 'Agent') -> Dict:
        """Attempt to communicate with another agent"""
        # Create utterance based on current language
        if not self.language.symbols:
            # Create first symbol
            meaning = np.random.rand(8)  # Random meaning vector
            symbol = self.language.create_symbol(meaning)
            utterance = symbol
        else:
            # Use existing symbols or create new ones
            symbols_to_use = []
            concepts = message.split()[:3]  # Simple tokenization
            
            for concept in concepts:
                meaning_vector = self._concept_to_vector(concept)
                existing_symbol = self.language.find_symbol(meaning_vector)
                
                if existing_symbol:
                    # Maybe mutate it
                    if random.random() < 0.1:
                        symbols_to_use.append(self.language.mutate_symbol(existing_symbol))
                    else:
                        symbols_to_use.append(existing_symbol)
                else:
                    # Create new symbol
                    symbols_to_use.append(self.language.create_symbol(meaning_vector))
                    
            utterance = " ".join(symbols_to_use)
            
        # Record the communication attempt
        comm_event = {
            "type": "communication",
            "from": self.id,
            "to": target_agent.id,
            "utterance": utterance,
            "intended_meaning": message,
            "timestamp": time.time()
        }
        
        self.memory.add(comm_event)
        self.language.utterances.append(comm_event)
        
        # Update relationship
        if target_agent.id not in self.relationships:
            self.relationships[target_agent.id] = 0.5
        self.relationships[target_agent.id] += 0.05
        
        return {
            "utterance": utterance,
            "emotion": self.emotional_state.current_emotion.value
        }
        
    def _concept_to_vector(self, concept: str) -> np.ndarray:
        """Convert a concept to a meaning vector"""
        # Simple hash-based encoding
        vector = np.zeros(8)
        for i, char in enumerate(concept[:8]):
            vector[i % 8] += ord(char) / 100.0
        return vector / np.linalg.norm(vector)
        
    def interpret_communication(self, utterance: str, sender: 'Agent') -> Dict:
        """Attempt to understand another agent's communication"""
        interpretation = {
            "understood": False,
            "inferred_meaning": None,
            "emotional_response": None
        }
        
        # Check if we know any of the symbols
        symbols = utterance.split()
        known_symbols = []
        unknown_symbols = []
        
        for symbol in symbols:
            if symbol in self.language.symbols:
                known_symbols.append(symbol)
            else:
                unknown_symbols.append(symbol)
                # Maybe learn it
                if random.random() < 0.3:  # Learning rate
                    # Infer meaning from context
                    inferred_meaning = self._infer_symbol_meaning(symbol, sender)
                    self.language.symbols[symbol] = inferred_meaning
                    
        if len(known_symbols) > len(unknown_symbols):
            interpretation["understood"] = True
            interpretation["inferred_meaning"] = "partial_understanding"
            interpretation["emotional_response"] = Emotion.CURIOUS
        elif known_symbols:
            interpretation["understood"] = False
            interpretation["inferred_meaning"] = "confusion"
            interpretation["emotional_response"] = Emotion.CONFUSED
        else:
            interpretation["understood"] = False
            interpretation["inferred_meaning"] = "complete_mystery"
            interpretation["emotional_response"] = Emotion.FEARFUL if random.random() < 0.3 else Emotion.CURIOUS
            
        # Update emotional state
        self.emotional_state.update(interpretation["emotional_response"])
        
        # Store in memory
        self.memory.add({
            "type": "received_communication",
            "from": sender.id,
            "utterance": utterance,
            "interpretation": interpretation
        })
        
        return interpretation
        
    def _infer_symbol_meaning(self, symbol: str, sender: 'Agent') -> np.ndarray:
        """Try to infer meaning of unknown symbol from context"""
        # Look at sender's emotional state and recent actions
        context_vector = np.zeros(8)
        
        # Encode sender's emotion
        emotion_encoding = hash(sender.emotional_state.current_emotion.value) % 8
        context_vector[emotion_encoding] = 1.0
        
        # Add randomness for evolution
        context_vector += np.random.normal(0, 0.2, 8)
        
        return context_vector / np.linalg.norm(context_vector)
        
    def create_belief(self, experience: Dict) -> Optional[str]:
        """Form new beliefs from experiences"""
        belief_text = None
        
        if experience["type"] == "near_death":
            # Create belief about mortality
            self.beliefs.add_belief("mortality_awareness", 0.8)
            belief_text = f"{self.name} understands: all_beings_fade"
            
        elif experience["type"] == "successful_cooperation":
            # Create belief about cooperation
            self.beliefs.add_belief("cooperation_good", 0.7)
            belief_text = f"{self.name} learns: together_stronger"
            
        elif experience["type"] == "resource_discovery":
            # Create belief about the world
            resource_type = experience["resource_type"]
            self.beliefs.add_belief(f"{resource_type}_location_pattern", 0.6)
            belief_text = f"{self.name} notices: {resource_type}_follows_pattern"
            
        elif experience["type"] == "repeated_symbol":
            # Create linguistic belief
            symbol = experience["symbol"]
            self.beliefs.add_belief(f"symbol_{symbol}_important", 0.5)
            belief_text = f"{self.name} feels: {symbol}_has_power"
            
        if belief_text:
            self.memory.add({
                "type": "belief_formation",
                "belief": belief_text,
                "trigger": experience
            })
            
        return belief_text
        
    def contemplate_existence(self) -> Dict:
        """Deep contemplation that might lead to myths or philosophy"""
        contemplation = {
            "type": None,
            "content": None,
            "resulted_in": None
        }
        
        # Different types of contemplation based on personality
        contemplation_type = np.random.choice([
            "origin", "purpose", "other_beings", "the_beyond", "patterns", "death"
        ], p=self._get_contemplation_weights())
        
        if contemplation_type == "origin":
            # Create origin myth
            myth = self._create_origin_myth()
            self.beliefs.add_myth(myth)
            contemplation = {
                "type": "myth_creation",
                "content": myth,
                "resulted_in": "new_origin_belief"
            }
            
        elif contemplation_type == "purpose":
            # Question existence
            if random.random() < 0.3:
                # Create new value
                value = random.choice(["harmony", "growth", "knowledge", "strength", "beauty"])
                self.beliefs.add_value(value, random.uniform(0.5, 1.0))
                contemplation = {
                    "type": "value_discovery",
                    "content": f"purpose_found_in_{value}",
                    "resulted_in": f"new_value_{value}"
                }
                
        elif contemplation_type == "the_beyond":
            # Sense something greater (the player/creator)
            if random.random() < 0.1:  # Rare insight
                self.beliefs.add_belief("the_watcher_exists", 0.3)
                contemplation = {
                    "type": "transcendent_awareness",
                    "content": "sense_of_being_observed",
                    "resulted_in": "awareness_of_watcher"
                }
                
        self.memory.add({
            "type": "contemplation",
            "details": contemplation,
            "timestamp": time.time()
        })
        
        return contemplation
        
    def _get_contemplation_weights(self) -> np.ndarray:
        """Get probability weights for contemplation types based on personality"""
        weights = np.ones(6) * 0.15
        
        # Adjust based on personality vector
        if self.personality_vector[4] > 0.6:  # Spiritual
            weights[3] += 0.2  # the_beyond
            weights[0] += 0.1  # origin
            
        if self.personality_vector[2] > 0.6:  # Creative
            weights[0] += 0.2  # origin (myths)
            
        if self.personality_vector[0] > 0.6:  # Curious
            weights[4] += 0.2  # patterns
            
        return weights / weights.sum()
        
    def _create_origin_myth(self) -> str:
        """Generate an origin myth based on experiences"""
        elements = []
        
        # Check memories for inspiration
        recent_memories = self.memory.get_recent(20)
        
        # Look for patterns
        if any(m.get("type") == "resource_discovery" for m in recent_memories):
            elements.append("abundance_from_void")
            
        if any(m.get("emotion") == "lonely" for m in recent_memories):
            elements.append("first_being_alone")
            
        if self.language.symbols:
            first_symbol = list(self.language.symbols.keys())[0]
            elements.append(f"first_word_{first_symbol}")
            
        # Create myth
        if not elements:
            elements = ["darkness_then_spark", "movement_began"]
            
        myth = f"In_beginning: {' '.join(elements)}. We_emerged."
        
        return myth
        
    def update(self, world: 'SimulationWorld'):
        """Update agent's state"""
        # Physical decay
        self.physical_state.decay()
        
        # Process recent experiences
        recent_memories = self.memory.get_recent(5)
        for memory in recent_memories:
            if memory.get("type") == "perception":
                # Check for patterns
                if self._detect_pattern(memory):
                    experience = {
                        "type": "pattern_recognition",
                        "pattern": "resource_regularity"
                    }
                    self.create_belief(experience)
                    
        # Emotional processing
        self._process_emotions()
        
        # Cultural evolution
        if random.random() < 0.01:  # Rare cultural innovation
            self._create_cultural_practice()
            
    def _detect_pattern(self, memory: Dict) -> bool:
        """Simple pattern detection in memories"""
        # This is simplified - could be much more complex
        similar_memories = self.memory.search(memory.get("type", ""))
        return len(similar_memories) > 5
        
    def _process_emotions(self):
        """Process and update emotional state based on recent events"""
        recent = self.memory.get_recent(3)
        
        if not recent:
            return
            
        # Simple emotional rules
        if any("attack" in str(m) for m in recent):
            self.emotional_state.update(Emotion.FEARFUL, 0.8)
        elif any("communication" in str(m) for m in recent):
            self.emotional_state.update(Emotion.HAPPY, 0.6)
        elif self.physical_state.energy < 0.3:
            self.emotional_state.update(Emotion.DESPERATE, 0.9)
            
    def _create_cultural_practice(self):
        """Innovate a new cultural practice"""
        practices = [
            "greeting_ritual",
            "resource_sharing_ceremony",
            "death_remembrance",
            "symbol_meditation",
            "movement_dance",
            "teaching_circle"
        ]
        
        if self.cultural_practices:
            # Modify existing practice
            base_practice = random.choice(list(self.cultural_practices))
            new_practice = f"{base_practice}_evolved_{len(self.cultural_practices)}"
        else:
            # Create new practice
            new_practice = random.choice(practices)
            
        self.cultural_practices.add(new_practice)
        
        # Try to teach others
        self.pending_actions.append({
            "action": Action.TEACH,
            "content": new_practice
        })
        
    def reproduce(self, partner: 'Agent', world: 'SimulationWorld') -> Optional['Agent']:
        """Create offspring with trait mixing and mutation"""
        if self.physical_state.energy < 0.5 or partner.physical_state.energy < 0.5:
            return None
            
        # Create offspring
        child_id = f"{self.id[:4]}_{partner.id[:4]}_{world.time}"
        child_pos = self.physical_state.position
        
        # Mix personalities with mutation
        child_personality = (self.personality_vector + partner.personality_vector) / 2
        child_personality += np.random.normal(0, 0.1, len(child_personality))
        child_personality = np.clip(child_personality, 0, 1)
        
        child = Agent(child_id, child_pos, self.model_config)
        child.personality_vector = child_personality
        
        # Inherit some language
        if self.language.symbols and partner.language.symbols:
            # Take random symbols from both parents
            parent_symbols = list(self.language.symbols.items()) + list(partner.language.symbols.items())
            num_inherit = min(5, len(parent_symbols))
            inherited = random.sample(parent_symbols, num_inherit)
            
            for symbol, meaning in inherited:
                # Maybe mutate
                if random.random() < world.config.language_mutation_rate:
                    symbol = self.language.mutate_symbol(symbol)
                child.language.symbols[symbol] = meaning
                
        # Inherit some beliefs
        if random.random() < 0.7:
            parent_beliefs = list(self.beliefs.beliefs.keys()) + list(partner.beliefs.beliefs.keys())
            if parent_beliefs:
                inherited_belief = random.choice(parent_beliefs)
                child.beliefs.add_belief(inherited_belief, 0.3)
                
        # Energy cost
        self.physical_state.energy *= 0.7
        partner.physical_state.energy *= 0.7
        
        return child
        
    def to_dict(self) -> Dict:
        """Serialize agent state"""
        return {
            "id": self.id,
            "name": self.name,
            "position": self.physical_state.position,
            "energy": self.physical_state.energy,
            "health": self.physical_state.health,
            "age": self.physical_state.age,
            "emotion": self.emotional_state.current_emotion.value,
            "beliefs": list(self.beliefs.beliefs.keys())[:5],
            "language_symbols": len(self.language.symbols),
            "relationships": len(self.relationships),
            "cultural_practices": list(self.cultural_practices)
        }

# Resource Manager
class ResourceManager:
    def __init__(self, world_size: Tuple[int, int]):
        self.world_size = world_size
        self.resources = {}  # position -> resource
        self.resource_patterns = {}  # Emergent patterns
        
    def spawn_resources(self, spawn_rate: float):
        """Spawn new resources in the world"""
        for _ in range(int(self.world_size[0] * self.world_size[1] * spawn_rate)):
            pos = (random.randint(0, self.world_size[0]-1), 
                   random.randint(0, self.world_size[1]-1))
            
            if pos not in self.resources:
                resource_type = random.choice(list(ResourceType))
                self.resources[pos] = {
                    "type": resource_type,
                    "amount": random.uniform(0.5, 1.0),
                    "position": pos
                }
                
    def gather(self, position: Tuple[int, int], amount: float = 0.1) -> Optional[Dict]:
        """Gather resource at position"""
        if position in self.resources:
            resource = self.resources[position]
            gathered = min(amount, resource["amount"])
            resource["amount"] -= gathered
            
            if resource["amount"] <= 0:
                del self.resources[position]
                
            return {
                "type": resource["type"],
                "amount": gathered
            }
        return None
        
    def get_nearby(self, position: Tuple[int, int], radius: int = 3) -> List[Dict]:
        """Get resources within radius of position"""
        x, y = position
        nearby = []
        
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                check_pos = (x + dx, y + dy)
                if check_pos in self.resources:
                    nearby.append(self.resources[check_pos].copy())
                    
        return nearby

# Event System
class Event:
    def __init__(self, event_type: str, data: Dict, position: Optional[Tuple[int, int]] = None):
        self.type = event_type
        self.data = data
        self.position = position
        self.timestamp = time.time()
        self.id = f"{event_type}_{self.timestamp}_{random.randint(1000, 9999)}"
        
class EventQueue:
    def __init__(self):
        self.events = deque()
        self.event_history = []
        self.event_patterns = defaultdict(int)
        
    def add(self, event: Event):
        self.events.append(event)
        self.event_history.append(event)
        self.event_patterns[event.type] += 1
        
    def process(self) -> List[Event]:
        """Get and clear current events"""
        current = list(self.events)
        self.events.clear()
        return current
        
    def get_recent(self, n: int = 10) -> List[Event]:
        """Get recent events from history"""
        return self.event_history[-n:]

# Main Simulation World
class SimulationWorld:
    def __init__(self, config: SimulationConfig = None):
        self.config = config or SimulationConfig()
        self.size = self.config.world_size
        self.grid = np.zeros(self.size)
        self.agents = {}
        self.resources = ResourceManager(self.size)
        self.time = 0
        self.events = EventQueue()
        self.cultures = {}  # Track emerging cultures
        self.languages = {}  # Track language families
        self.myths = []  # Collective myths
        self.conflicts = []  # Track conflicts
        self.alliances = []  # Track alliances
        
        # Initialize
        self._spawn_initial_agents()
        self._spawn_initial_resources()
        
        # Logging
        self.logger = self._setup_logging()
        
    def _setup_logging(self):
        """Setup logging for the simulation"""
        logger = logging.getLogger("ANIMA")
        logger.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        # File handler
        fh = logging.FileHandler('anima_simulation.log')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
        return logger
        
    def _spawn_initial_agents(self):
        """Create initial population"""
        for i in range(self.config.initial_agents):
            pos = (random.randint(0, self.size[0]-1), 
                   random.randint(0, self.size[1]-1))
            
            agent_id = f"agent_{i:04d}"
            agent = Agent(agent_id, pos, {"use_llm": False})  # Start without LLM
            self.agents[agent_id] = agent
            
            self.logger.info(f"Spawned {agent.name} at {pos}")
            
    def _spawn_initial_resources(self):
        """Create initial resources"""
        self.resources.spawn_resources(self.config.resource_spawn_rate * 10)  # More initial resources
        
    def tick(self):
        """Process one simulation step"""
        self.logger.debug(f"=== TICK {self.time} ===")
        
        # 1. Agent perception phase
        perceptions = {}
        for agent_id, agent in self.agents.items():
            if agent.physical_state.is_alive():
                perceptions[agent_id] = agent.perceive(self)
                
        # 2. Agent thinking phase
        decisions = {}
        for agent_id, agent in self.agents.items():
            if agent.physical_state.is_alive():
                # Use async thinking in real implementation
                decision = agent._simulate_thinking(perceptions[agent_id])
                decisions[agent_id] = decision
                
        # 3. Action execution phase
        self.process_agent_actions(decisions)
        
        # 4. Environmental updates
        self.update_environment()
        
        # 5. Cultural emergence check
        self.check_cultural_emergence()
        
        # 6. Process events
        current_events = self.events.process()
        for event in current_events:
            self.logger.info(f"Event: {event.type} - {event.data}")
            
        # 7. Chronicle major events
        if self.time % 100 == 0:
            self.chronicle_history()
            
        self.time += 1
        
    def process_agent_actions(self, decisions: Dict[str, Dict]):
        """Execute agent decisions"""
        for agent_id, decision in decisions.items():
            agent = self.agents[agent_id]
            action = decision.get("action")
            
            if action == Action.MOVE:
                self._handle_move(agent, decision.get("target"))
                
            elif action == Action.GATHER:
                self._handle_gather(agent, decision.get("target"))
                
            elif action == Action.COMMUNICATE:
                target_id = decision.get("target")
                if target_id in self.agents:
                    self._handle_communication(agent, self.agents[target_id])
                    
            elif action == Action.CREATE_SYMBOL:
                self._handle_symbol_creation(agent)
                
            elif action == Action.CONTEMPLATE:
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
                    }))
                    
            elif action == Action.REPRODUCE:
                partner_id = decision.get("target")
                if partner_id in self.agents:
                    self._handle_reproduction(agent, self.agents[partner_id])
                    
            elif action == Action.TEACH:
                self._handle_teaching(agent)
                
    def _handle_move(self, agent: Agent, target_pos: Optional[Tuple[int, int]]):
        """Handle agent movement"""
        if target_pos is None:
            return
            
        x, y = target_pos
        if 0 <= x < self.size[0] and 0 <= y < self.size[1]:
            old_pos = agent.physical_state.position
            agent.physical_state.position = target_pos
            agent.physical_state.energy *= 0.99  # Movement cost
            
            self.logger.debug(f"{agent.name} moved from {old_pos} to {target_pos}")
            
    def _handle_gather(self, agent: Agent, target_pos: Optional[Tuple[int, int]]):
        """Handle resource gathering"""
        if target_pos is None:
            target_pos = agent.physical_state.position
            
        resource = self.resources.gather(target_pos)
        if resource:
            agent.physical_state.carrying[resource["type"]] += resource["amount"]
            
            # Restore energy if food
            if resource["type"] == ResourceType.FOOD:
                agent.physical_state.energy = min(1.0, agent.physical_state.energy + 0.2)
                
            # Trigger belief formation
            agent.create_belief({
                "type": "resource_discovery",
                "resource_type": resource["type"].value
            })
            
            self.logger.info(f"{agent.name} gathered {resource['amount']:.2f} {resource['type'].value}")
            
    def _handle_communication(self, sender: Agent, receiver: Agent):
        """Handle communication between agents"""
        # Sender creates message
        message = sender.communicate("share_feeling", receiver)
        
        # Receiver interprets
        interpretation = receiver.interpret_communication(message["utterance"], sender)
        
        # Log communication event
        self.events.add(Event("communication", {
            "from": sender.name,
            "to": receiver.name,
            "utterance": message["utterance"],
            "understood": interpretation["understood"]
        }))
        
        # Track language evolution
        self._track_language_evolution(sender, receiver, message["utterance"])
        
    def _handle_symbol_creation(self, agent: Agent):
        """Handle creation of new symbols"""
        # Agent creates symbol for current feeling/thought
        meaning_vector = np.random.rand(8)
        symbol = agent.language.create_symbol(meaning_vector)
        
        self.events.add(Event("symbol_created", {
            "creator": agent.name,
            "symbol": symbol,
            "time": self.time
        }))
        
        self.logger.info(f"{agent.name} created symbol: {symbol}")
        
    def _handle_reproduction(self, parent1: Agent, parent2: Agent):
        """Handle agent reproduction"""
        child = parent1.reproduce(parent2, self)
        
        if child:
            self.agents[child.id] = child
            
            self.events.add(Event("birth", {
                "child": child.name,
                "parents": [parent1.name, parent2.name],
                "time": self.time
            }))
            
            self.logger.info(f"New agent born: {child.name} from {parent1.name} and {parent2.name}")
            
    def _handle_teaching(self, teacher: Agent):
        """Handle cultural transmission"""
        # Find nearby agents
        nearby = self.get_nearby_agents(teacher.physical_state.position, radius=3)
        
        if nearby and teacher.cultural_practices:
            practice = random.choice(list(teacher.cultural_practices))
            
            for agent_data in nearby:
                student = self.agents[agent_data["id"]]
                if random.random() < 0.5:  # Learning probability
                    student.cultural_practices.add(practice)
                    
                    self.logger.info(f"{teacher.name} taught {practice} to {student.name}")
                    
    def get_nearby_agents(self, position: Tuple[int, int], radius: int = 5) -> List[Dict]:
        """Get agents within radius of position"""
        x, y = position
        nearby = []
        
        for agent_id, agent in self.agents.items():
            ax, ay = agent.physical_state.position
            distance = abs(ax - x) + abs(ay - y)  # Manhattan distance
            
            if distance <= radius and agent.physical_state.is_alive():
                nearby.append({
                    "id": agent_id,
                    "name": agent.name,
                    "position": agent.physical_state.position,
                    "emotion": agent.emotional_state.current_emotion.value
                })
                
        return nearby
        
    def get_nearby_resources(self, position: Tuple[int, int], radius: int = 3) -> List[Dict]:
        """Get resources near position"""
        return self.resources.get_nearby(position, radius)
        
    def get_recent_events(self, n: int = 5) -> List[Dict]:
        """Get recent world events"""
        return [{"type": e.type, "data": e.data} for e in self.events.get_recent(n)]
        
    def update_environment(self):
        """Update world state"""
        # Spawn new resources
        self.resources.spawn_resources(self.config.resource_spawn_rate)
        
        # Update agents
        dead_agents = []
        for agent_id, agent in self.agents.items():
            agent.update(self)
            
            # Check death
            if not agent.physical_state.is_alive():
                dead_agents.append(agent_id)
                
        # Process deaths
        for agent_id in dead_agents:
            agent = self.agents[agent_id]
            self.events.add(Event("death", {
                "agent": agent.name,
                "age": agent.physical_state.age,
                "legacy": {
                    "symbols_created": len(agent.language.symbols),
                    "beliefs": list(agent.beliefs.beliefs.keys()),
                    "cultural_practices": list(agent.cultural_practices)
                }
            }))
            
            self.logger.info(f"{agent.name} has died at age {agent.physical_state.age}")
            del self.agents[agent_id]
            
    def check_cultural_emergence(self):
        """Detect emerging cultural patterns"""
        if self.time % 50 == 0:  # Check periodically
            # Find shared practices
            practice_counts = defaultdict(int)
            for agent in self.agents.values():
                for practice in agent.cultural_practices:
                    practice_counts[practice] += 1
                    
            # Identify dominant practices
            for practice, count in practice_counts.items():
                if count > len(self.agents) * 0.3:  # 30% adoption
                    if practice not in self.cultures:
                        self.cultures[practice] = {
                            "name": practice,
                            "founded": self.time,
                            "adherents": count
                        }
                        
                        self.events.add(Event("culture_emerged", {
                            "culture": practice,
                            "adherents": count
                        }))
                        
                        self.logger.info(f"New culture emerged: {practice} with {count} adherents")
                        
    def _track_language_evolution(self, sender: Agent, receiver: Agent, utterance: str):
        """Track language family evolution"""
        # Simple language family tracking
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
                
    def chronicle_history(self):
        """Create historical record of major events"""
        chronicle = {
            "time": self.time,
            "population": len(self.agents),
            "cultures": len(self.cultures),
            "languages": len(self.languages),
            "myths": len(self.myths),
            "major_events": []
        }
        
        # Find significant events
        event_counts = defaultdict(int)
        for event in self.events.event_history[-1000:]:  # Last 1000 events
            event_counts[event.type] += 1
            
        # Report unusual events
        for event_type, count in event_counts.items():
            expected = self.events.event_patterns.get(event_type, 0) / max(1, len(self.events.event_history))
            if count > expected * 1000 * 1.5:  # 50% more than expected
                chronicle["major_events"].append({
                    "type": event_type,
                    "frequency": count,
                    "significance": "unusual_activity"
                })
                
        self.logger.info(f"=== CHRONICLE at time {self.time} ===")
        self.logger.info(f"Population: {chronicle['population']}")
        self.logger.info(f"Cultures: {chronicle['cultures']}")
        self.logger.info(f"Language symbols: {chronicle['languages']}")
        self.logger.info(f"Myths: {chronicle['myths']}")
        
        # Sample some myths
        if self.myths:
            recent_myth = self.myths[-1]
            self.logger.info(f"Recent myth by {recent_myth['creator']}: {recent_myth['myth']}")
            
    def get_world_state(self) -> Dict:
        """Get current world state for visualization"""
        return {
            "time": self.time,
            "agents": [agent.to_dict() for agent in self.agents.values()],
            "resources": list(self.resources.resources.values()),
            "cultures": self.cultures,
            "languages": self.languages,
            "myths": self.myths[-5:],  # Last 5 myths
            "events": [{"type": e.type, "data": e.data} for e in self.events.get_recent(10)]
        }
        
    async def run(self, num_ticks: int = 1000):
        """Run simulation for specified ticks"""
        self.logger.info(f"Starting ANIMA simulation for {num_ticks} ticks")
        
        for i in range(num_ticks):
            self.tick()
            
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
        report = f"""
=== ANIMA FINAL REPORT ===
Time Elapsed: {self.time}
Final Population: {len(self.agents)}
Cultures Emerged: {len(self.cultures)}
Unique Language Symbols: {len(self.languages)}
Myths Created: {len(self.myths)}

Cultural Analysis:
"""
        for culture, data in self.cultures.items():
            report += f"\n- {culture}: Founded at time {data['founded']}, {data['adherents']} adherents"
            
        report += "\n\nLanguage Analysis:"
        # Find most common symbols
        symbol_usage = defaultdict(int)
        for agent in self.agents.values():
            for symbol in agent.language.symbols:
                symbol_usage[symbol] += 1
                
        top_symbols = sorted(symbol_usage.items(), key=lambda x: x[1], reverse=True)[:10]
        for symbol, count in top_symbols:
            report += f"\n- '{symbol}': used by {count} agents"
            
        report += "\n\nPhilosophical Developments:"
        # Aggregate beliefs
        belief_counts = defaultdict(int)
        for agent in self.agents.values():
            for belief in agent.beliefs.beliefs:
                belief_counts[belief] += 1
                
        top_beliefs = sorted(belief_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        for belief, count in top_beliefs:
            report += f"\n- {belief}: held by {count} agents"
            
        # Check for transcendent awareness
        transcendent_agents = [a for a in self.agents.values() 
                              if "the_watcher_exists" in a.beliefs.beliefs]
        if transcendent_agents:
            report += f"\n\n{len(transcendent_agents)} agents have become aware of 'The Watcher'"
            
        self.logger.info(report)
        
        with open("anima_final_report.txt", "w") as f:
            f.write(report)

# Main execution
if __name__ == "__main__":
    # Create and run simulation
    config = SimulationConfig(
        world_size=(30, 30),
        initial_agents=15,
        resource_spawn_rate=0.05,
        time_per_tick=0.1
    )
    
    world = SimulationWorld(config)
    
    # Run simulation
    asyncio.run(world.run(num_ticks=1000))