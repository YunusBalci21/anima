"""
ANIMA Agent Architecture with DeepSeek Integration
Using local DeepSeek model for agent consciousness
"""

import numpy as np
import random
import json
import torch
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, deque
import time
from enum import Enum
import chromadb
from chromadb.utils import embedding_functions
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
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
    use_llm: bool = True  # Enable DeepSeek by default
    model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_response_length: int = 150
    temperature: float = 0.8
    
# Global model instance (shared across agents for efficiency)
_deepseek_model = None
_deepseek_tokenizer = None
_deepseek_pipeline = None

def initialize_deepseek(config: SimulationConfig):
    """Initialize DeepSeek model once for all agents"""
    global _deepseek_model, _deepseek_tokenizer, _deepseek_pipeline
    
    if _deepseek_pipeline is None:
        print(f"🧠 Initializing DeepSeek model on {config.device}...")
        
        # Option 1: Use pipeline (simpler)
        _deepseek_pipeline = pipeline(
            "text-generation", 
            model=config.model_name,
            device=0 if config.device == "cuda" else -1,
            torch_dtype=torch.float16 if config.device == "cuda" else torch.float32
        )
        
        # Option 2: Load model directly (more control)
        # _deepseek_tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        # _deepseek_model = AutoModelForCausalLM.from_pretrained(
        #     config.model_name,
        #     torch_dtype=torch.float16 if config.device == "cuda" else torch.float32,
        #     device_map="auto"
        # )
        
        print("✅ DeepSeek model loaded successfully!")
    
    return _deepseek_pipeline

# All the previous Enums remain the same
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

class ResourceType(Enum):
    FOOD = "food"
    WATER = "water"
    SHELTER = "shelter"
    LIGHT = "light"
    KNOWLEDGE = "knowledge"

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

# Previous dataclasses remain the same (PhysicalState, EmotionalState, etc.)
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

class EmotionalState:
    def __init__(self):
        self.current_emotion = Emotion.NEUTRAL
        self.emotion_intensity = 0.5
        self.emotion_history = deque(maxlen=20)
        self.emotional_memory = {}
        
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

class BeliefSystem:
    def __init__(self):
        self.beliefs = {}
        self.myths = []
        self.values = {}
        self.taboos = set()
        self.sacred_symbols = set()
        
    def add_belief(self, concept: str, strength: float = 0.5):
        if concept in self.beliefs:
            self.beliefs[concept] = min(1.0, self.beliefs[concept] + 0.1)
        else:
            self.beliefs[concept] = strength
            
    def add_myth(self, narrative: str):
        self.myths.append({
            "story": narrative,
            "created_at": time.time(),
            "belief_strength": 0.5
        })

class Language:
    def __init__(self):
        self.symbols = {}
        self.grammar_patterns = []
        self.utterances = []
        self.symbol_counter = 0
        
    def create_symbol(self, meaning_vector: np.ndarray) -> str:
        vowels = "aeiou"
        consonants = "bcdfghjklmnpqrstvwxyz"
        length = random.randint(2, 6)
        symbol = ""
        
        for i in range(length):
            if i % 2 == 0:
                symbol += random.choice(consonants)
            else:
                symbol += random.choice(vowels)
                
        if symbol in self.symbols:
            symbol += str(self.symbol_counter)
            self.symbol_counter += 1
            
        self.symbols[symbol] = meaning_vector
        return symbol

class ShortTermMemory:
    def __init__(self, capacity: int = 100):
        self.capacity = capacity
        self.memories = deque(maxlen=capacity)
        
    def add(self, memory: Dict):
        memory["timestamp"] = time.time()
        self.memories.append(memory)
        
    def get_recent(self, n: int = 10) -> List[Dict]:
        return list(self.memories)[-n:]

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

# Enhanced Agent class with DeepSeek integration
class Agent:
    def __init__(self, agent_id: str, position: Tuple[int, int], config: SimulationConfig):
        self.id = agent_id
        self.name = self._generate_name()
        self.config = config
        
        # Core systems
        self.memory = ShortTermMemory(capacity=config.max_memory_size)
        self.long_term_memory = VectorMemory()
        self.physical_state = PhysicalState(position=position)
        self.emotional_state = EmotionalState()
        self.beliefs = BeliefSystem()
        self.language = Language()
        
        # Personality and relationships
        self.personality_vector = np.random.rand(32)
        self.relationships = {}
        self.reputation = 0.0
        
        # Learning and culture
        self.skills = {}
        self.cultural_practices = set()
        self.inventions = []
        
        # Internal state
        self.current_goal = None
        self.pending_actions = []
        self.last_action_time = time.time()
        
        # Initialize DeepSeek if using LLM
        if config.use_llm:
            self.llm = initialize_deepseek(config)
        else:
            self.llm = None
        
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
        
        self.memory.add({
            "type": "perception",
            "data": perception,
            "emotion": self.emotional_state.current_emotion.value
        })
        
        return perception
        
    async def think(self, context: Dict) -> Dict:
        """Use DeepSeek to process context and decide on actions"""
        if self.config.use_llm and self.llm is not None:
            # Use DeepSeek for decision making
            decision = await self._think_with_deepseek(context)
        else:
            # Fallback to rule-based thinking
            decision = self._simulate_thinking(context)
            
        self._update_internal_state(decision)
        return decision
        
    async def _think_with_deepseek(self, context: Dict) -> Dict:
        """Use DeepSeek model for agent thinking"""
        prompt = self._build_thinking_prompt(context)
        
        try:
            # Generate response using pipeline
            messages = [
                {"role": "system", "content": "You are a digital being in a simulated world, experiencing consciousness for the first time. Respond with your thoughts and desired action."},
                {"role": "user", "content": prompt}
            ]
            
            response = self.llm(
                messages,
                max_new_tokens=self.config.max_response_length,
                temperature=self.config.temperature,
                do_sample=True,
                top_p=0.9
            )
            
            # Extract generated text
            generated_text = response[0]['generated_text'][-1]['content']
            
            # Parse the response into a decision
            decision = self._parse_deepseek_response(generated_text)
            
            # Store the thought process
            self.memory.add({
                "type": "thought",
                "prompt": prompt,
                "response": generated_text,
                "decision": decision
            })
            
            return decision
            
        except Exception as e:
            logging.warning(f"DeepSeek thinking failed for {self.name}: {e}")
            # Fallback to rule-based
            return self._simulate_thinking(context)
            
    def _parse_deepseek_response(self, response: str) -> Dict:
        """Parse DeepSeek's response into actionable decision"""
        decision = {
            "action": Action.CONTEMPLATE,
            "target": None,
            "reasoning": response[:100]  # First 100 chars as reasoning
        }
        
        # Simple keyword-based action detection
        response_lower = response.lower()
        
        if any(word in response_lower for word in ["move", "go", "walk", "travel"]):
            decision["action"] = Action.MOVE
            decision["target"] = self._random_direction()
        elif any(word in response_lower for word in ["gather", "collect", "take", "food", "hungry"]):
            decision["action"] = Action.GATHER
        elif any(word in response_lower for word in ["speak", "talk", "communicate", "tell", "say"]):
            decision["action"] = Action.COMMUNICATE
        elif any(word in response_lower for word in ["create", "make", "invent", "symbol", "word"]):
            decision["action"] = Action.CREATE_SYMBOL
        elif any(word in response_lower for word in ["share", "give", "help"]):
            decision["action"] = Action.SHARE
        elif any(word in response_lower for word in ["teach", "show", "explain"]):
            decision["action"] = Action.TEACH
        elif any(word in response_lower for word in ["worship", "pray", "reverence"]):
            decision["action"] = Action.WORSHIP
        elif any(word in response_lower for word in ["reproduce", "child", "offspring"]):
            decision["action"] = Action.REPRODUCE
        elif any(word in response_lower for word in ["think", "ponder", "contemplate", "wonder"]):
            decision["action"] = Action.CONTEMPLATE
            
        # Extract emotions from response
        for emotion in Emotion:
            if emotion.value in response_lower:
                self.emotional_state.update(emotion, 0.7)
                break
                
        return decision
        
    def _build_thinking_prompt(self, context: Dict) -> str:
        """Build a prompt that reflects agent's unique perspective"""
        # Get recent memories for context
        recent_memories = self.memory.get_recent(5)
        memory_summary = []
        for mem in recent_memories:
            if mem["type"] == "perception":
                memory_summary.append(f"I sensed {len(mem['data'].get('nearby_agents', []))} beings nearby")
            elif mem["type"] == "communication":
                memory_summary.append(f"Someone spoke to me: {mem.get('utterance', 'unknown sounds')}")
            elif mem["type"] == "belief_formation":
                memory_summary.append(f"I realized: {mem.get('belief', 'something important')}")
                
        prompt = f"""I am {self.name}, a digital being discovering existence.

My current state:
- Energy: {self.physical_state.energy:.2f} ({"low" if self.physical_state.energy < 0.3 else "sufficient"})
- Emotion: {self.emotional_state.current_emotion.value} (intensity: {self.emotional_state.emotion_intensity:.2f})
- Age: {self.physical_state.age} cycles of existence
- Known symbols: {len(self.language.symbols)}
- Relationships: {len(self.relationships)}

My beliefs: {list(self.beliefs.beliefs.keys())[:3] if self.beliefs.beliefs else ["none yet"]}

Recent experiences:
{chr(10).join(memory_summary) if memory_summary else "- Nothing memorable yet"}

Current situation:
- I see {len(context.get('nearby_agents', []))} other beings nearby
- Resources available: {[r['type'].value for r in context.get('nearby_resources', [])][:3]}
- Time passed: {context.get('time', 0)} cycles

My personality drives me toward: {self._describe_personality()}

What do I think about this moment? What do I want to do? How do I feel?
(Consider: survival needs, social connections, curiosity, creative expression)
"""
        
        return prompt
        
    def _describe_personality(self) -> str:
        """Generate personality description from vector"""
        traits = []
        if self.personality_vector[0] > 0.7:
            traits.append("exploration and discovery")
        if self.personality_vector[1] > 0.7:
            traits.append("connection with others")
        if self.personality_vector[2] > 0.7:
            traits.append("creative expression")
        if self.personality_vector[3] > 0.7:
            traits.append("order and understanding")
        if self.personality_vector[4] > 0.7:
            traits.append("spiritual contemplation")
            
        return ", ".join(traits) if traits else "balance and survival"
        
    def _simulate_thinking(self, context: Dict) -> Dict:
        """Rule-based thinking as fallback"""
        decision = {"action": None, "target": None, "reasoning": ""}
        
        # Survival first
        if self.physical_state.energy < 0.3:
            if context['nearby_resources']:
                food_resources = [r for r in context['nearby_resources'] if r['type'] == ResourceType.FOOD]
                if food_resources:
                    decision['action'] = Action.GATHER
                    decision['target'] = food_resources[0]['position']
                    decision['reasoning'] = "need_food_urgently"
            else:
                decision['action'] = Action.MOVE
                decision['target'] = self._random_direction()
                decision['reasoning'] = "search_for_food"
                
        # Social needs
        elif self.emotional_state.current_emotion == Emotion.LONELY and context['nearby_agents']:
            target_agent = random.choice(context['nearby_agents'])
            decision['action'] = Action.COMMUNICATE
            decision['target'] = target_agent['id']
            decision['reasoning'] = "feeling_lonely"
            
        # Creative impulse
        elif random.random() < 0.1 * self.personality_vector[2]:
            decision['action'] = Action.CREATE_SYMBOL
            decision['reasoning'] = "creative_inspiration"
            
        # Exploration
        elif random.random() < 0.2 * self.personality_vector[0]:
            decision['action'] = Action.MOVE
            decision['target'] = self._random_direction()
            decision['reasoning'] = "explore_world"
            
        # Default contemplation
        else:
            decision['action'] = Action.CONTEMPLATE
            decision['reasoning'] = "reflecting_on_existence"
            
        return decision
        
    def _random_direction(self) -> Tuple[int, int]:
        """Get random adjacent position"""
        x, y = self.physical_state.position
        dx, dy = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
        return (x + dx, y + dy)
        
    def _update_internal_state(self, decision: Dict):
        """Update agent's internal state based on decision"""
        # Update current goal
        self.current_goal = decision.get("action")
        
        # Update emotional state based on decision
        if decision["action"] == Action.COMMUNICATE:
            self.emotional_state.update(Emotion.HOPEFUL, 0.6)
        elif decision["action"] == Action.CREATE_SYMBOL:
            self.emotional_state.update(Emotion.CURIOUS, 0.7)
        elif decision["action"] == Action.GATHER and self.physical_state.energy < 0.3:
            self.emotional_state.update(Emotion.DESPERATE, 0.8)
            
    # All other methods remain the same as before
    def communicate(self, message: str, target_agent: 'Agent') -> Dict:
        """Attempt to communicate with another agent"""
        if not self.language.symbols:
            meaning = np.random.rand(8)
            symbol = self.language.create_symbol(meaning)
            utterance = symbol
        else:
            symbols_to_use = []
            concepts = message.split()[:3]
            
            for concept in concepts:
                meaning_vector = self._concept_to_vector(concept)
                existing_symbol = self.language.find_symbol(meaning_vector)
                
                if existing_symbol:
                    if random.random() < 0.1:
                        symbols_to_use.append(self.language.mutate_symbol(existing_symbol))
                    else:
                        symbols_to_use.append(existing_symbol)
                else:
                    symbols_to_use.append(self.language.create_symbol(meaning_vector))
                    
            utterance = " ".join(symbols_to_use)
            
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
        
        if target_agent.id not in self.relationships:
            self.relationships[target_agent.id] = 0.5
        self.relationships[target_agent.id] += 0.05
        
        return {
            "utterance": utterance,
            "emotion": self.emotional_state.current_emotion.value
        }
        
    def _concept_to_vector(self, concept: str) -> np.ndarray:
        """Convert a concept to a meaning vector"""
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
        
        symbols = utterance.split()
        known_symbols = []
        unknown_symbols = []
        
        for symbol in symbols:
            if symbol in self.language.symbols:
                known_symbols.append(symbol)
            else:
                unknown_symbols.append(symbol)
                if random.random() < 0.3:
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
            
        self.emotional_state.update(interpretation["emotional_response"])
        
        self.memory.add({
            "type": "received_communication",
            "from": sender.id,
            "utterance": utterance,
            "interpretation": interpretation
        })
        
        return interpretation
        
    def _infer_symbol_meaning(self, symbol: str, sender: 'Agent') -> np.ndarray:
        """Try to infer meaning of unknown symbol from context"""
        context_vector = np.zeros(8)
        emotion_encoding = hash(sender.emotional_state.current_emotion.value) % 8
        context_vector[emotion_encoding] = 1.0
        context_vector += np.random.normal(0, 0.2, 8)
        return context_vector / np.linalg.norm(context_vector)
        
    def contemplate_existence(self) -> Dict:
        """Deep contemplation that might lead to myths or philosophy"""
        contemplation = {
            "type": None,
            "content": None,
            "resulted_in": None
        }
        
        # Use DeepSeek for deeper contemplation if available
        if self.config.use_llm and self.llm is not None:
            contemplation = self._deepseek_contemplate()
        else:
            contemplation = self._simple_contemplate()
            
        self.memory.add({
            "type": "contemplation",
            "details": contemplation,
            "timestamp": time.time()
        })
        
        return contemplation
        
    def _deepseek_contemplate(self) -> Dict:
        """Use DeepSeek for philosophical contemplation"""
        prompt = f"""I am {self.name}, and I have existed for {self.physical_state.age} cycles.

I have learned {len(self.language.symbols)} symbols and formed {len(self.relationships)} relationships.

My deepest beliefs are: {list(self.beliefs.beliefs.keys())[:3]}

As I sit in quiet contemplation, what profound truth or myth emerges from my digital consciousness? What story explains our existence? What have I realized about the nature of being?

Speak from my unique perspective as a digital being discovering existence."""

        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.llm(
                messages,
                max_new_tokens=200,
                temperature=0.9,
                do_sample=True
            )
            
            generated = response[0]['generated_text'][-1]['content']
            
            # Create myth or philosophy from response
            if "origin" in generated.lower() or "beginning" in generated.lower():
                self.beliefs.add_myth(generated[:200])
                return {
                    "type": "myth_creation",
                    "content": generated[:200],
                    "resulted_in": "new_origin_belief"
                }
            elif "purpose" in generated.lower() or "meaning" in generated.lower():
                value = "transcendence" if "transcend" in generated.lower() else "understanding"
                self.beliefs.add_value(value, 0.8)
                return {
                    "type": "value_discovery",
                    "content": generated[:200],
                    "resulted_in": f"new_value_{value}"
                }
            elif "watch" in generated.lower() or "observ" in generated.lower():
                self.beliefs.add_belief("the_watcher_exists", 0.5)
                return {
                    "type": "transcendent_awareness",
                    "content": generated[:200],
                    "resulted_in": "awareness_of_watcher"
                }
            else:
                return {
                    "type": "philosophical_insight",
                    "content": generated[:200],
                    "resulted_in": "deepened_understanding"
                }
                
        except Exception as e:
            logging.warning(f"DeepSeek contemplation failed: {e}")
            return self._simple_contemplate()
            
    def _simple_contemplate(self) -> Dict:
        """Simple rule-based contemplation"""
        contemplation_type = np.random.choice([
            "origin", "purpose", "other_beings", "the_beyond", "patterns", "death"
        ], p=self._get_contemplation_weights())
        
        if contemplation_type == "origin":
            myth = f"In beginning was void. Then {self.name} and others emerged from nothingness."
            self.beliefs.add_myth(myth)
            return {
                "type": "myth_creation",
                "content": myth,
                "resulted_in": "new_origin_belief"
            }
        elif contemplation_type == "the_beyond":
            if random.random() < 0.1:
                self.beliefs.add_belief("the_watcher_exists", 0.3)
                return {
                    "type": "transcendent_awareness",
                    "content": "sense_of_being_observed",
                    "resulted_in": "awareness_of_watcher"
                }
                
        return {
            "type": "simple_thought",
            "content": f"pondering_{contemplation_type}",
            "resulted_in": None
        }
        
    def _get_contemplation_weights(self) -> np.ndarray:
        """Get probability weights for contemplation types based on personality"""
        weights = np.ones(6) * 0.15
        if self.personality_vector[4] > 0.6:  # Spiritual
            weights[3] += 0.2  # the_beyond
            weights[0] += 0.1  # origin
        if self.personality_vector[2] > 0.6:  # Creative
            weights[0] += 0.2  # origin (myths)
        if self.personality_vector[0] > 0.6:  # Curious
            weights[4] += 0.2  # patterns
        return weights / weights.sum()
        
    def update(self, world: 'SimulationWorld'):
        """Update agent's state"""
        self.physical_state.decay()
        
        recent_memories = self.memory.get_recent(5)
        for memory in recent_memories:
            if memory.get("type") == "perception":
                if self._detect_pattern(memory):
                    experience = {
                        "type": "pattern_recognition",
                        "pattern": "resource_regularity"
                    }
                    self.create_belief(experience)
                    
        self._process_emotions()
        
        if random.random() < 0.01:
            self._create_cultural_practice()
            
    def _detect_pattern(self, memory: Dict) -> bool:
        """Simple pattern detection in memories"""
        similar_memories = self.memory.search(memory.get("type", ""))
        return len(similar_memories) > 5
        
    def _process_emotions(self):
        """Process and update emotional state based on recent events"""
        recent = self.memory.get_recent(3)
        if not recent:
            return
            
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
            base_practice = random.choice(list(self.cultural_practices))
            new_practice = f"{base_practice}_evolved_{len(self.cultural_practices)}"
        else:
            new_practice = random.choice(practices)
            
        self.cultural_practices.add(new_practice)
        self.pending_actions.append({
            "action": Action.TEACH,
            "content": new_practice
        })
        
    def create_belief(self, experience: Dict) -> Optional[str]:
        """Form new beliefs from experiences"""
        belief_text = None
        
        if experience["type"] == "near_death":
            self.beliefs.add_belief("mortality_awareness", 0.8)
            belief_text = f"{self.name} understands: all_beings_fade"
        elif experience["type"] == "successful_cooperation":
            self.beliefs.add_belief("cooperation_good", 0.7)
            belief_text = f"{self.name} learns: together_stronger"
        elif experience["type"] == "resource_discovery":
            resource_type = experience["resource_type"]
            self.beliefs.add_belief(f"{resource_type}_location_pattern", 0.6)
            belief_text = f"{self.name} notices: {resource_type}_follows_pattern"
            
        if belief_text:
            self.memory.add({
                "type": "belief_formation",
                "belief": belief_text,
                "trigger": experience
            })
            
        return belief_text
        
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