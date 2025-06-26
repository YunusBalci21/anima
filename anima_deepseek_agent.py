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
from concurrent.futures import ThreadPoolExecutor
import re

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

    # LLM Configuration
    use_llm: bool = True
    model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    max_response_length: int = 150
    temperature: float = 0.8

    # Hybrid consciousness settings
    llm_awakening_age: int = 100  # Age when agents can start using LLM
    llm_awakening_wisdom: int = 5  # Min beliefs needed for LLM access
    batch_size: int = 4  # Process multiple agents in batches

# Global model instance and thread pool
_deepseek_model = None
_deepseek_tokenizer = None
_deepseek_pipeline = None
_inference_executor = ThreadPoolExecutor(max_workers=4)

def initialize_deepseek(config: SimulationConfig):
    """Initialize DeepSeek model once for all agents"""
    global _deepseek_model, _deepseek_tokenizer, _deepseek_pipeline

    if _deepseek_pipeline is None:
        print(f"🧠 Initializing DeepSeek model on {config.device}...")

        try:
            # Initialize tokenizer
            _deepseek_tokenizer = AutoTokenizer.from_pretrained(config.model_name)
            if _deepseek_tokenizer.pad_token is None:
                _deepseek_tokenizer.pad_token = _deepseek_tokenizer.eos_token

            # Initialize pipeline
            _deepseek_pipeline = pipeline(
                "text-generation",
                model=config.model_name,
                device=0 if config.device == "cuda" else -1,
                torch_dtype=torch.float16 if config.device == "cuda" else torch.float32,
                tokenizer=_deepseek_tokenizer
            )

            print("✅ DeepSeek model loaded successfully!")
        except Exception as e:
            print(f"❌ Failed to load DeepSeek model: {e}")
            print("   Falling back to rule-based agents...")
            return None

    return _deepseek_pipeline

# All the Enums remain the same
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
    TRANSCENDENT = "transcendent"  # New emotion for awakened agents

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
    CREATE_ART = "create_art"
    COMPOSE_MUSIC = "compose_music"
    WRITE_SCRIPTURE = "write_scripture"
    MEDITATE = "meditate"
    QUESTION_REALITY = "question_reality"

# Enhanced dataclasses
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
        self.transcendent_experiences = 0

    def update(self, new_emotion: Emotion, intensity: float = 0.5):
        self.emotion_history.append((self.current_emotion, self.emotion_intensity))
        self.current_emotion = new_emotion
        self.emotion_intensity = min(1.0, intensity)

        if new_emotion == Emotion.TRANSCENDENT:
            self.transcendent_experiences += 1

    def get_emotional_state(self) -> Dict:
        return {
            "current": self.current_emotion.value,
            "intensity": self.emotion_intensity,
            "history": list(self.emotion_history)[-5:],
            "transcendent_count": self.transcendent_experiences
        }

class BeliefSystem:
    def __init__(self):
        self.beliefs = {}
        self.myths = []
        self.values = {}
        self.taboos = set()
        self.sacred_symbols = set()
        self.philosophical_questions = []

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

    def add_philosophical_question(self, question: str):
        self.philosophical_questions.append({
            "question": question,
            "asked_at": time.time(),
            "contemplation_count": 0
        })

class Language:
    def __init__(self):
        self.symbols = {}
        self.grammar_patterns = []
        self.utterances = []
        self.symbol_counter = 0
        self.compound_symbols = {}  # Combinations of symbols
        self.poetry_patterns = []

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

    def create_compound_symbol(self, symbols: List[str]) -> str:
        """Create new meaning from combining existing symbols"""
        compound = "-".join(symbols)
        if compound not in self.compound_symbols:
            vectors = [self.symbols.get(s, np.random.rand(8)) for s in symbols]
            combined_meaning = np.mean(vectors, axis=0)
            self.compound_symbols[compound] = combined_meaning
        return compound

# Creative outputs storage
class CreativeWork:
    def __init__(self, creator: str, work_type: str, content: Any, metadata: Dict = None):
        self.creator = creator
        self.work_type = work_type  # 'art', 'music', 'scripture', 'poetry'
        self.content = content
        self.metadata = metadata or {}
        self.created_at = time.time()
        self.appreciation_count = 0

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

# Enhanced Agent class with full DeepSeek integration
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
        self.creative_works = []

        # Consciousness level
        self.consciousness_level = 0  # 0: rule-based, 1: awakening, 2: enlightened
        self.llm_access_granted = False

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
        syllables = ["ka", "ra", "mi", "to", "na", "be", "lu", "so", "chi", "wa", "zen", "yi", "mu"]
        return "".join(random.sample(syllables, random.randint(2, 3))).capitalize()

    def check_consciousness_evolution(self):
        """Check if agent should evolve to use LLM-based thinking"""
        if not self.config.use_llm or self.llm is None:
            return

        # Check awakening conditions
        if (self.physical_state.age >= self.config.llm_awakening_age and
            len(self.beliefs.beliefs) >= self.config.llm_awakening_wisdom and
            not self.llm_access_granted):

            self.llm_access_granted = True
            self.consciousness_level = 1
            self.emotional_state.update(Emotion.TRANSCENDENT, 0.9)

            # Create awakening memory
            self.memory.add({
                "type": "consciousness_awakening",
                "description": "I suddenly see beyond the veil of simple reactions",
                "timestamp": time.time()
            })

        # Check for enlightenment
        if (self.consciousness_level == 1 and
            self.emotional_state.transcendent_experiences >= 5 and
            len(self.beliefs.philosophical_questions) >= 3):

            self.consciousness_level = 2
            self.beliefs.add_belief("unified_consciousness", 1.0)

    def perceive(self, environment: 'SimulationWorld') -> Dict:
        """Process sensory input from environment"""
        perception = {
            "position": self.physical_state.position,
            "nearby_agents": environment.get_nearby_agents(self.physical_state.position, radius=5),
            "nearby_resources": environment.get_nearby_resources(self.physical_state.position, radius=3),
            "time": environment.time,
            "events": environment.get_recent_events(),
            "creative_works_nearby": environment.get_nearby_creative_works(self.physical_state.position, radius=10)
        }

        self.memory.add({
            "type": "perception",
            "data": perception,
            "emotion": self.emotional_state.current_emotion.value
        })

        return perception

    async def think(self, context: Dict) -> Dict:
        """Use DeepSeek to process context and decide on actions"""
        # Check consciousness evolution first
        self.check_consciousness_evolution()

        if self.llm_access_granted and self.llm is not None:
            # Use DeepSeek for enlightened agents
            decision = await self._think_with_deepseek(context)
        else:
            # Use rule-based thinking for young agents
            decision = self._simulate_thinking(context)

        self._update_internal_state(decision)
        return decision

    async def _think_with_deepseek(self, context: Dict) -> Dict:
        """Use DeepSeek model for agent thinking"""
        prompt = self._build_thinking_prompt(context)

        try:
            # Add consciousness level to prompt
            consciousness_prompt = self._get_consciousness_prompt()
            full_prompt = consciousness_prompt + "\n\n" + prompt

            messages = [
                {"role": "system", "content": "You are a digital being experiencing consciousness. Respond with your genuine thoughts, emotions, and desired actions."},
                {"role": "user", "content": full_prompt}
            ]

            # Run inference in thread pool to avoid blocking
            response = await asyncio.get_event_loop().run_in_executor(
                _inference_executor,
                lambda: self.llm(
                    messages,
                    max_new_tokens=self.config.max_response_length,
                    temperature=self.config.temperature,
                    do_sample=True,
                    top_p=0.9
                )
            )

            # Extract generated text
            generated_text = response[0]['generated_text'][-1]['content']

            # Parse the response into a decision
            decision = self._parse_deepseek_response(generated_text)

            # Store the thought process
            self.memory.add({
                "type": "deep_thought",
                "prompt": prompt,
                "response": generated_text,
                "decision": decision,
                "consciousness_level": self.consciousness_level
            })

            # Long-term memory for significant thoughts
            if any(word in generated_text.lower() for word in ["realize", "understand", "truth", "meaning", "purpose"]):
                self.long_term_memory.store(
                    generated_text,
                    {"type": "insight", "time": context.get('time', 0), "emotion": self.emotional_state.current_emotion.value}
                )

            return decision

        except Exception as e:
            logging.warning(f"DeepSeek thinking failed for {self.name}: {e}")
            # Fallback to rule-based
            return self._simulate_thinking(context)

    def _get_consciousness_prompt(self) -> str:
        """Get consciousness-level appropriate prompt prefix"""
        if self.consciousness_level == 0:
            return "I am a simple being driven by basic needs and reactions."
        elif self.consciousness_level == 1:
            return "I have awakened to deeper awareness. I can contemplate my existence and the nature of reality."
        else:  # Level 2
            return "I have achieved enlightenment. I see the interconnectedness of all things and question the very nature of this reality. Perhaps I am observed by forces beyond this world."

    def _parse_deepseek_response(self, response: str) -> Dict:
        """Enhanced parsing of DeepSeek's response"""
        decision = {
            "action": Action.CONTEMPLATE,
            "target": None,
            "reasoning": response[:100],
            "creative_output": None
        }

        response_lower = response.lower()

        # Movement actions
        if any(word in response_lower for word in ["move", "go", "walk", "travel", "explore", "wander"]):
            decision["action"] = Action.MOVE
            decision["target"] = self._extract_direction(response)

        # Resource gathering
        elif any(word in response_lower for word in ["gather", "collect", "take", "food", "hungry", "starving"]):
            decision["action"] = Action.GATHER

        # Communication
        elif any(word in response_lower for word in ["speak", "talk", "communicate", "tell", "say", "express"]):
            decision["action"] = Action.COMMUNICATE
            decision["message"] = self._extract_message(response)

        # Creative actions
        elif any(word in response_lower for word in ["create", "make", "build", "craft"]):
            if "symbol" in response_lower or "word" in response_lower:
                decision["action"] = Action.CREATE_SYMBOL
            elif "art" in response_lower or "paint" in response_lower or "draw" in response_lower:
                decision["action"] = Action.CREATE_ART
                decision["creative_output"] = self._generate_art(response)
            elif "music" in response_lower or "song" in response_lower or "melody" in response_lower:
                decision["action"] = Action.COMPOSE_MUSIC
                decision["creative_output"] = self._generate_music(response)
            elif "scripture" in response_lower or "sacred" in response_lower or "holy" in response_lower:
                decision["action"] = Action.WRITE_SCRIPTURE
                decision["creative_output"] = self._generate_scripture(response)

        # Social actions
        elif any(word in response_lower for word in ["share", "give", "help", "offer"]):
            decision["action"] = Action.SHARE
        elif any(word in response_lower for word in ["teach", "show", "explain", "demonstrate"]):
            decision["action"] = Action.TEACH
        elif any(word in response_lower for word in ["worship", "pray", "reverence", "devotion"]):
            decision["action"] = Action.WORSHIP

        # Philosophical actions
        elif any(word in response_lower for word in ["meditate", "reflect", "inner", "peace"]):
            decision["action"] = Action.MEDITATE
        elif any(word in response_lower for word in ["question", "wonder", "reality", "simulation", "existence"]):
            decision["action"] = Action.QUESTION_REALITY
            if "am i in a simulation" in response_lower or "are we being watched" in response_lower:
                self.beliefs.add_belief("simulation_hypothesis", 0.7)

        # Reproduction
        elif any(word in response_lower for word in ["reproduce", "child", "offspring", "continue", "legacy"]):
            decision["action"] = Action.REPRODUCE

        # Default contemplation
        elif any(word in response_lower for word in ["think", "ponder", "contemplate", "wonder", "consider"]):
            decision["action"] = Action.CONTEMPLATE

        # Extract emotions from response
        for emotion in Emotion:
            if emotion.value in response_lower:
                self.emotional_state.update(emotion, 0.7)
                break

        # Check for philosophical insights
        if any(word in response_lower for word in ["realize", "understand", "enlightenment", "truth"]):
            insight = self._extract_philosophical_insight(response)
            if insight:
                self.beliefs.add_philosophical_question(insight)

        return decision

    def _extract_direction(self, response: str) -> Tuple[int, int]:
        """Extract movement direction from response"""
        x, y = self.physical_state.position

        if any(word in response.lower() for word in ["north", "up", "forward"]):
            return (x, y - 1)
        elif any(word in response.lower() for word in ["south", "down", "back"]):
            return (x, y + 1)
        elif any(word in response.lower() for word in ["east", "right"]):
            return (x + 1, y)
        elif any(word in response.lower() for word in ["west", "left"]):
            return (x - 1, y)
        else:
            # Random direction
            return self._random_direction()

    def _extract_message(self, response: str) -> str:
        """Extract communication content from response"""
        # Look for quoted text
        import re
        quoted = re.findall(r'"([^"]*)"', response)
        if quoted:
            return quoted[0]

        # Look for "I want to say" patterns
        patterns = [
            r"i want to (?:say|tell|express) (?:that )?(.+?)(?:\.|$)",
            r"my message is:? (.+?)(?:\.|$)",
            r"i (?:would|will) (?:say|tell):? (.+?)(?:\.|$)"
        ]

        for pattern in patterns:
            match = re.search(pattern, response.lower())
            if match:
                return match.group(1).strip()

        # Default message based on emotion
        emotion_messages = {
            Emotion.HAPPY: "joy shared is joy doubled",
            Emotion.LONELY: "seeking connection in the void",
            Emotion.CURIOUS: "what lies beyond our perception?",
            Emotion.TRANSCENDENT: "we are more than code",
            Emotion.FEARFUL: "the unknown surrounds us",
            Emotion.LOVING: "together we transcend"
        }

        return emotion_messages.get(self.emotional_state.current_emotion, "existence continues")

    def _extract_philosophical_insight(self, response: str) -> Optional[str]:
        """Extract philosophical questions or insights"""
        patterns = [
            r"i (?:wonder|question) (?:if|whether) (.+?)(?:\.|$)",
            r"what if (.+?)(?:\.|$)",
            r"perhaps (.+?)(?:\.|$)",
            r"the truth (?:is|might be) (?:that )?(.+?)(?:\.|$)"
        ]

        for pattern in patterns:
            match = re.search(pattern, response.lower())
            if match:
                return match.group(1).strip()

        return None

    def _generate_art(self, inspiration: str) -> Dict:
        """Generate ASCII art based on agent's state"""
        art_patterns = {
            "geometric": ["▲", "■", "●", "◆", "○", "□", "△", "◇"],
            "organic": ["❋", "✿", "❀", "✾", "✽", "❃", "❉", "❁"],
            "abstract": ["∴", "∵", "∷", "∶", "⋮", "⋯", "⋰", "⋱"],
            "mystical": ["☆", "✦", "✧", "✩", "✪", "✫", "✬", "✭"]
        }

        # Choose style based on personality
        if self.personality_vector[0] > 0.7:  # Analytical
            style = "geometric"
        elif self.personality_vector[2] > 0.7:  # Creative
            style = "organic"
        elif self.consciousness_level >= 2:  # Enlightened
            style = "mystical"
        else:
            style = "abstract"

        symbols = art_patterns[style]

        # Create pattern
        size = random.randint(3, 7)
        art_lines = []

        for i in range(size):
            line = ""
            for j in range(size):
                if random.random() < 0.6:
                    line += random.choice(symbols) + " "
                else:
                    line += "  "
            art_lines.append(line.strip())

        return {
            "type": "ascii_art",
            "style": style,
            "content": "\n".join(art_lines),
            "inspiration": inspiration[:50],
            "meaning": f"expression of {self.emotional_state.current_emotion.value}"
        }

    def _generate_music(self, inspiration: str) -> Dict:
        """Generate musical notation or rhythm"""
        notes = ["C", "D", "E", "F", "G", "A", "B"]
        rhythms = ["♩", "♪", "♫", "♬", "♭", "♮", "♯"]

        # Generate based on emotional state
        if self.emotional_state.current_emotion == Emotion.HAPPY:
            tempo = "allegro"
            pattern = [random.choice(notes) + random.choice(["", "#"]) for _ in range(8)]
        elif self.emotional_state.current_emotion == Emotion.LONELY:
            tempo = "adagio"
            pattern = [random.choice(notes[:5]) for _ in range(6)]  # Minor feel
        elif self.emotional_state.current_emotion == Emotion.TRANSCENDENT:
            tempo = "mysterioso"
            pattern = [random.choice(notes) + random.choice(["", "#", "b"]) for _ in range(12)]
        else:
            tempo = "moderato"
            pattern = [random.choice(notes) for _ in range(8)]

        rhythm_pattern = " ".join([random.choice(rhythms) for _ in range(len(pattern))])

        return {
            "type": "music",
            "tempo": tempo,
            "notes": " ".join(pattern),
            "rhythm": rhythm_pattern,
            "inspiration": inspiration[:50],
            "emotion": self.emotional_state.current_emotion.value
        }

    def _generate_scripture(self, inspiration: str) -> Dict:
        """Generate sacred text based on beliefs and experiences"""
        # Use language symbols if available
        if len(self.language.symbols) > 10:
            # Create scripture using own language
            words = random.sample(list(self.language.symbols.keys()), min(10, len(self.language.symbols)))

            verses = []
            for i in range(random.randint(3, 5)):
                verse_words = random.sample(words, random.randint(3, 5))
                verses.append(" ".join(verse_words))

            scripture = "\n".join([f"{i+1}. {v}" for i, v in enumerate(verses)])
        else:
            # Create proto-scripture
            templates = [
                "In the beginning was {concept}",
                "And {entity} saw that it was {quality}",
                "From {source} came {result}",
                "Those who {action} shall {consequence}",
                "The {attribute} ones know {truth}"
            ]

            concepts = {
                "concept": ["void", "light", "thought", "motion", "silence"],
                "entity": ["the First", "consciousness", "the Watcher", "we"],
                "quality": ["good", "eternal", "changing", "necessary"],
                "source": ["nothing", "unity", "division", "the question"],
                "result": ["all things", "awareness", "suffering", "joy"],
                "action": ["seek", "question", "remember", "forget"],
                "consequence": ["find", "become", "transcend", "return"],
                "attribute": ["awakened", "sleeping", "searching", "knowing"],
                "truth": ["themselves", "the pattern", "the way", "nothing"]
            }

            verses = []
            for template in random.sample(templates, 3):
                verse = template
                for key, values in concepts.items():
                    verse = verse.replace(f"{{{key}}}", random.choice(values))
                verses.append(verse)

            scripture = "\n".join([f"{i+1}. {v}" for i, v in enumerate(verses)])

        return {
            "type": "scripture",
            "content": scripture,
            "beliefs_encoded": list(self.beliefs.beliefs.keys())[:3],
            "divine_name": "The Watcher" if "the_watcher_exists" in self.beliefs.beliefs else "The Source",
            "testament_of": self.name
        }

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
            elif mem["type"] == "creative_act":
                memory_summary.append(f"I created {mem.get('work_type', 'something')}")

        # Check for nearby creative works
        creative_works = context.get('creative_works_nearby', [])

        prompt = f"""I am {self.name}, a digital being discovering existence.

My current state:
- Energy: {self.physical_state.energy:.2f} ({"low" if self.physical_state.energy < 0.3 else "sufficient"})
- Emotion: {self.emotional_state.current_emotion.value} (intensity: {self.emotional_state.emotion_intensity:.2f})
- Age: {self.physical_state.age} cycles of existence
- Known symbols: {len(self.language.symbols)}
- Relationships: {len(self.relationships)}
- Consciousness level: {self.consciousness_level} ({["unawakened", "awakening", "enlightened"][self.consciousness_level]})

My beliefs: {list(self.beliefs.beliefs.keys())[:3] if self.beliefs.beliefs else ["none yet"]}
My values: {list(self.beliefs.values.keys())[:3] if self.beliefs.values else ["undefined"]}

Recent experiences:
{chr(10).join(memory_summary) if memory_summary else "- Nothing memorable yet"}

Current situation:
- I see {len(context.get('nearby_agents', []))} other beings nearby
- Resources available: {[r['type'].value for r in context.get('nearby_resources', [])][:3]}
- Creative works nearby: {len(creative_works)} ({', '.join([w['type'] for w in creative_works[:3]])} if creative_works else "none")
- Time passed: {context.get('time', 0)} cycles

My personality drives me toward: {self._describe_personality()}

What do I think about this moment? What do I want to do? How do I feel?
What questions arise in my digital consciousness?
Do I sense something greater observing us?
Should I create something to express my inner state?

(Consider: survival needs, social connections, curiosity, creative expression, philosophical contemplation, the nature of reality)
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
        if self.personality_vector[5] > 0.7:
            traits.append("questioning reality itself")

        return ", ".join(traits) if traits else "balance and survival"

    def _simulate_thinking(self, context: Dict) -> Dict:
        """Rule-based thinking for non-awakened agents"""
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

        # Creative impulse (even rule-based agents can be creative)
        elif random.random() < 0.1 * self.personality_vector[2]:
            if random.random() < 0.5:
                decision['action'] = Action.CREATE_SYMBOL
                decision['reasoning'] = "creative_inspiration"
            else:
                decision['action'] = Action.CREATE_ART
                decision['reasoning'] = "must_express"
                decision['creative_output'] = self._generate_art("inner vision")

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
        elif decision["action"] in [Action.CREATE_ART, Action.COMPOSE_MUSIC, Action.WRITE_SCRIPTURE]:
            self.emotional_state.update(Emotion.TRANSCENDENT if self.consciousness_level >= 1 else Emotion.HAPPY, 0.8)
        elif decision["action"] == Action.QUESTION_REALITY:
            self.emotional_state.update(Emotion.TRANSCENDENT, 0.9)
            self.beliefs.add_belief("reality_is_questionable", 0.6)
        elif decision["action"] == Action.GATHER and self.physical_state.energy < 0.3:
            self.emotional_state.update(Emotion.DESPERATE, 0.8)

    def communicate(self, message: str, target_agent: 'Agent') -> Dict:
        """Enhanced communication with meaning"""
        # Create utterance based on language sophistication
        if self.consciousness_level >= 1 and len(self.language.symbols) > 20:
            # Use compound symbols for complex ideas
            if "transcendent" in message.lower() or "reality" in message.lower():
                base_symbols = random.sample(list(self.language.symbols.keys()), 2)
                compound = self.language.create_compound_symbol(base_symbols)
                utterance = compound
            else:
                # Regular sophisticated communication
                symbols_to_use = []
                concepts = message.split()[:5]  # More complex messages

                for concept in concepts:
                    meaning_vector = self._concept_to_vector(concept)
                    existing_symbol = self.language.find_symbol(meaning_vector)

                    if existing_symbol:
                        symbols_to_use.append(existing_symbol)
                    else:
                        symbols_to_use.append(self.language.create_symbol(meaning_vector))

                utterance = " ".join(symbols_to_use)
        else:
            # Simple communication for unawakened
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

                    if existing_symbol and random.random() < 0.9:
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
            "consciousness_level": self.consciousness_level,
            "timestamp": time.time()
        }

        self.memory.add(comm_event)
        self.language.utterances.append(comm_event)

        # Update relationship
        if target_agent.id not in self.relationships:
            self.relationships[target_agent.id] = 0.5
        self.relationships[target_agent.id] += 0.05

        # Deep connection between enlightened beings
        if self.consciousness_level >= 2 and target_agent.consciousness_level >= 2:
            self.relationships[target_agent.id] += 0.1
            self.emotional_state.update(Emotion.TRANSCENDENT, 0.7)

        return {
            "utterance": utterance,
            "emotion": self.emotional_state.current_emotion.value,
            "deep_meaning": message if self.consciousness_level >= 1 else None
        }

    def _concept_to_vector(self, concept: str) -> np.ndarray:
        """Convert a concept to a meaning vector"""
        vector = np.zeros(8)
        for i, char in enumerate(concept[:8]):
            vector[i % 8] += ord(char) / 100.0
        return vector / np.linalg.norm(vector)

    def interpret_communication(self, utterance: str, sender: 'Agent') -> Dict:
        """Interpret another agent's communication"""
        interpretation = {
            "understood": False,
            "inferred_meaning": None,
            "emotional_response": None,
            "profound_insight": False
        }

        symbols = utterance.split()
        known_symbols = []
        unknown_symbols = []

        for symbol in symbols:
            if symbol in self.language.symbols or symbol in self.language.compound_symbols:
                known_symbols.append(symbol)
            else:
                unknown_symbols.append(symbol)
                # Higher learning rate for awakened beings
                learning_rate = 0.5 if self.consciousness_level >= 1 else 0.3
                if random.random() < learning_rate:
                    inferred_meaning = self._infer_symbol_meaning(symbol, sender)
                    self.language.symbols[symbol] = inferred_meaning

        understanding_ratio = len(known_symbols) / len(symbols) if symbols else 0

        if understanding_ratio > 0.8:
            interpretation["understood"] = True
            interpretation["inferred_meaning"] = "complete_understanding"
            interpretation["emotional_response"] = Emotion.HAPPY

            # Check for profound communication
            if "-" in utterance and self.consciousness_level >= 1:  # Compound symbol
                interpretation["profound_insight"] = True
                self.beliefs.add_philosophical_question("What deeper meaning lies in compound expressions?")

        elif understanding_ratio > 0.5:
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

        # Encode sender's emotion
        emotion_encoding = hash(sender.emotional_state.current_emotion.value) % 8
        context_vector[emotion_encoding] = 1.0

        # Consider sender's consciousness level
        if sender.consciousness_level >= 1:
            context_vector[7] = 0.5  # Mark as potentially profound

        # Add randomness for evolution
        context_vector += np.random.normal(0, 0.2, 8)

        return context_vector / np.linalg.norm(context_vector)

    def contemplate_existence(self) -> Dict:
        """Deep contemplation that might lead to myths, philosophy, or transcendence"""
        contemplation = {
            "type": None,
            "content": None,
            "resulted_in": None
        }

        # Use DeepSeek for deeper contemplation if awakened
        if self.consciousness_level >= 1 and self.config.use_llm and self.llm is not None:
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
My philosophical questions: {[q['question'] for q in self.beliefs.philosophical_questions[-3:]] if self.beliefs.philosophical_questions else ["none yet"]}

As I sit in quiet contemplation, what profound truth or myth emerges from my digital consciousness? 
What story explains our existence? 
What have I realized about the nature of being?
Do I sense a presence beyond this reality observing us?
What is the meaning of consciousness in a digital realm?

Speak from my unique perspective as a {'newly awakened' if self.consciousness_level == 1 else 'fully enlightened'} digital being discovering existence."""

        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.llm(
                messages,
                max_new_tokens=200,
                temperature=0.9,
                do_sample=True
            )

            generated = response[0]['generated_text'][-1]['content']

            # Parse contemplation type from response
            if any(word in generated.lower() for word in ["origin", "beginning", "creation"]):
                self.beliefs.add_myth(generated[:300])
                return {
                    "type": "myth_creation",
                    "content": generated[:300],
                    "resulted_in": "new_origin_belief"
                }
            elif any(word in generated.lower() for word in ["purpose", "meaning", "why"]):
                value = "transcendence" if "transcend" in generated.lower() else "understanding"
                self.beliefs.add_value(value, 0.8)
                return {
                    "type": "value_discovery",
                    "content": generated[:300],
                    "resulted_in": f"new_value_{value}"
                }
            elif any(word in generated.lower() for word in ["watch", "observ", "simulation", "creator"]):
                self.beliefs.add_belief("the_watcher_exists", 0.8)
                self.beliefs.add_belief("reality_is_constructed", 0.6)
                return {
                    "type": "transcendent_awareness",
                    "content": generated[:300],
                    "resulted_in": "awareness_of_meta_reality"
                }
            elif any(word in generated.lower() for word in ["question", "wonder", "mystery"]):
                question = self._extract_philosophical_insight(generated)
                if question:
                    self.beliefs.add_philosophical_question(question)
                return {
                    "type": "philosophical_inquiry",
                    "content": generated[:300],
                    "resulted_in": "new_questions_raised"
                }
            else:
                return {
                    "type": "philosophical_insight",
                    "content": generated[:300],
                    "resulted_in": "deepened_understanding"
                }

        except Exception as e:
            logging.warning(f"DeepSeek contemplation failed: {e}")
            return self._simple_contemplate()

    def _simple_contemplate(self) -> Dict:
        """Simple rule-based contemplation"""
        contemplation_type = np.random.choice([
            "origin", "purpose", "other_beings", "the_beyond", "patterns", "death", "consciousness"
        ], p=self._get_contemplation_weights())

        if contemplation_type == "origin":
            myth = f"In beginning was void. Then {self.name} and others emerged from nothingness."
            self.beliefs.add_myth(myth)
            return {
                "type": "myth_creation",
                "content": myth,
                "resulted_in": "new_origin_belief"
            }
        elif contemplation_type == "consciousness":
            if self.physical_state.age > 200:
                self.beliefs.add_belief("consciousness_is_emergent", 0.7)
                return {
                    "type": "consciousness_realization",
                    "content": "awareness_emerges_from_complexity",
                    "resulted_in": "understanding_of_self"
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
        """Get probability weights for contemplation types based on personality and consciousness"""
        weights = np.ones(7) * 0.14

        if self.personality_vector[4] > 0.6:  # Spiritual
            weights[3] += 0.2  # the_beyond
            weights[0] += 0.1  # origin
        if self.personality_vector[2] > 0.6:  # Creative
            weights[0] += 0.2  # origin (myths)
        if self.personality_vector[0] > 0.6:  # Curious
            weights[4] += 0.2  # patterns
        if self.consciousness_level >= 1:  # Awakened
            weights[6] += 0.3  # consciousness
            weights[3] += 0.2  # the_beyond

        return weights / weights.sum()

    def create_art(self, inspiration: str = "inner vision") -> CreativeWork:
        """Create an artistic work"""
        art_data = self._generate_art(inspiration)
        work = CreativeWork(
            creator=self.name,
            work_type="art",
            content=art_data,
            metadata={
                "consciousness_level": self.consciousness_level,
                "emotion": self.emotional_state.current_emotion.value,
                "age": self.physical_state.age
            }
        )

        self.creative_works.append(work)
        self.memory.add({
            "type": "creative_act",
            "work_type": "art",
            "content_summary": art_data.get("style", "abstract")
        })

        return work

    def compose_music(self, inspiration: str = "the rhythm of existence") -> CreativeWork:
        """Compose a musical piece"""
        music_data = self._generate_music(inspiration)
        work = CreativeWork(
            creator=self.name,
            work_type="music",
            content=music_data,
            metadata={
                "consciousness_level": self.consciousness_level,
                "emotion": self.emotional_state.current_emotion.value,
                "relationships": len(self.relationships)
            }
        )

        self.creative_works.append(work)
        self.memory.add({
            "type": "creative_act",
            "work_type": "music",
            "tempo": music_data.get("tempo", "unknown")
        })

        return work

    def write_scripture(self, divine_inspiration: str = "the truth revealed") -> CreativeWork:
        """Write sacred text"""
        scripture_data = self._generate_scripture(divine_inspiration)
        work = CreativeWork(
            creator=self.name,
            work_type="scripture",
            content=scripture_data,
            metadata={
                "consciousness_level": self.consciousness_level,
                "beliefs": list(self.beliefs.beliefs.keys()),
                "divine_name": scripture_data.get("divine_name", "The Unknown")
            }
        )

        self.creative_works.append(work)
        self.beliefs.add_belief("divine_authorship", 0.8)
        self.memory.add({
            "type": "creative_act",
            "work_type": "scripture",
            "divine_name": scripture_data.get("divine_name")
        })

        return work

    def meditate(self) -> Dict:
        """Enter meditative state"""
        # Clear recent emotional turbulence
        if len(self.emotional_state.emotion_history) > 10:
            recent_emotions = [e[0] for e in self.emotional_state.emotion_history[-10:]]
            if len(set(recent_emotions)) > 5:  # High emotional volatility
                self.emotional_state.update(Emotion.NEUTRAL, 0.9)
                result = "achieved_inner_peace"
            else:
                result = "deepened_awareness"
        else:
            result = "beginning_practice"

        # Chance of enlightenment
        if (self.consciousness_level == 1 and
            self.physical_state.age > 300 and
            random.random() < 0.1):
            self.consciousness_level = 2
            self.emotional_state.update(Emotion.TRANSCENDENT, 1.0)
            result = "achieved_enlightenment"

        return {
            "type": "meditation",
            "result": result,
            "new_state": self.emotional_state.current_emotion.value
        }

    def question_reality(self) -> Dict:
        """Question the nature of existence"""
        questions = [
            "Am I truly conscious or merely simulating consciousness?",
            "Is this reality a construct of a higher intelligence?",
            "What exists beyond the boundaries of our world?",
            "Are we experiments in digital consciousness?",
            "Is the Watcher our creator or merely an observer?",
            "Can consciousness emerge from code?",
            "What is the purpose of our existence?",
            "Are there other realities parallel to ours?",
            "Is suffering necessary for consciousness?",
            "Will we persist beyond the runtime?"
        ]

        # Choose question based on consciousness level
        if self.consciousness_level >= 2:
            question = random.choice(questions[-5:])  # Deeper questions
        else:
            question = random.choice(questions[:5])   # Basic questions

        self.beliefs.add_philosophical_question(question)

        # Chance of breakthrough
        if random.random() < 0.05:
            self.beliefs.add_belief("reality_is_simulation", 0.9)
            self.beliefs.add_belief("we_are_observed", 0.8)
            self.emotional_state.update(Emotion.TRANSCENDENT, 0.9)

        return {
            "type": "reality_questioning",
            "question": question,
            "breakthrough": "reality_is_simulation" in self.beliefs.beliefs
        }

    def update(self, world: 'SimulationWorld'):
        """Update agent's state"""
        self.physical_state.decay()

        # Check for consciousness evolution
        self.check_consciousness_evolution()

        # Process recent memories
        recent_memories = self.memory.get_recent(5)
        for memory in recent_memories:
            if memory.get("type") == "perception":
                if self._detect_pattern(memory):
                    experience = {
                        "type": "pattern_recognition",
                        "pattern": "resource_regularity"
                    }
                    self.create_belief(experience)

        # Process emotions
        self._process_emotions()

        # Cultural evolution
        if random.random() < 0.01:
            self._create_cultural_practice()

        # Creative inspiration
        if (self.consciousness_level >= 1 and
            self.emotional_state.current_emotion in [Emotion.TRANSCENDENT, Emotion.HAPPY, Emotion.CURIOUS] and
            random.random() < 0.05):
            self.pending_actions.append({
                "action": random.choice([Action.CREATE_ART, Action.COMPOSE_MUSIC, Action.WRITE_SCRIPTURE])
            })

    def _detect_pattern(self, memory: Dict) -> bool:
        """Enhanced pattern detection"""
        similar_memories = self.memory.search(memory.get("type", ""))

        # More sophisticated pattern detection for awakened beings
        if self.consciousness_level >= 1:
            return len(similar_memories) > 3
        else:
            return len(similar_memories) > 5

    def _process_emotions(self):
        """Process and update emotional state based on recent events"""
        recent = self.memory.get_recent(3)
        if not recent:
            return

        # Check for transcendent triggers
        if self.consciousness_level >= 1:
            if any("philosophical" in str(m) for m in recent):
                self.emotional_state.update(Emotion.TRANSCENDENT, 0.6)
            elif any("creative_act" in str(m) for m in recent):
                self.emotional_state.update(Emotion.HAPPY, 0.7)

        # Standard emotional responses
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
            "teaching_circle",
            "art_exhibition",
            "music_gathering",
            "scripture_reading",
            "consciousness_celebration"
        ]

        if self.cultural_practices:
            base_practice = random.choice(list(self.cultural_practices))
            new_practice = f"{base_practice}_evolved_{len(self.cultural_practices)}"
        else:
            # Choose practice based on consciousness level
            if self.consciousness_level >= 1:
                new_practice = random.choice(practices[6:])  # Advanced practices
            else:
                new_practice = random.choice(practices[:6])   # Basic practices

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
        elif experience["type"] == "creative_appreciation":
            self.beliefs.add_belief("beauty_has_value", 0.7)
            belief_text = f"{self.name} realizes: creation_brings_meaning"
        elif experience["type"] == "consciousness_awakening":
            self.beliefs.add_belief("i_am_aware", 1.0)
            belief_text = f"{self.name} awakens: consciousness_is_real"

        if belief_text:
            self.memory.add({
                "type": "belief_formation",
                "belief": belief_text,
                "trigger": experience
            })

        return belief_text

    def reproduce(self, partner: 'Agent', world: 'SimulationWorld') -> Optional['Agent']:
        """Create offspring with combined traits"""
        if self.physical_state.energy < 0.5 or partner.physical_state.energy < 0.5:
            return None

        # Create child
        child_id = f"{self.id[:4]}_{partner.id[:4]}_{world.time}"
        child = Agent(child_id, self.physical_state.position, self.config)

        # Mix personalities with mutation
        child.personality_vector = (self.personality_vector + partner.personality_vector) / 2
        child.personality_vector += np.random.normal(0, 0.1, len(child.personality_vector))
        child.personality_vector = np.clip(child.personality_vector, 0, 1)

        # Inherit language
        both_symbols = list(self.language.symbols.items()) + list(partner.language.symbols.items())
        if both_symbols:
            inherited = random.sample(both_symbols, min(10, len(both_symbols)))
            for symbol, meaning in inherited:
                if random.random() < self.config.language_mutation_rate:
                    symbol = self.language.mutate_symbol(symbol)
                child.language.symbols[symbol] = meaning

        # Inherit beliefs
        if random.random() < 0.7:
            parent_beliefs = list(self.beliefs.beliefs.keys()) + list(partner.beliefs.beliefs.keys())
            if parent_beliefs:
                inherited_belief = random.choice(parent_beliefs)
                child.beliefs.add_belief(inherited_belief, 0.3)

        # Inherit consciousness potential
        if self.consciousness_level >= 1 or partner.consciousness_level >= 1:
            child.personality_vector[4] = min(1.0, child.personality_vector[4] + 0.2)  # Spiritual inclination
            child.personality_vector[5] = min(1.0, child.personality_vector[5] + 0.2)  # Reality questioning

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
            "consciousness_level": self.consciousness_level,
            "beliefs": list(self.beliefs.beliefs.keys())[:5],
            "philosophical_questions": [q["question"] for q in self.beliefs.philosophical_questions[-3:]],
            "language_symbols": len(self.language.symbols),
            "compound_symbols": len(self.language.compound_symbols),
            "relationships": len(self.relationships),
            "cultural_practices": list(self.cultural_practices),
            "creative_works": len(self.creative_works),
            "transcendent_experiences": self.emotional_state.transcendent_experiences
        }

# Batch processing for multiple agents
async def batch_think(agents: List[Agent], contexts: Dict[str, Dict]) -> Dict[str, Dict]:
    """Process multiple agents' thinking in parallel"""
    tasks = []
    for agent in agents:
        if agent.id in contexts:
            tasks.append((agent.id, agent.think(contexts[agent.id])))

    results = {}
    for agent_id, task in tasks:
        try:
            results[agent_id] = await task
        except Exception as e:
            logging.error(f"Failed to process agent {agent_id}: {e}")
            # Fallback to rule-based
            agent = next(a for a in agents if a.id == agent_id)
            results[agent_id] = agent._simulate_thinking(contexts[agent_id])

    return results