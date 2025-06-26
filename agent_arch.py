# agent_arch.py

import numpy as np
import random
import json
import time
from collections import deque
from typing import Dict, List, Tuple, Optional, Any
import chromadb
from chromadb.utils import embedding_functions

from language_evolve import Language

# Enums
from enum import Enum

class Emotion(Enum):
    NEUTRAL   = "neutral"
    HAPPY     = "happy"
    ANGRY     = "angry"
    FEARFUL   = "fearful"
    CURIOUS   = "curious"
    LONELY    = "lonely"
    LOVING    = "loving"
    CONFUSED  = "confused"
    HOPEFUL   = "hopeful"
    DESPERATE = "desperate"

class ResourceType(Enum):
    FOOD      = "food"
    WATER     = "water"
    SHELTER   = "shelter"
    LIGHT     = "light"     # mystical
    KNOWLEDGE = "knowledge" # abstract

class Action(Enum):
    MOVE          = "move"
    GATHER        = "gather"
    COMMUNICATE   = "communicate"
    SHARE         = "share"
    ATTACK        = "attack"
    REPRODUCE     = "reproduce"
    CONTEMPLATE   = "contemplate"
    CREATE_SYMBOL = "create_symbol"
    TEACH         = "teach"
    WORSHIP       = "worship"

# Physical State
from dataclasses import dataclass

@dataclass
class PhysicalState:
    energy: float = 1.0
    health: float = 1.0
    age:    int   = 0
    position: Tuple[int,int] = (0,0)
    carrying: Dict[ResourceType,float] = None

    def __post_init__(self):
        if self.carrying is None:
            self.carrying = {r:0.0 for r in ResourceType}

    def decay(self):
        self.energy *= 0.98
        self.health *= 0.995
        self.age    += 1

    def is_alive(self) -> bool:
        return self.energy > 0 and self.health > 0

# Emotional State
class EmotionalState:
    def __init__(self):
        self.current_emotion   = Emotion.NEUTRAL
        self.emotion_intensity = 0.5
        self.emotion_history   = deque(maxlen=20)

    def update(self, new_emotion: Emotion, intensity: float = 0.5):
        self.emotion_history.append((self.current_emotion, self.emotion_intensity))
        self.current_emotion   = new_emotion
        self.emotion_intensity = min(1.0, intensity)

    def get_emotional_state(self) -> Dict:
        return {
            "current":   self.current_emotion.value,
            "intensity": self.emotion_intensity,
            "history":   list(self.emotion_history)[-5:]
        }

# Belief System
class BeliefSystem:
    def __init__(self):
        self.beliefs        = {}   # concept -> strength
        self.myths          = []   # narrative beliefs
        self.values         = {}   # moral_concept -> importance
        self.taboos         = set()
        self.sacred_symbols = set()

    def add_belief(self, concept: str, strength: float = 0.5):
        self.beliefs[concept] = min(1.0, self.beliefs.get(concept, 0.0) + strength)

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

# Memory Systems

class ShortTermMemory:
    def __init__(self, capacity: int = 100):
        """
        A simple in-memory FIFO store for recent events, perceptions, etc.

        :param capacity: maximum number of memories to retain
        """
        self.capacity = capacity
        self.memories = deque(maxlen=capacity)

    def add(self, memory: Dict[str, Any]) -> None:
        """
        Add a new memory, stamping it with the current timestamp.

        :param memory: a dict describing the event or perception
        """
        memory["timestamp"] = time.time()
        self.memories.append(memory)

    def get_recent(self, n: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve up to the last n memories.

        :param n: number of most recent memories to return
        """
        return list(self.memories)[-n:]

    def search(self, query: str) -> List[Dict[str, Any]]:
        """
        Find all memories whose stringified form contains the query (case-insensitive).
        Uses plain str() so it will never fail on enums, numpy arrays, etc.

        :param query: substring to look for
        """
        lower_q = query.lower()
        return [m for m in self.memories if lower_q in str(m).lower()]

class VectorMemory:
    def __init__(self):
        self.client = chromadb.Client()
        self.collection = self.client.create_collection(
            name=f"agent_memory_{random.randint(1e3,1e4)}",
            embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction()
        )

    def store(self, text: str, metadata: Dict):
        self.collection.add(
            documents=[text],
            metadatas=[metadata],
            ids=[f"mem_{time.time()}_{random.randint(1e3,1e4)}"]
        )

    def retrieve(self, query: str, n_results: int = 5) -> List[Dict]:
        res = self.collection.query(query_texts=[query], n_results=n_results)
        return res

# Agent
class Agent:
    def __init__(self, agent_id: str, position: Tuple[int,int], model_config: Dict[str,Any]):
        self.id = agent_id
        self.name = self._generate_name()
        self.model_config = model_config

        # systems
        self.memory           = ShortTermMemory(capacity=model_config.get("max_memory_size",100))
        self.long_term_memory = VectorMemory()
        self.physical_state   = PhysicalState(position=position)
        self.emotional_state  = EmotionalState()
        self.beliefs          = BeliefSystem()
        self.language         = Language()

        # personality & culture
        self.personality_vector = np.random.rand(32)
        self.relationships: Dict[str,float] = {}
        self.cultural_practices = set()
        self.inventions         = []

        # internal
        self.current_goal     = None
        self.pending_actions  = []
        self.last_action_time = time.time()
        self.reputation       = 0.0
        self.skills           = {}

    def _generate_name(self) -> str:
        syll = ["ka","ra","mi","to","na","be","lu","so","chi","wa"]
        return "".join(random.sample(syll, random.randint(2,3))).capitalize()

    def perceive(self, environment: 'SimulationWorld') -> Dict[str,Any]:
        p = {
            "position": self.physical_state.position,
            "nearby_agents":    environment.get_nearby_agents(self.physical_state.position,5),
            "nearby_resources": environment.get_nearby_resources(self.physical_state.position,3),
            "time":      environment.time,
            "events":    environment.get_recent_events()
        }
        self.memory.add({"type":"perception","data":p,"emotion":self.emotional_state.current_emotion.value})
        return p

    def _simulate_thinking(self, context: Dict) -> Dict:
        """Simple rule‐based decision making."""
        dec = {"action":None, "target":None, "reasoning":""}

        # survival
        if self.physical_state.energy < 0.3:
            if context["nearby_resources"]:
                foods = [r for r in context["nearby_resources"] if r["type"]==ResourceType.FOOD]
                if foods:
                    dec.update(action=Action.GATHER, target=foods[0]["position"], reasoning="urgent_hunger")
                    return dec
            dec.update(action=Action.MOVE, target=self._random_direction(), reasoning="seek_food")
            return dec

        # social
        if self.emotional_state.current_emotion==Emotion.LONELY and context["nearby_agents"]:
            tgt = random.choice(context["nearby_agents"])
            dec.update(action=Action.COMMUNICATE, target=tgt["id"], reasoning="seek_company")
            return dec

        # creative
        if random.random() < 0.1*self.personality_vector[2]:
            dec.update(action=Action.CREATE_SYMBOL, reasoning="creative_impulse")
            return dec

        # exploration
        if random.random() < 0.2*self.personality_vector[0]:
            dec.update(action=Action.MOVE, target=self._random_direction(), reasoning="curiosity")
            return dec

        # default
        dec.update(action=Action.CONTEMPLATE, reasoning="reflect")
        return dec

    def _random_direction(self) -> Tuple[int,int]:
        x,y = self.physical_state.position
        dx,dy = random.choice([(1,0),(-1,0),(0,1),(0,-1)])
        return (x+dx, y+dy)

    def communicate(self, message: str, target: 'Agent') -> Dict[str,Any]:
        # build an utterance from symbols
        if not self.language.symbols:
            vec = np.random.rand(8)
            sym = self.language.create_symbol(vec)
            utt = sym
        else:
            tokens = message.split()[:3]
            syms = []
            for t in tokens:
                vec = self._concept_to_vector(t)
                exist = self.language.find_symbol(vec)
                if exist and random.random()<0.9:
                    syms.append(exist)
                else:
                    syms.append(self.language.create_symbol(vec))
            utt = " ".join(syms)
        self.memory.add({"type":"communication","from":self.id,"to":target.id,"utterance":utt})
        self.language.utterances.append(utt)
        self.relationships[target.id] = self.relationships.get(target.id,0.5)+0.05
        return {"utterance":utt,"emotion":self.emotional_state.current_emotion.value}

    def interpret_communication(self, utterance: str, sender: 'Agent') -> Dict[str,Any]:
        symbols = utterance.split()
        known = [s for s in symbols if s in self.language.symbols]
        unknown = [s for s in symbols if s not in self.language.symbols]
        if len(known)>len(unknown):
            resp = Emotion.CURIOUS
            inf  = "partial"
            und  = True
        elif known:
            resp = Emotion.CONFUSED
            inf  = "confusion"
            und  = False
        else:
            resp = (Emotion.FEARFUL if random.random()<0.3 else Emotion.CURIOUS)
            inf  = "mystery"
            und  = False
        self.emotional_state.update(resp)
        self.memory.add({"type":"received_comm","utterance":utterance,"understood":und})
        return {"understood":und,"inferred":inf,"emotional_response":resp}

    def _concept_to_vector(self, concept: str) -> np.ndarray:
        v = np.zeros(8)
        for i,ch in enumerate(concept[:8]):
            v[i] = ord(ch)/100.0
        return v/np.linalg.norm(v)

    def create_belief(self, exp: Dict) -> Optional[str]:
        txt = None
        t = exp.get("type")
        if t=="near_death":
            self.beliefs.add_belief("mortality_awareness",0.8)
            txt = f"{self.name} knows: all_beings_fade"
        elif t=="successful_cooperation":
            self.beliefs.add_belief("cooperation_good",0.7)
            txt = f"{self.name} learns: together_stronger"
        elif t=="resource_discovery":
            r = exp.get("resource_type")
            self.beliefs.add_belief(f"{r}_pattern",0.6)
            txt = f"{self.name} notices: {r}_follows_pattern"
        if txt:
            self.memory.add({"type":"belief_formation","belief":txt,"trigger":exp})
        return txt

    def contemplate_existence(self) -> Dict[str,Any]:
        # simple rule-based
        choices = ["origin","purpose","the_beyond","patterns","death"]
        c = random.choice(choices)
        if c=="origin":
            myth = f"In_beginning: void_then_{self.name}. We_emerged."
            self.beliefs.add_myth(myth)
            return {"type":"myth","content":myth}
        if c=="the_beyond" and random.random()<0.1:
            self.beliefs.add_belief("the_watcher_exists",0.3)
            return {"type":"transcendent","content":"sense_of_being_observed"}
        return {"type":"ponder","content":f"pondering_{c}"}

    def update(self, world: 'SimulationWorld'):
        self.physical_state.decay()
        recent = self.memory.get_recent(5)
        for m in recent:
            if m.get("type")=="perception" and len(self.memory.search("perception"))>5:
                self.create_belief({"type":"pattern_recognition"})
        # emotion rules
        if self.physical_state.energy<0.3:
            self.emotional_state.update(Emotion.DESPERATE,0.9)

    def reproduce(self, partner: 'Agent', world: 'SimulationWorld') -> Optional['Agent']:
        if self.physical_state.energy<0.5 or partner.physical_state.energy<0.5:
            return None
        pid = f"{self.id[:4]}_{partner.id[:4]}_{world.time}"
        child = Agent(pid, self.physical_state.position, self.model_config)
        # personality mix
        pv = (self.personality_vector+partner.personality_vector)/2
        pv += np.random.normal(0,0.1,len(pv))
        child.personality_vector = np.clip(pv,0,1)
        # language inherit
        both = list(self.language.symbols.items())+list(partner.language.symbols.items())
        for sym,vec in random.sample(both, min(5,len(both))):
            child.language.symbols[sym] = vec
        # beliefs inherit
        if random.random()<0.7 and self.beliefs.beliefs:
            b = random.choice(list(self.beliefs.beliefs.keys()))
            child.beliefs.add_belief(b,0.3)
        self.physical_state.energy *= 0.7
        partner.physical_state.energy *= 0.7
        return child

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "name": self.name,
            "position": self.physical_state.position,
            "energy": self.physical_state.energy,
            "health": self.physical_state.health,
            "age": self.physical_state.age,
            "emotion": self.emotional_state.current_emotion.value,
            "beliefs": list(self.beliefs.beliefs.keys()),
            "language_symbols": len(self.language.symbols),
            "relationships": len(self.relationships),
            "cultural_practices": list(self.cultural_practices)
        }
