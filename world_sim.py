# world_sim.py

import numpy as np
import time
import logging
import random
import json
from collections import defaultdict, deque
from typing import Dict, Tuple, List, Optional, Any
from dataclasses import dataclass

from agent_arch import Agent, Action, ResourceType

# Configuration
@dataclass
class SimulationConfig:
    world_size:          Tuple[int,int] = (50,50)
    initial_agents:      int = 20
    resource_spawn_rate: float = 0.1
    time_per_tick:       float = 0.1
    max_memory_size:     int = 100
    language_mutation_rate: float = 0.05
    death_threshold:     float = 0.0
    reproduction_threshold: float = 0.8

# Events
class Event:
    def __init__(self, event_type: str, data: Dict[str,Any], position: Optional[Tuple[int,int]]=None):
        self.type      = event_type
        self.data      = data
        self.position  = position
        self.timestamp = time.time()
        self.id        = f"{event_type}_{self.timestamp}_{random.randint(1e3,1e4)}"

class EventQueue:
    def __init__(self):
        self.events        = deque()
        self.event_history = []
        self.event_count   = defaultdict(int)

    def add(self, event: Event):
        self.events.append(event)
        self.event_history.append(event)
        self.event_count[event.type] += 1

    def process(self) -> List[Event]:
        out = list(self.events)
        self.events.clear()
        return out

    def get_recent(self, n: int = 10) -> List[Event]:
        return self.event_history[-n:]

# Resource Manager
class ResourceManager:
    def __init__(self, world_size: Tuple[int,int]):
        self.world_size = world_size
        self.resources  = {}  # pos -> {type,amount}

    def spawn_resources(self, spawn_rate: float):
        area = self.world_size[0]*self.world_size[1]
        for _ in range(int(area*spawn_rate)):
            x = random.randrange(self.world_size[0])
            y = random.randrange(self.world_size[1])
            if (x,y) not in self.resources:
                rtype = random.choice(list(ResourceType))
                amt   = random.uniform(0.5,1.0)
                self.resources[(x,y)] = {"type":rtype, "amount":amt, "position":(x,y)}

    def gather(self, position: Tuple[int,int], amount: float = 0.1) -> Optional[Dict]:
        if position in self.resources:
            res = self.resources[position]
            taken = min(amount, res["amount"])
            res["amount"] -= taken
            if res["amount"]<=0:
                del self.resources[position]
            return {"type":res["type"], "amount":taken}
        return None

    def get_nearby(self, pos: Tuple[int,int], radius: int = 3) -> List[Dict]:
        x,y = pos
        out = []
        for dx in range(-radius, radius+1):
            for dy in range(-radius, radius+1):
                p = (x+dx, y+dy)
                if p in self.resources:
                    out.append(self.resources[p].copy())
        return out

# Simulation World
class SimulationWorld:
    def __init__(self, config: SimulationConfig = None):
        self.config   = config or SimulationConfig()
        self.size     = self.config.world_size
        self.agents   = {}
        self.resources= ResourceManager(self.size)
        self.events   = EventQueue()
        self.time     = 0

        # emergent tracking
        self.cultures = {}
        self.languages= {}
        self.myths    = []

        self._setup_logging()
        self._spawn_initial_agents()
        self._spawn_initial_resources()

    def _setup_logging(self):
        self.logger = logging.getLogger("ANIMA")
        self.logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        self.logger.addHandler(ch)

    def _spawn_initial_agents(self):
        for i in range(self.config.initial_agents):
            pos = (random.randrange(self.size[0]), random.randrange(self.size[1]))
            aid = f"agent_{i:04d}"
            ag  = Agent(aid, pos, vars(self.config))
            self.agents[aid] = ag
            self.logger.info(f"Spawned {ag.name} at {pos}")

    def _spawn_initial_resources(self):
        self.resources.spawn_resources(self.config.resource_spawn_rate * 10)

    def get_nearby_agents(self, pos: Tuple[int,int], radius: int) -> List[Dict]:
        out=[]
        x,y=pos
        for ag in self.agents.values():
            ax,ay = ag.physical_state.position
            if abs(ax-x)+abs(ay-y) <= radius and ag.physical_state.is_alive():
                out.append({"id":ag.id,"name":ag.name,"position":ag.physical_state.position,"emotion":ag.emotional_state.current_emotion.value})
        return out

    def get_nearby_resources(self, pos: Tuple[int,int], radius: int) -> List[Dict]:
        return self.resources.get_nearby(pos,radius)

    def get_recent_events(self, n: int = 5) -> List[Dict]:
        return [{"type":e.type,"data":e.data} for e in self.events.get_recent(n)]

    def tick(self):
        # 1) Perceive
        perceptions = {aid:ag.perceive(self) for aid,ag in self.agents.items() if ag.physical_state.is_alive()}
        # 2) Decide
        decisions   = {aid:ag._simulate_thinking(perceptions[aid]) for aid,ag in self.agents.items() if ag.physical_state.is_alive()}
        # 3) Execute
        self._process_actions(decisions)
        # 4) Update world
        self._update_environment()
        # 5) Emergence checks
        self._check_cultures()
        # 6) Process events
        for ev in self.events.process():
            self.logger.info(f"Event: {ev.type} {ev.data}")
        self.time += 1

    def _process_actions(self, decisions: Dict[str,Dict]):
        for aid,dec in decisions.items():
            ag = self.agents[aid]
            act = dec.get("action")
            tgt = dec.get("target")
            if act==Action.MOVE:
                x,y = tgt or ag.physical_state.position
                if 0<=x<self.size[0] and 0<=y<self.size[1]:
                    ag.physical_state.position = tgt
                    ag.physical_state.energy *= 0.99
            elif act==Action.GATHER:
                pos = tgt or ag.physical_state.position
                res = self.resources.gather(pos)
                if res:
                    ag.physical_state.carrying[res["type"]] += res["amount"]
                    if res["type"]==ResourceType.FOOD:
                        ag.physical_state.energy = min(1.0, ag.physical_state.energy+0.2)
                    ag.create_belief({"type":"resource_discovery","resource_type":res["type"].value})
            elif act==Action.COMMUNICATE and tgt in self.agents:
                msg = ag.communicate("share_feeling", self.agents[tgt])
                interp = self.agents[tgt].interpret_communication(msg["utterance"], ag)
                self.events.add(Event("communication",{"from":ag.name,"to":self.agents[tgt].name,"understood":interp["understood"]}))
            elif act==Action.CREATE_SYMBOL:
                vec = np.random.rand(8)
                sym = ag.language.create_symbol(vec)
                self.events.add(Event("symbol_created",{"creator":ag.name,"symbol":sym}))
            elif act==Action.CONTEMPLATE:
                c = ag.contemplate_existence()
                if c["type"]=="myth":
                    self.myths.append({"creator":ag.name,"myth":c["content"],"time":self.time})
                    self.events.add(Event("myth_created",{"creator":ag.name,"content":c["content"]}))
            elif act==Action.REPRODUCE and tgt in self.agents:
                child = ag.reproduce(self.agents[tgt], self)
                if child:
                    self.agents[child.id] = child
                    self.events.add(Event("birth",{"child":child.name,"parents":[ag.name,self.agents[tgt].name]}))

    def _update_environment(self):
        self.resources.spawn_resources(self.config.resource_spawn_rate)
        to_remove=[]
        for aid,ag in self.agents.items():
            ag.update(self)
            if not ag.physical_state.is_alive():
                to_remove.append(aid)
        for aid in to_remove:
            ag = self.agents.pop(aid)
            self.events.add(Event("death",{"agent":ag.name,"age":ag.physical_state.age}))

    def _check_cultures(self):
        # simple: if >30% share a practice
        counts=defaultdict(int)
        for ag in self.agents.values():
            for pr in ag.cultural_practices:
                counts[pr]+=1
        for pr,c in counts.items():
            if c>len(self.agents)*0.3 and pr not in self.cultures:
                self.cultures[pr] = {"founded":self.time,"adherents":c}
                self.events.add(Event("culture_emerged",{"culture":pr,"adherents":c}))
