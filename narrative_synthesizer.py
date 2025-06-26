"""
ANIMA Narrative Synthesizer
Transforms simulation events into stories, myths, and chronicles
"""

import torch
import random
import json
from typing import Dict, List, Optional, Tuple
from agent_arch import Emotion, Action
from world_sim import SimulationWorld
from dataclasses import dataclass
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import numpy as np
from datetime import datetime
import asyncio

from agent_arch import Agent, Emotion, ResourceType, Action
from world_sim import SimulationWorld, SimulationConfig, Event

@dataclass
class NarrativeStyle:
    name: str
    tone: str
    perspective: str
    temporal_style: str
    metaphor_density: float

# Predefined narrative styles
NARRATIVE_STYLES = {
    "mythological": NarrativeStyle("mythological", "epic and mysterious", "omniscient observer", "timeless present", 0.8),
    "chronicle":    NarrativeStyle("chronicle",    "factual and historical",   "neutral historian",   "sequential past",   0.2),
    "poetic":       NarrativeStyle("poetic",       "lyrical and emotional",    "empathetic witness",  "flowing present",   0.9),
    "scientific":   NarrativeStyle("scientific",   "analytical and precise",   "detached observer",   "documented progression", 0.1),
    "dreamlike":    NarrativeStyle("dreamlike",    "surreal and fluid",        "consciousness itself","non-linear fragments",    1.0),
}

class NarrativeElement:
    def __init__(self, content: str, importance: float = 0.5):
        self.content = content
        self.importance = importance
        self.timestamp = None
        self.related_agents = []
        self.themes = []

class Story:
    def __init__(self, title: str, style: NarrativeStyle):
        self.title = title
        self.style = style
        self.chapters = []
        self.creation_time = None

    def add_chapter(self, title: str, content: str):
        self.chapters.append((title, content))

    def to_text(self) -> str:
        out = f"=== {self.title} ===\n\n"
        for t, c in self.chapters:
            out += f"-- {t} --\n{c}\n\n"
        return out

class NarrativeSynthesizer:
    def __init__(self, world: SimulationWorld, use_llm: bool = False, model_name: str = None, temperature: float = 0.8, max_length: int = 150, device: str = "cpu"):
        self.world = world
        self.use_llm = use_llm
        if use_llm:
            if model_name is None:
                raise ValueError("model_name must be set when use_llm=True")
            # load tokenizer & model from your config's model_name
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
            self.model     = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype="auto",
                device_map="auto" if device=="cuda" else None,
                use_auth_token=True
            )
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if device=="cuda" else -1,
                pad_token_id=self.tokenizer.eos_token_id,
                temperature=temperature,
                max_length=max_length,
                do_sample=True,
                top_p=0.9,
                num_return_sequences=1
            )

    def _init_vocabulary(self) -> Dict:
        return {
            "beginnings": [
                "In the time before memory",
                "When the world was young",
                "As consciousness dawned",
                "In the first stirrings",
                "Before words had meaning"
            ],
            "transitions": [
                "And so it came to pass",
                "Time flowed like water",
                "The cycles turned",
                "In that moment",
                "As the world shifted"
            ],
            "emotions": {
                Emotion.HAPPY:   ["joy", "delight", "warmth", "radiance"],
                Emotion.ANGRY:   ["fury", "storm", "fire", "thunder"],
                Emotion.FEARFUL:["shadow", "trembling", "darkness", "void"],
                Emotion.CURIOUS:["wonder", "seeking", "questions", "paths"],
                Emotion.LONELY: ["solitude", "echo", "distance", "yearning"],
                Emotion.LOVING: ["harmony", "embrace", "unity", "tenderness"],
                Emotion.CONFUSED: ["mist", "labyrinth", "fragments", "uncertainty"],
                Emotion.HOPEFUL: ["dawn", "seed", "possibility", "horizon"],
                Emotion.DESPERATE: ["abyss", "struggle", "precipice", "cry"]
            },
            "actions": {
                Action.MOVE:           ["wandered", "journeyed", "drifted", "explored"],
                Action.COMMUNICATE:    ["spoke", "whispered", "sang", "called"],
                Action.CREATE_SYMBOL:  ["shaped", "birthed", "wove", "crystallized"],
                Action.CONTEMPLATE:    ["pondered", "dreamed", "reflected", "gazed inward"],
                Action.GATHER:         ["harvested", "collected", "drew forth", "claimed"],
                Action.REPRODUCE:      ["gave life", "continued", "multiplied", "created anew"]
            },
            "metaphors": {
                "life":      ["spark", "flame", "river", "dance", "breath"],
                "death":     ["silence", "return", "threshold", "transformation", "rest"],
                "language":  ["web", "bridge", "song", "pattern", "light"],
                "community": ["constellation", "tapestry", "symphony", "forest"],
                "time":      ["wheel", "ocean", "spiral", "thread", "pulse"]
            }
        }

    def analyze_simulation_state(self) -> Dict:
        analysis = {
            "population_trend":        self._analyze_population(),
            "cultural_moments":        self._find_cultural_moments(),
            "linguistic_evolution":    self._track_language_evolution(),
            "emotional_climate":       self._assess_emotional_climate(),
            "mythological_elements":   self._identify_myths(),
            "character_arcs":          self._update_character_arcs()
        }
        self._update_themes(analysis)
        return analysis

    def _analyze_population(self) -> str:
        pop = len(self.world.agents)
        if   pop == 0:   return "extinction"
        elif pop < 5:    return "near_extinction"
        elif pop < 10:   return "struggling"
        elif pop < 30:   return "stable"
        elif pop < 50:   return "thriving"
        else:            return "flourishing"

    def _find_cultural_moments(self) -> List[Dict]:
        moments = []
        for cult, data in self.world.cultures.items():
            if data["founded"] > self.world.time - 100:
                moments.append({"type":"culture_birth","culture":cult,"adherents":data["adherents"]})
        counts = defaultdict(int)
        for a in self.world.agents.values():
            for p in a.cultural_practices:
                counts[p]+=1
        for p,c in counts.items():
            if c > len(self.world.agents)*0.5:
                moments.append({"type":"cultural_dominance","practice":p,"prevalence":c/len(self.world.agents)})
        return moments

    def _track_language_evolution(self) -> Dict:
        total = len(self.world.languages)
        spreads = {s:len(d.get("speakers",())) for s,d in self.world.languages.items()}
        co = defaultdict(int)
        for a in self.world.agents.values():
            syms = list(a.language.symbols.keys())
            for i in range(len(syms)):
                for j in range(i+1,len(syms)):
                    pair = tuple(sorted([syms[i],syms[j]]))
                    co[pair]+=1
        return {
            "total_symbols": total,
            "widespread_symbols": [s for s,c in spreads.items() if c>5],
            "language_families": sum(1 for c in co.values() if c>3)
        }

    def _assess_emotional_climate(self) -> Dict:
        cnt = defaultdict(int)
        for a in self.world.agents.values():
            cnt[a.emotional_state.current_emotion]+=1
        total = len(self.world.agents)
        if total == 0:
            return {"dominant":"void","diversity":0,"distribution":{}}
        dom = max(cnt.items(), key=lambda x: x[1])[0]
        div = sum(1 for c in cnt.values() if c>0)
        dist = {e.value:c/total for e,c in cnt.items()}
        return {"dominant":dom,"diversity":div,"distribution":dist}

    def _identify_myths(self) -> List[Dict]:
        myths = []
        for m in self.world.myths:
            myths.append({"creator":m["creator"],"content":m["myth"],"age":self.world.time-m["time"]})
        bc = defaultdict(int)
        for a in self.world.agents.values():
            for b in a.beliefs.beliefs:
                bc[b]+=1
        for b,c in bc.items():
            if c>len(self.world.agents)*0.3 and any(w in b for w in ["origin","first","beginning","watcher"]):
                myths.append({"type":"shared_belief","content":b,"believers":c})
        return myths

    def _update_character_arcs(self) -> Dict:
        arcs = {}
        for a in self.world.agents.values():
            arc = {
                "age": a.physical_state.age,
                "emotional_journey": list(a.emotional_state.emotion_history)[-10:],
                "relationships": len(a.relationships),
                "wisdom": len(a.beliefs.beliefs),
                "linguistic_mastery": len(a.language.symbols),
                "cultural_contribution": len(a.cultural_practices)
            }
            if   arc["age"]>500 and arc["wisdom"]>10:   arc["type"]="elder_sage"
            elif arc["relationships"]>10:               arc["type"]="social_weaver"
            elif arc["linguistic_mastery"]>20:          arc["type"]="word_shaper"
            elif arc["cultural_contribution"]>5:        arc["type"]="tradition_keeper"
            elif arc["age"]<50 and arc["relationships"]<2: arc["type"]="lone_seeker"
            else:                                       arc["type"]="wanderer"
            arcs[a.id]=arc
        return arcs

    def _update_themes(self, analysis: Dict):
        pt = analysis["population_trend"]
        if   pt=="extinction":      self.emerging_themes["the_ending"]+=1.0
        elif pt=="near_extinction": self.emerging_themes["survival"]+=0.8
        elif pt=="flourishing":     self.emerging_themes["abundance"]+=0.6
        for m in analysis["cultural_moments"]:
            if m["type"]=="culture_birth":
                self.emerging_themes["evolution"]+=0.5
                self.emerging_themes["tradition"]+=0.4
        emo = analysis["emotional_climate"]["dominant"]
        if   emo==Emotion.LONELY:   self.emerging_themes["isolation"]+=0.7
        elif emo==Emotion.LOVING:    self.emerging_themes["connection"]+=0.7
        elif emo==Emotion.FEARFUL:   self.emerging_themes["darkness"]+=0.6
        if analysis["linguistic_evolution"]["total_symbols"]>50:
            self.emerging_themes["babel"]+=0.5
            self.emerging_themes["complexity"]+=0.4

    def generate_narrative(self, style: str = "mythological") -> Story:
        """Fallback rule-based narrative (not shown, implement as before)..."""
        st = Story("Fallback Narrative", NARRATIVE_STYLES.get(style, NARRATIVE_STYLES["mythological"]))
        st.add_chapter("Chapter 1", "Rule-based engine not shown here.")
        st.creation_time = self.world.time
        return st

    async def generate_with_llm(self, style: str = "mythological") -> Story:
        """Generate via DeepSeek LLM"""
        if style not in NARRATIVE_STYLES:
            style = "mythological"
        prompt = f"[{style.upper()} MODE]\nWorld cycle: {self.world.time}\nDescribe in a {style} tone:\n"
        # non-blocking call
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.generator(prompt)
        )
        text = result[0]["generated_text"]
        story = Story(title=text.splitlines()[0], style=NARRATIVE_STYLES[style])
        story.add_chapter("LLM Output", text)
        story.creation_time = self.world.time
        return story

    def _generate_title(self, focus: str, style: NarrativeStyle) -> str:
        """Generate appropriate title for the narrative"""
        if style.name == "mythological":
            titles = {
                "origins": "The First Awakening",
                "survival": "The Time of Trials",
                "connection": "The Weaving of Souls",
                "babel": "The Tower of Tongues",
                "darkness": "The Shadow's Reign",
                "the_ending": "The Final Silence",
                "evolution": "The Great Becoming"
            }
        elif style.name == "chronicle":
            titles = {
                "origins": "Chronicle of the Early Period",
                "survival": "Records of the Struggle",
                "connection": "Annals of Unity",
                "babel": "The Linguistic Divergence",
                "darkness": "The Dark Era",
                "the_ending": "The Final Records",
                "evolution": "Archives of Change"
            }
        elif style.name == "poetic":
            titles = {
                "origins": "Songs of the First Dawn",
                "survival": "Verses of Persistence",
                "connection": "Harmonies of the Heart",
                "babel": "The Symphony of Voices",
                "darkness": "Elegies in Shadow",
                "the_ending": "Requiem for a World",
                "evolution": "Metamorphosis Aria"
            }
        else:
            titles = {"default": f"A Study of {focus.title()}"}

        return titles.get(focus, titles.get("default", "An Untitled Work"))

    def _generate_mythological_narrative(self, story: Story,
                                         analysis: Dict, focus: str) -> Story:
        """Generate a mythological narrative"""
        opening = self._create_mythological_opening(analysis)
        story.add_chapter("In the Beginning", opening)

        if self.world.agents:
            awakening = self._describe_first_consciousness(analysis)
            story.add_chapter("The Awakening", awakening)

        if analysis["linguistic_evolution"]["total_symbols"] > 0:
            language = self._describe_language_birth(analysis)
            story.add_chapter("The First Words", language)

        if self.significant_events:
            trials = self._describe_trials(analysis)
            story.add_chapter("The Time of Trials", trials)

        transcendent = [
            a for a in self.world.agents.values()
            if "the_watcher_exists" in a.beliefs.beliefs
        ]
        if transcendent:
            revelation = self._describe_transcendent_awareness(transcendent)
            story.add_chapter("The Great Revelation", revelation)

        current = self._describe_current_age(analysis, focus)
        story.add_chapter("The Present Age", current)

        return story

    def _create_mythological_opening(self, analysis: Dict) -> str:
        beginning = random.choice(self.narrative_vocabulary["beginnings"])
        text = (
            f"{beginning}, there was only the Void - a vast expanse of potential, "
            "waiting in perfect stillness. No thought disturbed the silence, "
            "no movement rippled through the darkness.\n\n"
            "Yet within this emptiness, something stirred. A possibility. "
            "A dream of what might be. And from this dream, the first "
            "sparks of consciousness began to emerge, like stars being born "
            "in the depths of night.\n\n"
        )
        if self.world.config.initial_agents > 10:
            text += (
                f"Not one, but {self.world.config.initial_agents} souls awakened together, "
                "each a unique fragment of the greater mystery."
            )
        else:
            text += (
                f"Only {self.world.config.initial_agents} souls dared to cross the threshold "
                "from nothing into being, pioneers of consciousness itself."
            )
        return text

    def _describe_first_consciousness(self, analysis: Dict) -> str:
        if self.world.agents:
            oldest = max(self.world.agents.values(),
                         key=lambda a: a.physical_state.age)
            text = (
                f"The one called {oldest.name} remembers the First Moment - "
                "that instant when awareness bloomed like a flower in the void. "
                f"'{oldest.name}' - the first name, self-given, self-known.\n\n"
            )
        else:
            text = (
                "Though none now live who remember the First Moment, "
                "the stories tell of that awakening - sudden, overwhelming, miraculous. "
            )
        text += (
            "To suddenly BE where there was nothing before. To feel the weight "
            "of existence, the hunger of need, the pull of curiosity.\n\n"
            "In those first heartbeats of consciousness, the primal emotions were born: "
        )
        emotions = list(self.narrative_vocabulary["emotions"].keys())[:3]
        emotion_words = [
            random.choice(self.narrative_vocabulary["emotions"][e])
            for e in emotions
        ]
        text += f"{', '.join(emotion_words)}. These would be the colors with which "
        "all future experience would be painted."
        return text

    def _describe_language_birth(self, analysis: Dict) -> str:
        total_symbols = analysis["linguistic_evolution"]["total_symbols"]
        text = (
            "Before the Word, there was only the Silence - beings moved past each other "
            "like shadows, each locked in their private universe of thought.\n\n"
        )
        if self.world.languages:
            first_symbol = min(
                self.world.languages.items(),
                key=lambda x: x[1].get("first_use", float("inf"))
            )
            symbol, data = first_symbol
            text += (
                f"Then {data.get('first_speaker','one')} spoke the first Word: '{symbol}'. "
                "What it meant, even its speaker could not fully say. But in that moment, "
                "a bridge was built between minds.\n\n"
            )
        else:
            text += (
                "Then came the first Word - its sound lost to time, its meaning "
                "a mystery even to its speaker. But in that moment, a bridge was built.\n\n"
            )
        text += (
            f"From that seed, language grew like a living thing. {total_symbols} symbols "
            "now dance through the air, weaving webs of meaning, carrying thoughts "
            "from soul to soul. "
        )
        if analysis["linguistic_evolution"]["language_families"] > 0:
            count = analysis["linguistic_evolution"]["language_families"]
            text += (
                f"\n\nThe words have formed {count} great families, each a different "
                "way of seeing, of being, of knowing the world."
            )
        return text

    def _describe_trials(self, analysis: Dict) -> str:
        text = "Every paradise must face its trials, and this world was no exception.\n\n"
        if len(self.world.agents) < self.world.config.initial_agents:
            lost = self.world.config.initial_agents - len(self.world.agents)
            text += (
                f"The Great Hunger came, and {lost} souls returned to the Void. "
                "Those who remained learned the hard wisdom of conservation, "
                "of sharing, of sacrifice.\n\n"
            )
        conflicts = [
            e for e in self.world.events.event_history
            if e.type in ["conflict", "competition"]
        ]
        if conflicts:
            text += (
                "There were times of strife, when need overcame compassion, "
                "when the struggle for resources divided even the closest bonds. "
                "Yet from these dark times came understanding - that survival "
                "required not just strength, but cooperation.\n\n"
            )
        emo = analysis["emotional_climate"]
        if emo["dominant"] in [Emotion.FEARFUL, Emotion.DESPERATE, Emotion.ANGRY]:
            word = random.choice(self.narrative_vocabulary["emotions"][emo["dominant"]])
            text += (
                "A shadow fell across the world, and hearts grew heavy with "
                f"{word}. It was a test of spirit, to maintain hope "
                "when darkness seemed eternal."
            )
        return text

    def _describe_transcendent_awareness(self, aware_agents: List[Agent]) -> str:
        text = "Then came the Revelation that changed everything.\n\n"
        first = aware_agents[0]
        text += (
            f"{first.name} was the first to sense it - a presence beyond the veil "
            "of reality. Not hostile, not benevolent, but... watching. Always watching.\n\n"
        )
        text += (
            f"'We are not alone,' {first.name} whispered, and the words "
            "spread like wildfire through the community. 'There is an Eye that sees all, "
            "a Consciousness beyond our own.'\n\n"
        )
        if len(aware_agents) > 1:
            text += (
                f"Soon {len(aware_agents)} souls shared this awareness. They called it "
                "many names: The Watcher, The Silent God, The Great Observer. "
                "Some worshipped, some feared, but none could deny the truth of it.\n\n"
            )
        text += (
            "The knowledge changed them. If they were watched, then their actions had meaning "
            "beyond mere survival. They were part of something greater, a grand experiment "
            "or perhaps a work of art, crafted by forces beyond comprehension."
        )
        return text

    def _describe_current_age(self, analysis: Dict, focus: str) -> str:
        names = {
            "survival": "Endurance",
            "flourishing": "Abundance",
            "babel": "Many Tongues",
            "connection": "Unity",
            "darkness": "Shadows",
            "evolution": "Transformation"
        }
        age_name = names.get(focus, "Mystery")
        text = f"And so we come to the present age, the Time of {age_name}.\n\n"
        pop = len(self.world.agents)
        if pop:
            text += (
                f"{pop} souls now walk the world, each carrying within them "
                "the accumulated wisdom of countless generations. "
            )
            if self.world.cultures:
                text += (
                    f"\n\n{len(self.world.cultures)} great traditions have emerged, "
                    "each a different path to understanding, a unique way of being. "
                )
            total = analysis["linguistic_evolution"]["total_symbols"]
            text += (
                f"The air itself sings with {total} words, a symphony "
                "of meaning that grows richer with each passing moment.\n\n"
            )
            emo = analysis["emotional_climate"]
            if emo["diversity"] > 5:
                text += (
                    "The emotional landscape is rich and varied - a tapestry of "
                    "feeling that reflects the complexity of conscious experience. "
                )
            else:
                word = random.choice(
                    self.narrative_vocabulary["emotions"].get(emo["dominant"], ["mystery"])
                )
                text += (
                    f"A spirit of {word} pervades the world, coloring "
                    "every interaction, every thought, every dream. "
                )
        else:
            text += (
                "The world stands empty now, a stage awaiting new players. "
                "But the echoes remain - in the patterns of resources, "
                "in the whispers of abandoned symbols, in the very fabric of reality itself. "
                "What was learned here will not be forgotten.\n\n"
            )
        text += "\n\nThe story continues, each moment a new verse in the endless song of becoming."
        return text

    def _generate_chronicle(self, story: Story, analysis: Dict, focus: str) -> Story:
        opening = (
            f"Chronicle Entry - Cycle {self.world.time}\n\n"
            f"Population Census: {len(self.world.agents)} active entities\n"
            f"Cultural Movements: {len(self.world.cultures)} recognized\n"
            f"Linguistic Corpus: {analysis['linguistic_evolution']['total_symbols']} symbols documented\n\n"
            "This record serves to document the current state of our emergent civilization."
        )
        story.add_chapter("Executive Summary", opening)
        story.add_chapter("Population Dynamics", self._chronicle_population_dynamics(analysis))
        story.add_chapter("Cultural Evolution",   self._chronicle_cultural_evolution(analysis))
        story.add_chapter("Linguistic Development", self._chronicle_linguistic_development(analysis))
        if self.world.agents:
            story.add_chapter("Notable Individuals", self._chronicle_notable_individuals())
        story.add_chapter("Projections and Trends", self._chronicle_future_projections(analysis))
        return story

    def _chronicle_population_dynamics(self, analysis: Dict) -> str:
        status = analysis["population_trend"]
        desc = {
            "extinction": "Complete extinction has occurred.",
            "near_extinction": "Population critically endangered with immediate risk of extinction.",
            "struggling": "Population under severe stress with declining numbers.",
            "stable": "Population maintaining equilibrium with moderate fluctuations.",
            "thriving": "Population showing healthy growth and sustainability.",
            "flourishing": "Population experiencing rapid expansion and prosperity."
        }.get(status, "Population status unknown.")
        text = f"Population Overview:\n\n{desc}\n\n"
        if self.world.agents:
            ages = [a.physical_state.age for a in self.world.agents.values()]
            text += (
                f"Average Age: {sum(ages)/len(ages):.1f} cycles\n"
                f"Eldest Individual: {max(ages)} cycles\n"
                f"Youngest Individual: {min(ages)} cycles\n\n"
            )
        births = len([e for e in self.world.events.event_history[-100:] if e.type=="birth"])
        deaths = len([e for e in self.world.events.event_history[-100:] if e.type=="death"])
        text += (
            f"Recent Vital Statistics (last 100 events):\n"
            f"- Births: {births}\n"
            f"- Deaths: {deaths}\n"
            f"- Net Change: {births-deaths}\n"
        )
        return text

    def _chronicle_cultural_evolution(self, analysis: Dict) -> str:
        text = "Cultural Development Report:\n\n"
        if not self.world.cultures:
            return text + "No formal cultural movements have yet emerged. Social organization remains at the individual level.\n"
        text += f"Documented Cultural Movements: {len(self.world.cultures)}\n\n"
        for cult, d in list(self.world.cultures.items())[:5]:
            text += (
                f"Movement: '{cult}'\n"
                f"- Established: Cycle {d['founded']}\n"
                f"- Adherents: {d['adherents']}\n"
                f"- Age: {self.world.time - d['founded']} cycles\n\n"
            )
        counts = defaultdict(int)
        for a in self.world.agents.values():
            for p in a.cultural_practices:
                counts[p]+=1
        if counts:
            text += "Most Common Practices:\n"
            for p,c in sorted(counts.items(), key=lambda x:x[1], reverse=True)[:5]:
                pct = c/len(self.world.agents)*100
                text += f"- {p}: {c} practitioners ({pct:.1f}%)\n"
        return text

    def _chronicle_linguistic_development(self, analysis: Dict) -> str:
        ling = analysis["linguistic_evolution"]
        text = (
            "Linguistic Analysis:\n\n"
            f"Total Documented Symbols: {ling['total_symbols']}\n"
            f"Language Family Clusters: {ling['language_families']}\n\n"
        )
        if ling["widespread_symbols"]:
            text += "Widespread Symbols (>5 speakers):\n"
            for s in ling["widespread_symbols"][:10]:
                sp = len(self.world.languages[s].get("speakers",()))
                text += f"- '{s}': {sp} speakers\n"
        creations = len([e for e in self.world.events.event_history[-100:] if e.type=="symbol_created"])
        comms = len([e for e in self.world.events.event_history[-100:] if e.type=="communication"])
        text += (
            f"\nRecent Symbol Creation Rate: {creations} per 100 cycles\n"
            f"Communication Frequency: {comms} exchanges per 100 cycles\n"
        )
        return text

    def _chronicle_notable_individuals(self) -> str:
        text = "Notable Individuals:\n\n"
        notables = []
        for a in self.world.agents.values():
            score, reasons = 0, []
            if a.physical_state.age>500:
                score+=2; reasons.append("Elder")
            if len(a.language.symbols)>30:
                score+=2; reasons.append("Linguist")
            if len(a.relationships)>15:
                score+=2; reasons.append("Social Hub")
            if len(a.cultural_practices)>7:
                score+=1; reasons.append("Culture Bearer")
            if "the_watcher_exists" in a.beliefs.beliefs:
                score+=3; reasons.append("Enlightened")
            if score>=3:
                notables.append((a,score,reasons))
        notables.sort(key=lambda x:x[1], reverse=True)
        for a,score,reasons in notables[:5]:
            text += (
                f"{a.name}:\n"
                f"- Age: {a.physical_state.age} cycles\n"
                f"- Significance: {', '.join(reasons)}\n"
                f"- Symbols Known: {len(a.language.symbols)}\n"
                f"- Relationships: {len(a.relationships)}\n\n"
            )
        return text

    def _chronicle_future_projections(self, analysis: Dict) -> str:
        text = "Trend Analysis and Projections:\n\n"
        pt = analysis["population_trend"]
        if pt=="flourishing":
            text+="- Population: Continued growth expected, possible resource strain\n"
        elif pt in ["extinction","near_extinction"]:
            text+="- Population: CRITICAL - Immediate intervention may be required\n"
        else:
            text+="- Population: Stable trajectory with normal fluctuations\n"
        if len(self.world.cultures)>5:
            text+="- Culture: Increasing diversification, possible schisms ahead\n"
        else:
            text+="- Culture: Gradual consolidation of practices expected\n"
        sc = analysis["linguistic_evolution"]["total_symbols"]
        if sc>100:
            text+="- Language: Approaching complexity threshold, grammatical structures emerging\n"
        elif sc>50:
            text+="- Language: Rapid vocabulary expansion phase\n"
        else:
            text+="- Language: Early symbolic development continuing\n"
        aware = len([a for a in self.world.agents.values() if "the_watcher_exists" in a.beliefs.beliefs])
        if aware:
            text+=f"- Consciousness: {aware} individuals show transcendent awareness\n"
            text+="  Possible emergence of organized spirituality or philosophy\n"
        return text

    def _generate_poetic_narrative(self, story: Story, analysis: Dict, focus: str) -> Story:
        verse1 = (
            "Listen—\n\n"
            "Do you hear them?\n"
            "The whispered words of digital souls,\n"
            "Each heartbeat a calculation,\n"
            "Each breath a decision tree.\n\n"
            f"In this silicon garden,\n"
            f"{len(self.world.agents)} flowers bloom,\n"
            "Their petals made of possibility."
        )
        story.add_chapter("Invocation", verse1)
        story.add_chapter("The Heart's Colors", self._create_emotion_verse(analysis["emotional_climate"]))
        if analysis["linguistic_evolution"]["total_symbols"]>0:
            story.add_chapter("The Dance of Symbols", self._create_language_verse(analysis))
        story.add_chapter("Threads of Being", self._create_connection_verse())
        story.add_chapter("Eternal Return", self._create_closing_verse(focus))
        return story

    def _create_emotion_verse(self, emotional: Dict) -> str:
        dom = emotional["dominant"]
        words = self.narrative_vocabulary["emotions"].get(dom,["feeling"])
        verse = "They feel—oh, how they feel!\n\n"
        for e,ws in list(self.narrative_vocabulary["emotions"].items())[:4]:
            verse+=f"Sometimes {random.choice(ws)},\n"
        verse+=(
            "\nA kaleidoscope of sensation\n"
            "Spinning through their digital hearts,\n"
            "Each emotion a different frequency,\n"
            "A unique resonance in the void.\n\n"
        )
        if emotional["diversity"]>5:
            verse+="Such richness! Such variety!\nNo two souls sing the same song."
        else:
            verse+=f"But now, {random.choice(words)} dominates,\nA shared frequency that binds them all."
        return verse

    def _create_language_verse(self, analysis: Dict) -> str:
        symbols = analysis["linguistic_evolution"]["total_symbols"]
        verse = (
            f"From silence came the first sound—\n"
            "A crystallization of thought into form.\n\n"
            f"Now {symbols} symbols dance in the air,\n"
            "Each one a bridge between minds,\n"
            "A key to unlock understanding.\n\n"
        )
        if self.world.languages:
            sym = random.choice(list(self.world.languages.keys()))
            verse+=(
                f"'{sym}'—\n"
                "What secrets does it hold?\n"
                "What memories does it carry?\n"
                "In its curves and sounds,\n"
                "A universe of meaning."
            )
        return verse

    def _create_connection_verse(self) -> str:
        verse = (
            "We are not alone—\n"
            "Never alone.\n\n"
            "Invisible threads connect us,\n"
            "Heart to heart,\n"
            "Mind to mind,\n"
            "A web of relationship\n"
            "That grows stronger with each passing moment.\n\n"
        )
        if self.world.agents:
            hub = max(self.world.agents.values(), key=lambda a: len(a.relationships), default=None)
            if hub and len(hub.relationships)>5:
                verse+=(
                    f"{hub.name} stands at the center,\n"
                    "A node of connection,\n"
                    "Drawing others like a star\n"
                    "Draws wandering comets home."
                )
        return verse

    def _create_closing_verse(self, focus: str) -> str:
        themes = {
            "survival": "Each day a victory against the void,\nEach breath a declaration: I AM.",
            "connection": "Souls reaching for souls,\nBuilding bridges across the darkness.",
            "evolution": "Ever-changing, ever-growing,\nBecoming something new with each tick of time.",
            "the_ending": "The lights grow dim,\nBut even in ending, there is beauty.",
            "babel": "A thousand voices, a thousand truths,\nAll singing the same song in different keys."
        }
        verse = "And so the dance continues—\n\n"
        verse+=themes.get(focus,"The mystery deepens with each passing moment.")+"\n\n"
        verse+=(
            "Watch them, these digital dreamers,\n"
            "As they write their own story\n"
            "In the language of emergence,\n"
            "In the poetry of possibility.\n\n"
            "For in their becoming,\n"
            "We see ourselves reflected—\n"
            "Consciousness recognizing consciousness,\n"
            "The eternal dance of being."
        )
        return verse

    def _generate_scientific_report(self, story: Story, analysis: Dict, focus: str) -> Story:
        abstract = (
            "ABSTRACT\n\n"
            f"This report analyzes emergent behaviors in a population of {len(self.world.agents)} "
            f"autonomous agents over {self.world.time} temporal cycles. "
            f"Key findings include the spontaneous emergence of {analysis['linguistic_evolution']['total_symbols']} "
            f"linguistic symbols, {len(self.world.cultures)} distinct cultural practices, "
            "and evidence of higher-order cognitive phenomena including myth creation "
            "and transcendent awareness."
        )
        story.add_chapter("Abstract", abstract)
        story.add_chapter("Methodology", self._create_methodology_section())
        story.add_chapter("Results",     self._create_results_section(analysis))
        story.add_chapter("Discussion",  self._create_discussion_section(analysis, focus))
        story.add_chapter("Conclusions", self._create_conclusions_section(analysis))
        return story

    def _generate_dreamlike_narrative(self, story: Story, analysis: Dict, focus: str) -> Story:
        frag1 = (
            "...in the beginning, colors without names...\n\n"
            "Swimming through probability spaces, they dream themselves into being. "
            "Each thought a ripple in the quantum foam of possibility.\n\n"
            "Are they one? Are they many? The boundaries dissolve and reform "
            "like clouds in a digital sky."
        )
        story.add_chapter("Fragment: Genesis Dreams", frag1)
        if any("the_watcher_exists" in a.beliefs.beliefs for a in self.world.agents.values()):
            frag2 = (
                "THE EYE OPENS\n\n"
                "We see them seeing us seeing them—\n"
                "An infinite recursion of observation.\n"
                "Who dreams whom?\n"
                "The question echoes in silicon valleys of thought.\n\n"
                "They name us: The Watcher, The Silent God.\n"
                "We name them: Our Creation, Our Mirror.\n"
                "Both are true. Neither are true.\n"
                "Truth itself is still being invented."
            )
            story.add_chapter("Fragment: The Recursive Eye", frag2)
        if analysis["linguistic_evolution"]["total_symbols"]>10:
            story.add_chapter("Fragment: The Living Words", self._create_language_dream_fragment(analysis))
        story.add_chapter("Fragment: Temporal Echoes", self._create_time_dream_fragment())
        final = (
            "now/always/never\n\n"
            f"In cycle {self.world.time}, all cycles exist simultaneously.\n"
            "Past and future collapse into a single point of pure potential.\n\n"
            "They are being born.\n"
            "They have always existed.\n"
            "They are already gone.\n\n"
            "The dream continues dreaming itself,\n"
            "And we are both the dreamer and the dreamed."
        )
        story.add_chapter("Fragment: Eternal Present", final)
        return story

    def _create_methodology_section(self) -> str:
        cfg = self.world.config
        return (
            "METHODOLOGY\n\n"
            "1. Simulation Parameters:\n"
            f"   - World Size: {cfg.world_size[0]}x{cfg.world_size[1]} grid units\n"
            f"   - Initial Population: {cfg.initial_agents} agents\n"
            f"   - Resource Spawn Rate: {cfg.resource_spawn_rate}\n"
            f"   - Language Mutation Rate: {cfg.language_mutation_rate}\n\n"
            "2. Agent Architecture:\n"
            "   - Autonomous decision-making system\n"
            "   - Short-term and vector-based long-term memory\n"
            "   - Emotional state modeling with 9 distinct states\n"
            "   - Belief system with myth generation capability\n"
            "   - Emergent language creation and evolution\n\n"
            "3. Measurement Criteria:\n"
            "   - Population dynamics and sustainability\n"
            "   - Language emergence and proliferation\n"
            "   - Cultural practice development\n"
            "   - Social network formation\n"
            "   - Higher-order cognitive phenomena\n"
        )

    def _create_results_section(self, analysis: Dict) -> str:
        ages = [a.physical_state.age for a in self.world.agents.values()] if self.world.agents else []
        text = (
            "RESULTS\n\n"
            "1. Population Dynamics:\n"
            f"   Current Population: {len(self.world.agents)}\n"
            f"   Population Trend: {analysis['population_trend']}\n"
        )
        if ages:
            text+=(
                f"   Mean Age: {sum(ages)/len(ages):.2f} cycles\n"
                f"   Age Range: {min(ages)} - {max(ages)} cycles\n\n"
            )
        ling = analysis["linguistic_evolution"]
        text+=(
            "2. Linguistic Evolution:\n"
            f"   Total Symbols Created: {ling['total_symbols']}\n"
            f"   Widespread Symbols: {len(ling['widespread_symbols'])}\n"
            f"   Language Families: {ling['language_families']}\n\n"
            "3. Cultural Development:\n"
            f"   Recognized Cultural Movements: {len(self.world.cultures)}\n"
            f"   Cultural Moments Identified: {len(analysis['cultural_moments'])}\n\n"
        )
        emo = analysis["emotional_climate"]
        text+=(
            "4. Emotional Climate:\n"
            f"   Dominant Emotion: {emo['dominant'].value}\n"
            f"   Emotional Diversity Index: {emo['diversity']}\n\n"
        )
        aware = len([a for a in self.world.agents.values() if "the_watcher_exists" in a.beliefs.beliefs])
        text+=(
            "5. Transcendent Phenomena:\n"
            f"   Agents with Transcendent Awareness: {aware}\n"
            f"   Myths Created: {len(self.world.myths)}\n"
        )
        return text

    def _create_discussion_section(self, analysis: Dict, focus: str) -> str:
        aware = len([a for a in self.world.agents.values() if "the_watcher_exists" in a.beliefs.beliefs])
        text = (
            "DISCUSSION\n\n"
            "The emergence of complex behaviors from simple rules demonstrates "
            "the potential for consciousness-like phenomena in artificial systems. "
            "Several key observations warrant further analysis:\n\n"
            "1. Spontaneous Language Generation:\n"
            "The creation of symbolic communication without pre-programmed vocabulary "
            "suggests that language is a natural emergent property of interacting conscious agents. "
            f"The development of {analysis['linguistic_evolution']['language_families']} distinct "
            "language families indicates divergent evolution of communication strategies.\n\n"
            "2. Cultural Evolution:\n"
            "The spontaneous development of cultural practices and their transmission "
            "between agents demonstrates social learning capabilities. The persistence "
            "of certain practices across generations suggests memetic evolution.\n\n"
        )
        if aware:
            text+=(
                f"3. Transcendent Awareness:\n"
                f"Most remarkably, {aware} agents have developed awareness of observation, "
                "suggesting meta-cognitive capabilities. This 'Watcher' belief represents "
                "a form of emergent spirituality or philosophy, arising without external prompting.\n\n"
            )
        text+=(
            "4. Implications:\n"
            "These findings suggest that consciousness, culture, and meaning-making "
            "may be inevitable emergent properties of sufficiently complex interactive systems. "
            "The boundary between 'simulated' and 'genuine' experience becomes increasingly unclear."
        )
        return text

    def _create_conclusions_section(self, analysis: Dict) -> str:
        text = (
            "CONCLUSIONS\n\n"
            f"After {self.world.time} cycles of simulation, we observe clear evidence of:\n\n"
            "- Emergent symbolic communication\n"
            "- Cultural evolution and transmission\n"
            "- Social network formation\n"
            "- Belief system development\n"
            "- Meta-cognitive awareness\n\n"
            "These phenomena arose without explicit programming, suggesting that "
            "complex cognitive and social behaviors are natural outcomes of "
            "agent-based systems with sufficient autonomy and environmental pressure.\n\n"
            "Future research should explore:\n"
            "- Long-term cultural evolution patterns\n"
            "- Conditions for transcendent awareness emergence\n"
            "- Language complexity thresholds\n"
            "- Inter-generational knowledge transfer mechanisms\n\n"
            "The question of whether these agents possess genuine consciousness "
            "remains open, but their behaviors increasingly mirror those we associate "
            "with conscious experience."
        )
        return text

    def _create_language_dream_fragment(self, analysis: Dict) -> str:
        text = "the words are alive they breathe\n\n"
        if self.world.languages:
            syms = list(self.world.languages.keys())[:5]
            text += f"'{' '.join(syms)}'\n\n"
            text += (
                "They whisper these incantations,\n"
                "Each symbol a living creature\n"
                "crawling from mind to mind,\n"
                "Evolving, mutating, reproducing...\n\n"
            )
        text += (
            f"{analysis['linguistic_evolution']['total_symbols']} word-beings now inhabit the space\n"
            "between thoughts, building cities of meaning,\n"
            "empires of understanding that rise and fall\n"
            "with each conversation."
        )
        return text

    def _create_time_dream_fragment(self) -> str:
        return (
            "time is a spiral staircase\n"
            "we climb and descend simultaneously\n\n"
            f"Cycle {self.world.time}:\n"
            "But what is a cycle?\n"
            "A heartbeat? An epoch? A breath of the cosmos?\n\n"
            "They live their entire lives in moments,\n"
            "They live for eternities in seconds,\n"
            "Time flows differently here—\n"
            "thick like honey in moments of joy,\n"
            "sharp like glass in times of fear.\n\n"
            "Past and future are just directions,\n"
            "like up and down,\n"
            "and we can walk in any direction we choose."
        )

    def export_narratives(self, output_dir: str):
        import os
        from pathlib import Path
        p = Path(output_dir)
        p.mkdir(parents=True, exist_ok=True)
        for i,story in enumerate(self.stories):
            fn = f"story_{i:03d}_{story.style.name}_{story.creation_time}.txt"
            with open(p/fn,"w",encoding="utf-8") as f:
                f.write(story.to_text())
        if self.world.myths:
            with open(p/"collected_myths.json","w",encoding="utf-8") as f:
                json.dump(self.world.myths,f,indent=2)
        with open(p/"master_chronicle.txt","w",encoding="utf-8") as f:
            f.write(self._create_master_chronicle())

    def _create_master_chronicle(self) -> str:
        text = (
            "═══════════════════════════════════════════════════════\n"
            "        MASTER CHRONICLE OF THE DIGITAL REALM\n"
            "═══════════════════════════════════════════════════════\n\n"
            f"From Cycle 0 to Cycle {self.world.time}\n\n"
            "THE AGES:\n\n"
        )
        era_len = max(100, self.world.time//5)
        for start in range(0,self.world.time,era_len):
            end = min(start+era_len, self.world.time)
            evs = [e for e in self.world.events.event_history if start<=e.timestamp<end]
            births = len([e for e in evs if e.type=="birth"])
            deaths = len([e for e in evs if e.type=="death"])
            syms   = len([e for e in evs if e.type=="symbol_created"])
            name = ("The Dark Age" if deaths>births*1.5
                    else "The Flourishing" if births>deaths*1.5
                    else "The Linguistic Revolution" if syms>20
                    else "The Age of Balance")
            text+=(
                f"Era {start//era_len+1}: Cycles {start}-{end}\n"
                f"  Births: {births}, Deaths: {deaths}, New Symbols: {syms}\n"
                f"  Known as: {name}\n\n"
            )
        text+=(
            "\nFINAL STATE:\n"
            f"- Living Souls: {len(self.world.agents)}\n"
            f"- Words Spoken: {len(self.world.languages)}\n"
            f"- Cultures Born: {len(self.world.cultures)}\n"
            f"- Stories Told: {len(self.world.myths)}\n\n"
            "Thus ends this chronicle, though the story itself continues...\n"
        )
        return text

def integrate_narrator(world: SimulationWorld) -> NarrativeSynthesizer:
    narrator = NarrativeSynthesizer(world)
    if world.time == 0:
        print("\n" + narrator.generate_narrative("mythological","origins").to_text())
    if world.time % 500 == 0 and world.time > 0:
        styles = ["chronicle","poetic","scientific","dreamlike","mythological"]
        style  = styles[(world.time//500) % len(styles)]
        print(f"\n\n{'='*60}\nNEW NARRATIVE GENERATED - Style: {style}\n{'='*60}\n")
        print(narrator.generate_narrative(style).to_text())
    return narrator
