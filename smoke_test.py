import asyncio
from world_sim import SimulationWorld, SimulationConfig
from narrative_synthesizer import integrate_narrator, NarrativeSynthesizer

# 1) spin up minimal world
config = SimulationConfig()
world  = SimulationWorld(config)
narr   = integrate_narrator(world)

# 2) turn on llm mode
narr.use_llm = True

# 3) invoke your async method
story = asyncio.run(narr.generate_with_llm("mythological"))

# 4) inspect
print(story.title)
print(story.to_text())
