from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from anima_deepseek_agent import initialize_deepseek, SimulationConfig
from world_sim import SimulationWorld
from narrative_synthesizer import NarrativeSynthesizer, NARRATIVE_STYLES

app = FastAPI(title="ANIMA Narrator Service")
app.mount("/", StaticFiles(directory="static", html=True), name="static")

# 1) Build your sim config—
config = SimulationConfig(
    world_size=(50, 50),
    initial_agents=20,
    resource_spawn_rate=0.05,
    time_per_tick=0.1,
    max_memory_size=100,
    language_mutation_rate=0.01,
    death_threshold=0.0,
    reproduction_threshold=0.8,
    # LLM fields:
    use_llm=True,
    model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    temperature=0.8,
    max_response_length=150,
    device="cuda"
)

# 2) Authenticate & load the HF model once
initialize_deepseek(config)

# 3) Create world + LLM narrator
world = SimulationWorld(config)
narrator = NarrativeSynthesizer(
    world,
    use_llm=config.use_llm,
    model_name=config.model_name,
    temperature=config.temperature,
    max_length=config.max_response_length,
    device=config.device
)

@app.on_event("startup")
async def tick_loop():
    import asyncio
    async def loop():
        while True:
            world.tick()
            await asyncio.sleep(config.time_per_tick)
    asyncio.create_task(loop())

@app.get("/narrative/llm/{style}")
async def llm_narrative(style: str):
    if style not in NARRATIVE_STYLES:
        raise HTTPException(404, f"Unknown style {style}")
    story = await narrator.generate_with_llm(style)
    return {"title": story.title, "text": story.to_text()}

@app.get("/styles")
def styles():
    return list(NARRATIVE_STYLES.keys())
