from fastapi import FastAPI
from vllm import EngineArgs, LLMEngine
from redis import Redis
import os

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    global engine, redis_client
    
    # Configuração do motor LLM
    engine_args = EngineArgs(
        model="deepseek-ai/deepseek-coder-6.7b-instruct",  # Start with 6.7B, can change to 33B later if enough VRAM
        tensor_parallel_size=int(os.getenv("GPU_COUNT", "1")),
        gpu_memory_utilization=0.95
    )
    engine = LLMEngine.from_engine_args(engine_args)
    
    # Configuração do Redis
    redis_client = Redis(
        host=os.getenv("REDIS_HOST", "gpu-redis"), 
        port=int(os.getenv("REDIS_PORT", "6379"))
    )

@app.get("/")
def health_check():
    return {
        "status": "online", 
        "model": "DeepSeek-Coder-6.7B",
        "redis": "Connected" if redis_client.ping() else "Failed"
    }

@app.post("/generate")
async def generate_text(prompt: str):
    # For now, a placeholder. We'll implement actual generation later.
    return {
        "response": "Implementação em progresso...",
        "prompt": prompt
    }
