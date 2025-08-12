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
        model="deepseek-ai/deepseek-coder-33b-instruct",
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
        "model": "DeepSeek-Coder-33B",
        "redis": redis_client.ping()
    }

@app.post("/generate")
async def generate_text(prompt: str):
    return {
        "response": "Implementação em progresso...",
        "prompt": prompt
    }
