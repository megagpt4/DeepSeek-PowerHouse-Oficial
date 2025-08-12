from fastapi import FastAPI
from vllm import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
import os

app = FastAPI()

@app.on_event("startup")
async def startup_event():
    global engine
    engine_args = AsyncEngineArgs(
        model="deepseek-ai/deepseek-coder-6.7b-instruct",
        tensor_parallel_size=int(os.getenv("GPU_COUNT", "1")),
        gpu_memory_utilization=0.95,
        disable_log_stats=True
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)

@app.get("/")
def health_check():
    return {"status": "online", "gpus": os.getenv("GPU_COUNT", "1")}

@app.post("/generate")
async def generate_text(prompt: str, max_tokens: int = 512):
    from vllm.sampling_params import SamplingParams
    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=max_tokens
    )
    
    results_generator = engine.generate(prompt, sampling_params)
    async for request_output in results_generator:
        return {"response": request_output.outputs[0].text}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
