from vllm import EngineArgs, LLMEngine
from redis import Redis

print("🚀 DeepSeek PowerHouse Engine iniciado!")

# Configuração inicial
engine = LLMEngine.from_engine_args(EngineArgs(
    model="deepseek-ai/deepseek-coder-33b-instruct",
    tensor_parallel_size=1,  # Será configurado posteriormente
    gpu_memory_utilization=0.95
))

redis_client = Redis(host='gpu-redis', port=6379)
redis_client.ping()  # Teste básico de conexão

print("✅ Sistemas inicializados com sucesso!")
