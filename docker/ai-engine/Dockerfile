FROM nvcr.io/nvidia/pytorch:23.10-py3

# Instalação de dependências críticas
RUN pip install vllm==0.3.2 transformers==4.37.0 \
    redis fastapi uvicorn python-dotenv

WORKDIR /app
COPY . .

# Configuração de variáveis de ambiente
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
CMD ["python3", "src/main.py"]
