FROM nvcr.io/nvidia/pytorch:23.10-py3

RUN pip install vllm==0.4.2 transformers==4.41.0 \
    fastapi uvicorn python-dotenv redis

WORKDIR /app/src
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility


