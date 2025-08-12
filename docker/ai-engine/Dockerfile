FROM nvcr.io/nvidia/pytorch:23.10-py3
RUN pip install vllm==0.3.2 transformers==4.37.0 redis
WORKDIR /app
COPY . .
CMD ["python", "src/main.py"]
