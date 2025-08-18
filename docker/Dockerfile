FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN TMPDIR=/home/hjt/mountpoint/tmp  pip install -r requirements.txt  -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY app/ /app/
COPY .env /app/.env

# Download embedding model during build using mirror
# ENV HF_ENDPOINT=https://hf-mirror.com
# RUN pip install huggingface_hub && \
#     huggingface-cli download BAAI/bge-large-zh --local-dir /app/models/bge-large-zh

EXPOSE 8010

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8010"]
