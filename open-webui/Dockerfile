# syntax=docker/dockerfile:1

ARG USE_CUDA=false
ARG USE_CUDA_VER=cu121
ARG USE_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
ARG USE_RERANKING_MODEL=""
ARG BUILD_HASH=dev-build
ARG UID=0
ARG GID=0

# ⚙️ Frontend 빌드 (Svelte)
FROM --platform=$BUILDPLATFORM node:22-alpine3.20 AS build
ARG BUILD_HASH
WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY . .
ENV NODE_OPTIONS=--max-old-space-size=4096
ENV APP_BUILD_HASH=${BUILD_HASH}
RUN npm run build

# ⚙️ Backend (Python 3.11)
FROM python:3.11-slim-bookworm AS base

ARG USE_CUDA
ARG USE_CUDA_VER
ARG USE_EMBEDDING_MODEL
ARG USE_RERANKING_MODEL
ARG UID
ARG GID

ENV ENV=prod \
    PORT=8080 \
    USE_CUDA_DOCKER=${USE_CUDA} \
    USE_CUDA_DOCKER_VER=${USE_CUDA_VER} \
    USE_EMBEDDING_MODEL_DOCKER=${USE_EMBEDDING_MODEL} \
    USE_RERANKING_MODEL_DOCKER=${USE_RERANKING_MODEL} \
    WHISPER_MODEL="base" \
    WHISPER_MODEL_DIR="/app/backend/data/cache/whisper/models" \
    RAG_EMBEDDING_MODEL="${USE_EMBEDDING_MODEL}" \
    SENTENCE_TRANSFORMERS_HOME="/app/backend/data/cache/embedding/models" \
    TIKTOKEN_ENCODING_NAME="cl100k_base" \
    TIKTOKEN_CACHE_DIR="/app/backend/data/cache/tiktoken" \
    HF_HOME="/app/backend/data/cache/embedding/models"

WORKDIR /app/backend

# 🔧 사용자 생성 (비 root 환경도 지원)
RUN if [ "$UID" != "0" ]; then \
    addgroup --gid $GID app && \
    adduser --uid $UID --gid $GID --home /home/app --disabled-password app; \
    fi

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential git pandoc gcc curl jq netcat-openbsd ffmpeg libsm6 libxext6 python3-dev && \
    rm -rf /var/lib/apt/lists/*

# 🔄 requirements.txt만 먼저 복사하여 캐시 활용
COPY ./backend/requirements.txt ./requirements.txt

RUN pip3 install --prefer-binary uv && \
    if [ "$USE_CUDA" = "true" ]; then \
        pip3 install --prefer-binary torch torchvision torchaudio \
            --index-url https://download.pytorch.org/whl/${USE_CUDA_VER} --no-cache-dir; \
    else \
        pip3 install --prefer-binary torch torchvision torchaudio \
            --index-url https://download.pytorch.org/whl/cpu --no-cache-dir; \
    fi && \
    uv pip install --system -r requirements.txt --no-cache-dir

# 🔄 나머지 백엔드 복사
COPY ./backend .

# 🔡 모델 캐시 강제 다운로드 (에러 방지용)
RUN python -c "import os; from sentence_transformers import SentenceTransformer; SentenceTransformer(os.environ['RAG_EMBEDDING_MODEL'], device='cpu')" && \
    python -c "import os; from faster_whisper import WhisperModel; WhisperModel(os.environ['WHISPER_MODEL'], device='cpu', compute_type='int8', download_root=os.environ['WHISPER_MODEL_DIR'])" && \
    python -c "import os; import tiktoken; tiktoken.get_encoding(os.environ['TIKTOKEN_ENCODING_NAME'])"

# 📦 프론트엔드 정적 파일 복사
COPY --from=build /app/build /app/build
COPY --from=build /app/CHANGELOG.md /app/CHANGELOG.md
COPY --from=build /app/package.json /app/package.json

EXPOSE 8080

HEALTHCHECK CMD curl --silent --fail http://localhost:${PORT:-8080}/health | jq -ne 'input.status == true' || exit 1

USER $UID:$GID

ARG BUILD_HASH
ENV WEBUI_BUILD_VERSION=${BUILD_HASH}
ENV DOCKER=true

CMD [ "bash", "start.sh" ]
