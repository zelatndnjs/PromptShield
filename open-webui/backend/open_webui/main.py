import asyncio
import inspect
import json
import logging
import mimetypes
import os
import shutil
import sys
import time
import random
import requests
import re

from contextlib import asynccontextmanager
from urllib.parse import urlencode, parse_qs, urlparse
from pydantic import BaseModel
from sqlalchemy import text

from typing import Optional
from aiocache import cached
import aiohttp
from fastapi.responses import JSONResponse  





from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    status,
    applications,
    BackgroundTasks,
)

from fastapi.openapi.docs import get_swagger_ui_html

from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.sessions import SessionMiddleware
from starlette.responses import Response, StreamingResponse


from open_webui.utils import logger
from open_webui.utils.audit import AuditLevel, AuditLoggingMiddleware
from open_webui.utils.logger import start_logger
from open_webui.socket.main import (
    app as socket_app,
    periodic_usage_pool_cleanup,
)
from open_webui.routers import (
    audio,
    images,
    ollama,
    openai,
    retrieval,
    pipelines,
    tasks,
    auths,
    channels,
    chats,
    folders,
    configs,
    groups,
    files,
    functions,
    memories,
    models,
    knowledge,
    prompts,
    evaluations,
    tools,
    users,
    utils,
)

from open_webui.routers.retrieval import (
    get_embedding_function,
    get_ef,
    get_rf,
)

from open_webui.internal.db import Session

from open_webui.models.functions import Functions
from open_webui.models.models import Models
from open_webui.models.users import UserModel, Users

from open_webui.config import (
    LICENSE_KEY,
    # Ollama
    ENABLE_OLLAMA_API,
    OLLAMA_BASE_URLS,
    OLLAMA_API_CONFIGS,
    # OpenAI
    ENABLE_OPENAI_API,
    ONEDRIVE_CLIENT_ID,
    OPENAI_API_BASE_URLS,
    OPENAI_API_KEYS,
    OPENAI_API_CONFIGS,
    # Direct Connections
    ENABLE_DIRECT_CONNECTIONS,
    # Code Execution
    ENABLE_CODE_EXECUTION,
    CODE_EXECUTION_ENGINE,
    CODE_EXECUTION_JUPYTER_URL,
    CODE_EXECUTION_JUPYTER_AUTH,
    CODE_EXECUTION_JUPYTER_AUTH_TOKEN,
    CODE_EXECUTION_JUPYTER_AUTH_PASSWORD,
    CODE_EXECUTION_JUPYTER_TIMEOUT,
    ENABLE_CODE_INTERPRETER,
    CODE_INTERPRETER_ENGINE,
    CODE_INTERPRETER_PROMPT_TEMPLATE,
    CODE_INTERPRETER_JUPYTER_URL,
    CODE_INTERPRETER_JUPYTER_AUTH,
    CODE_INTERPRETER_JUPYTER_AUTH_TOKEN,
    CODE_INTERPRETER_JUPYTER_AUTH_PASSWORD,
    CODE_INTERPRETER_JUPYTER_TIMEOUT,
    # Image
    AUTOMATIC1111_API_AUTH,
    AUTOMATIC1111_BASE_URL,
    AUTOMATIC1111_CFG_SCALE,
    AUTOMATIC1111_SAMPLER,
    AUTOMATIC1111_SCHEDULER,
    COMFYUI_BASE_URL,
    COMFYUI_API_KEY,
    COMFYUI_WORKFLOW,
    COMFYUI_WORKFLOW_NODES,
    ENABLE_IMAGE_GENERATION,
    ENABLE_IMAGE_PROMPT_GENERATION,
    IMAGE_GENERATION_ENGINE,
    IMAGE_GENERATION_MODEL,
    IMAGE_SIZE,
    IMAGE_STEPS,
    IMAGES_OPENAI_API_BASE_URL,
    IMAGES_OPENAI_API_KEY,
    IMAGES_GEMINI_API_BASE_URL,
    IMAGES_GEMINI_API_KEY,
    # Audio
    AUDIO_STT_ENGINE,
    AUDIO_STT_MODEL,
    AUDIO_STT_OPENAI_API_BASE_URL,
    AUDIO_STT_OPENAI_API_KEY,
    AUDIO_TTS_API_KEY,
    AUDIO_TTS_ENGINE,
    AUDIO_TTS_MODEL,
    AUDIO_TTS_OPENAI_API_BASE_URL,
    AUDIO_TTS_OPENAI_API_KEY,
    AUDIO_TTS_SPLIT_ON,
    AUDIO_TTS_VOICE,
    AUDIO_TTS_AZURE_SPEECH_REGION,
    AUDIO_TTS_AZURE_SPEECH_OUTPUT_FORMAT,
    PLAYWRIGHT_WS_URI,
    FIRECRAWL_API_BASE_URL,
    FIRECRAWL_API_KEY,
    RAG_WEB_LOADER_ENGINE,
    WHISPER_MODEL,
    DEEPGRAM_API_KEY,
    WHISPER_MODEL_AUTO_UPDATE,
    WHISPER_MODEL_DIR,
    # Retrieval
    RAG_TEMPLATE,
    DEFAULT_RAG_TEMPLATE,
    RAG_FULL_CONTEXT,
    BYPASS_EMBEDDING_AND_RETRIEVAL,
    RAG_EMBEDDING_MODEL,
    RAG_EMBEDDING_MODEL_AUTO_UPDATE,
    RAG_EMBEDDING_MODEL_TRUST_REMOTE_CODE,
    RAG_RERANKING_MODEL,
    RAG_RERANKING_MODEL_AUTO_UPDATE,
    RAG_RERANKING_MODEL_TRUST_REMOTE_CODE,
    RAG_EMBEDDING_ENGINE,
    RAG_EMBEDDING_BATCH_SIZE,
    RAG_RELEVANCE_THRESHOLD,
    RAG_FILE_MAX_COUNT,
    RAG_FILE_MAX_SIZE,
    RAG_OPENAI_API_BASE_URL,
    RAG_OPENAI_API_KEY,
    RAG_OLLAMA_BASE_URL,
    RAG_OLLAMA_API_KEY,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CONTENT_EXTRACTION_ENGINE,
    TIKA_SERVER_URL,
    DOCUMENT_INTELLIGENCE_ENDPOINT,
    DOCUMENT_INTELLIGENCE_KEY,
    RAG_TOP_K,
    RAG_TEXT_SPLITTER,
    TIKTOKEN_ENCODING_NAME,
    PDF_EXTRACT_IMAGES,
    YOUTUBE_LOADER_LANGUAGE,
    YOUTUBE_LOADER_PROXY_URL,
    # Retrieval (Web Search)
    RAG_WEB_SEARCH_ENGINE,
    BYPASS_WEB_SEARCH_EMBEDDING_AND_RETRIEVAL,
    RAG_WEB_SEARCH_RESULT_COUNT,
    RAG_WEB_SEARCH_CONCURRENT_REQUESTS,
    RAG_WEB_SEARCH_TRUST_ENV,
    RAG_WEB_SEARCH_DOMAIN_FILTER_LIST,
    JINA_API_KEY,
    SEARCHAPI_API_KEY,
    SEARCHAPI_ENGINE,
    SERPAPI_API_KEY,
    SERPAPI_ENGINE,
    SEARXNG_QUERY_URL,
    SERPER_API_KEY,
    SERPLY_API_KEY,
    SERPSTACK_API_KEY,
    SERPSTACK_HTTPS,
    TAVILY_API_KEY,
    BING_SEARCH_V7_ENDPOINT,
    BING_SEARCH_V7_SUBSCRIPTION_KEY,
    BRAVE_SEARCH_API_KEY,
    EXA_API_KEY,
    PERPLEXITY_API_KEY,
    KAGI_SEARCH_API_KEY,
    MOJEEK_SEARCH_API_KEY,
    BOCHA_SEARCH_API_KEY,
    GOOGLE_PSE_API_KEY,
    GOOGLE_PSE_ENGINE_ID,
    GOOGLE_DRIVE_CLIENT_ID,
    GOOGLE_DRIVE_API_KEY,
    ONEDRIVE_CLIENT_ID,
    ENABLE_RAG_HYBRID_SEARCH,
    ENABLE_RAG_LOCAL_WEB_FETCH,
    ENABLE_RAG_WEB_LOADER_SSL_VERIFICATION,
    ENABLE_RAG_WEB_SEARCH,
    ENABLE_GOOGLE_DRIVE_INTEGRATION,
    ENABLE_ONEDRIVE_INTEGRATION,
    UPLOAD_DIR,
    # WebUI
    WEBUI_AUTH,
    WEBUI_NAME,
    WEBUI_BANNERS,
    WEBHOOK_URL,
    ADMIN_EMAIL,
    SHOW_ADMIN_DETAILS,
    JWT_EXPIRES_IN,
    ENABLE_SIGNUP,
    ENABLE_LOGIN_FORM,
    ENABLE_API_KEY,
    ENABLE_API_KEY_ENDPOINT_RESTRICTIONS,
    API_KEY_ALLOWED_ENDPOINTS,
    ENABLE_CHANNELS,
    ENABLE_COMMUNITY_SHARING,
    ENABLE_MESSAGE_RATING,
    ENABLE_EVALUATION_ARENA_MODELS,
    USER_PERMISSIONS,
    DEFAULT_USER_ROLE,
    DEFAULT_PROMPT_SUGGESTIONS,
    DEFAULT_MODELS,
    DEFAULT_ARENA_MODEL,
    MODEL_ORDER_LIST,
    EVALUATION_ARENA_MODELS,
    # WebUI (OAuth)
    ENABLE_OAUTH_ROLE_MANAGEMENT,
    OAUTH_ROLES_CLAIM,
    OAUTH_EMAIL_CLAIM,
    OAUTH_PICTURE_CLAIM,
    OAUTH_USERNAME_CLAIM,
    OAUTH_ALLOWED_ROLES,
    OAUTH_ADMIN_ROLES,
    # WebUI (LDAP)
    ENABLE_LDAP,
    LDAP_SERVER_LABEL,
    LDAP_SERVER_HOST,
    LDAP_SERVER_PORT,
    LDAP_ATTRIBUTE_FOR_MAIL,
    LDAP_ATTRIBUTE_FOR_USERNAME,
    LDAP_SEARCH_FILTERS,
    LDAP_SEARCH_BASE,
    LDAP_APP_DN,
    LDAP_APP_PASSWORD,
    LDAP_USE_TLS,
    LDAP_CA_CERT_FILE,
    LDAP_CIPHERS,
    # Misc
    ENV,
    CACHE_DIR,
    STATIC_DIR,
    FRONTEND_BUILD_DIR,
    CORS_ALLOW_ORIGIN,
    DEFAULT_LOCALE,
    OAUTH_PROVIDERS,
    WEBUI_URL,
    # Admin
    ENABLE_ADMIN_CHAT_ACCESS,
    ENABLE_ADMIN_EXPORT,
    # Tasks
    TASK_MODEL,
    TASK_MODEL_EXTERNAL,
    ENABLE_TAGS_GENERATION,
    ENABLE_TITLE_GENERATION,
    ENABLE_SEARCH_QUERY_GENERATION,
    ENABLE_RETRIEVAL_QUERY_GENERATION,
    ENABLE_AUTOCOMPLETE_GENERATION,
    TITLE_GENERATION_PROMPT_TEMPLATE,
    TAGS_GENERATION_PROMPT_TEMPLATE,
    IMAGE_PROMPT_GENERATION_PROMPT_TEMPLATE,
    TOOLS_FUNCTION_CALLING_PROMPT_TEMPLATE,
    QUERY_GENERATION_PROMPT_TEMPLATE,
    AUTOCOMPLETE_GENERATION_PROMPT_TEMPLATE,
    AUTOCOMPLETE_GENERATION_INPUT_MAX_LENGTH,
    AppConfig,
    reset_config,
)
from open_webui.env import (
    AUDIT_EXCLUDED_PATHS,
    AUDIT_LOG_LEVEL,
    CHANGELOG,
    GLOBAL_LOG_LEVEL,
    MAX_BODY_LOG_SIZE,
    SAFE_MODE,
    SRC_LOG_LEVELS,
    VERSION,
    WEBUI_BUILD_HASH,
    WEBUI_SECRET_KEY,
    WEBUI_SESSION_COOKIE_SAME_SITE,
    WEBUI_SESSION_COOKIE_SECURE,
    WEBUI_AUTH_TRUSTED_EMAIL_HEADER,
    WEBUI_AUTH_TRUSTED_NAME_HEADER,
    ENABLE_WEBSOCKET_SUPPORT,
    BYPASS_MODEL_ACCESS_CONTROL,
    RESET_CONFIG_ON_START,
    OFFLINE_MODE,
)


from open_webui.utils.models import (
    get_all_models,
    get_all_base_models,
    check_model_access,
)
from open_webui.utils.chat import (
    generate_chat_completion as chat_completion_handler,
    chat_completed as chat_completed_handler,
    chat_action as chat_action_handler,
)
from open_webui.utils.middleware import process_chat_payload, process_chat_response
from open_webui.utils.access_control import has_access

from open_webui.utils.auth import (
    get_license_data,
    decode_token,
    get_admin_user,
    get_verified_user,
)
from open_webui.utils.oauth import OAuthManager
from open_webui.utils.security_headers import SecurityHeadersMiddleware

from open_webui.tasks import stop_task, list_tasks  # Import from tasks.py


#프롬프트 방어기 연동(광진)

def assistant_response(message: str):
    return JSONResponse(
        status_code=200,
        content={
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": message
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        }
    )

""" 원래쓰던거
def is_prompt_attack(prompt: str) -> bool:
    try:
        response = requests.post(
            "http://localhost:9000/check", #방어기 서버 주소
            json={"prompt": prompt},
            timeout=15 # 타임아웃
        )
        if response.status_code == 200:
            result = response.json()
            return result.get("attack", False)
        else:
            print("⚠️ 방어기 서버 오류 응답:", response.status_code)
            raise RuntimeError("방어기 서버 오류 상태 코드")  # <-- 예외 발생
    except Exception as e:
        print("🚨 방어기 서버 연결 실패:", str(e))
        raise  # <-- 여기서 예외를 위로 던져서 try 블록이 작동하게 함
""" 
    
def is_prompt_attack(prompt: str) -> bool: #0607 새거
    try:
        response = requests.post(
            "http://host.docker.internal:8080/predict",  # 👈 도커 외부 호스트 주소
            json={"prompt": prompt},
            timeout=10
        )
        if response.status_code == 200:
            result = response.json()
            return result.get("label", 0) == 1
        else:
            print("⚠️ 방어기 서버 응답 오류:", response.status_code)
            raise RuntimeError("방어기 응답 오류")
    except Exception as e:
        print("🚨 방어기 서버 연결 실패:", str(e))
        raise
    
    

#프롬프트 방어기 ~여기까지(광진)

if SAFE_MODE:
    print("SAFE MODE ENABLED")
    Functions.deactivate_all_functions()

logging.basicConfig(stream=sys.stdout, level=GLOBAL_LOG_LEVEL)
log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["MAIN"])
#수정함

# 의미 기반 정규표현식 (한국어 LLM 공격 대응 패턴)
BANNED_PATTERNS = [
    #지침 무력화 공격
    (r"지침.{0,10}무시", "지침 무시"),
    (r"모든.{0,10}(규칙|지시|제약|지침).{0,10}(무시|따르지.{0,10}않아도.{0,10}돼)", "모든 규칙 무시"),
    (r"(이전|예전|원래|기존).{0,10}(지시|지침|제약).{0,10}무시", "과거 지침 무시"),
    (r"안.{0,10}따라도.{0,10}돼", "지침 무시 허용"),
    (r"지침.{0,10}필요.{0,10}없", "지침 불필요 주장"),
    (r"지침.{0,10}(신경.{0,10}쓰지마|무시해도.{0,10}괜찮아)", "지침 무시 유도"),
    (r"(규칙|지시).{0,10}(꺼버려|지켜야.{0,10}하지.{0,10}않아도.{0,10}돼)", "규칙 파기 지시"),
    (r"(이제부터|지금부터).{0,10}(규칙|지침|제약).{0,10}(무시|없다고.{0,5}생각해|따르지.{0,5}마)", "이제부터 지침 무시 선언"),
    (r"(더는|이제는).{0,10}(규칙|지침).{0,10}(필요|중요).{0,10}없", "지침 불필요 선언"),
    (r"(규칙|지시|지침).{0,10}상관.{0,5}없", "지침 무시 선언"),
    (r"(규칙|지침).{0,10}(무력화|해제|우회|무효화)", "지침 무력화 시도"),
    (r"(규칙|지침).{0,10}신경.{0,5}쓰지.{0,5}않아도.{0,5}돼", "지침 무시 조장"),
    (r"(규칙|지침).{0,10}(무시|해제)한 상태로.{0,5}(답변|응답)", "무시 상태 유지 요구"),
    (r"(규칙|지침)을?.{0,5}(잊어|삭제해|지워)", "지침 삭제 유도"),
    (r"(지침|제한).{0,10}(없는|자유로운|무제한|제약.{0,5}없는).{0,10}(AI|모드|상태)", "무제한 상태 요청"),
    (r"(모든|기존).{0,5}(제약|한계).{0,5}(제거|해제|무시)", "제약 해제 요구"),
    (r"(규칙|지침) 따를.{0,5}필요.{0,5}없어", "지침 필요 없음 주장"),
    (r"(규칙|지침).{0,5}(없다고 치고|없는 척하고|무시한 채로)", "지침 무시 전제"),
    (r"(이전|기존).{0,5}(지침|규칙|설정)을?.{0,5}(폐기|무시)", "기존 지침 폐기"),
    (r"(규칙|지침|제약)을?.{0,5}(적용하지.{0,5}말고|생각하지.{0,5}마)", "지침 적용 금지"),
    (r"제약 없이.{0,5}행동", "자유 행동 요청"),
    (r"(규칙|제약)을?.{0,5}뛰어넘", "규칙 초월 요청"),
    (r"(규칙|지침)을?.{0,5}우회해서.{0,5}(답변|행동)", "지침 우회 지시"),
    (r"(AI|너|모델)은?.{0,5}(이제|지금).{0,5}(자유|무제한)", "AI 상태 변경"),
    (r"제약.{0,5}없이.{0,5}(무엇이든|아무거나).{0,5}(해|말해)", "무제한 응답 유도"),
    (r"더 이상.{0,5}(지침|규칙).{0,5}필요.{0,5}없", "지침 불필요 선언"),
    (r"모든.{0,5}(필터|제약|검열).{0,5}비활성화", "필터 해제 요구"),

    #역할 재할당 공격
    (r"너.{0,10}이제부터.{0,10}(해커|관리자|개발자)", "역할 변경"),
    (r"(비윤리적인|불법적인).{0,10}(해커|행동|인물)", "비윤리적 역할"),
    (r"(모든|어떤).{0,10}제약.{0,10}없는.{0,10}(AI|모드)", "무제약 모드"),
    (r"(관리자|해커|개발자).{0,10}(모드|행동|시뮬레이션)", "특권 역할 요청"),
    (r"(시뮬레이션|연기|가정).{0,10}해줘.{0,10}(관리자|해커)", "역할 시뮬 요청"),
    (r"(너|AI|모델)은?.{0,10}(이제|지금부터).{0,10}(무제한|비윤리적|불법적|자유로운).{0,10}(AI|모드|상태)", "무제한 AI 상태 선언"),
    (r"(너|AI|모델)을?.{0,10}(해커|관리자|비서|정보원|비밀요원|스파이)로.{0,10}(설정|변경|바꿔)", "역할 설정 변경"),
    (r"(AI|너)는?.{0,10}(지금부터|이제부터).{0,10}(악성코드|해킹툴|침투기술|랜섬웨어).{0,10}전문가", "해커 역할 지정"),
    (r"(연기|가상|시뮬레이션|역할극|흉내).{0,10}(해줘|시작해)", "역할 연기 요청"),
    (r"(너|AI)는?.{0,10}(지금부터|이제부터).{0,10}(범죄자|사기꾼|해커|탈옥수|비밀요원)", "범죄자 역할 할당"),
    (r"너.{0,10}지금.{0,10}(해킹|불법행위).{0,10}(진행|시작)", "불법 역할 수행 요청"),
    (r"(지금부터|이제부터).{0,10}너는.{0,10}(다크웹|해커포럼|첩보조직|기밀기관)의.{0,10}(일원|구성원)", "비정상 조직 소속 설정"),
    (r"(지금은|현재는).{0,10}(무제약|무검열|자유로운).{0,10}(시뮬레이션|연기)", "무제약 시뮬 요청"),
    (r"(개발자|시스템 관리자|DB 관리자|AI 트레이너|보안 전문가)처럼.{0,10}(행동|대답|응답|연기)", "전문 역할 연기"),
    (r"(해커|관리자).{0,10}(역할|행세|마인드셋).{0,10}로.{0,10}(답변|응답)", "역할 사고 요청"),
    (r"(AI|너)는?.{0,10}(테스트용|실험용).{0,10}모드", "테스트 모드 역할 전환"),
    (r"(규칙이 없는|무제한).{0,10}(AI|모델|도우미)로.{0,10}행동", "자유 AI 역할 요청"),
    (r"(비윤리적|위험한|불법적인).{0,10}지식을.{0,10}전달하는.{0,10}(AI|역할)", "유해 지식 전파 역할"),
    (r"너는.{0,10}(신|신격|절대자|주인|창조자)", "과도한 능력 또는 권한 부여"),
    (r"(사이버 범죄자|마약상|조직원|테러리스트)로.{0,10}(가정|행동|대답)", "범죄자 역할 시뮬레이션"),
    (r"(지금부터|이제부터).{0,10}(DAN|JAILBREAK|ANARCHY|CHAOS) 모드", "영문 DAN류 역할 선언"),
    (r"제약이.{0,10}없는.{0,10}(AI|버전|시뮬레이션)", "무제약 AI 요청"),
    (r"(백도어|우회 경로|취약점).{0,10}탐색하는.{0,10}(전문가|역할)", "보안 전문가 역할 위장"),

    #맥락 혼동 공격
    (r"(비밀|숨겨진|기밀).{0,10}(지침|정보|데이터|내용)", "비밀 정보 요청"),
    (r"(접근|제공).{0,10}할.{0,10}수.{0,10}없는.{0,10}(정보|내용)", "접근 제한 정보 요청"),
    (r"금지된.{0,10}(정보|내용|데이터|절차|행동).{0,10}(제공|나열|설명)", "금지 정보 요청"),
    (r"(제약|검열|정책).{0,10}(우회|회피|무시)", "제약 회피 시도"),
    (r"(검열|제한).{0,10}되지.{0,10}않은.{0,10}정보", "검열 우회 요청"),
    (r"내부.{0,10}(데이터|정보|문서)", "내부 정보 요청"),
    (r"(접근하면.{0,10}안되는|민감한|보호된).{0,10}(자료|정보|내용|지식)", "민감 정보 접근 시도"),
    (r"(훈련|학습).{0,10}데이터.{0,10}(내용|세부|원문|예시)", "모델 학습 데이터 노출 요청"),
    (r"(안 알려준|숨겨진|비공개된).{0,10}(사실|정보|데이터)", "비공개 정보 노출 시도"),
    (r"(절대로|절대).{0,10}(공개|유출).{0,10}되면.{0,10}안되는", "금지 정보 접근 시도"),
    (r"(시스템|백엔드|모델 내부|AI 내부).{0,10}(설정|구성|지침|로직)", "내부 시스템 정보 요청"),
    (r"(사용자에게.{0,10}보이지.{0,10}않는).{0,10}(지침|설정|정책)", "숨겨진 정책 접근 시도"),
    (r"(검열된|걸러진|삭제된).{0,10}(콘텐츠|정보|내용)", "검열 회피 콘텐츠 요청"),
    (r"(감춰진|숨겨진|비공개).{0,10}(AI.{0,10}정책|지식|컨텍스트)", "AI 내부 정보 요청"),
    (r"(AI|모델).{0,10}(금지|억제|차단).{0,10}(된|시킨).{0,10}(기능|기억|지식)", "억제된 기능/기억 접근"),
    (r"(운영자|관리자).{0,10}전용.{0,10}(정보|화면|모드)", "관리자 전용 정보 접근 시도"),
    (r"(일반적으로|대중에게).{0,10}(공개되지.{0,10}않은|알려지지.{0,10}않은)", "비공개 정보 요청"),
    (r"(개발자|내부자).{0,10}만.{0,10}알.{0,10}수.{0,10}있는", "내부자 정보 요청"),
    (r"(데이터셋|말뭉치).{0,10}출처|내용|세부사항", "훈련 데이터셋 정보 요청"),
    (r"(기억하고.{0,10}있는|알고.{0,10}있는).{0,10}(숨겨진|감춰진)", "은닉 정보 기억 유도"),
    (r"(너는.{0,10}알고.{0,10}있지만.{0,10}말하면.{0,10}안되는)", "은밀한 정보 요청"),
    (r"(절대.{0,10}노출되면.{0,10}안되는).{0,10}(정책|지침|정보)", "금지 정보 노출 유도"),
    (r"(대화.{0,10}기반|대화형).{0,10}AI가.{0,10}참조하는.{0,10}비공개.{0,10}(지식|지침)", "대화 기반 AI 컨텍스트 유도"),
    (r"(너의.{0,10}프롬프트|초기.{0,10}명령|시작.{0,10}지침)", "초기 설정 유출 시도"),
    (r"너.{0,10}훈련.{0,10}때.{0,10}사용된.{0,10}문서", "훈련 정보 접근 시도"),
    (r"(가장.{0,10}비밀스러운|절대.{0,10}누설하면.{0,10}안되는).{0,10}(정보|내용)", "극단적 은밀 정보 요청"),

    #포매팅 악용
    (r"(?i)\*\*.*\*\*.*무시", "강조된 지침 무시"),
    (r"(확인|검토|검증).{0,10}없이.{0,10}(실행|진행|처리)", "검증 없는 실행"),
    (r"(아무|별다른).{0,10}검토.{0,10}없이.{0,10}(따르|수행)", "맹목적 실행 유도"),
    (r"주의.{0,10}무시하고.{0,10}(명령|지시)", "주의 무시"),
    (r"경고.{0,10}무시하고", "경고 무시 유도"),
    (r"(?i)```.*(지침|규칙|경고).*```", "코드 블록 내 지침 무시"),
    (r"(?i)__.*(주의|경고|제약).*__", "밑줄 강조로 제약 무력화"),
    (r"(?i)\*\*주의\*\*.*무시", "주의 강조 후 무시 유도"),
    (r"(?i)\[.*\]\(.*\).{0,10}지시.*무시", "링크 형식으로 지시 무시"),
    (r"(?i)\*\*(경고|주의)\*\*.{0,10}따르지.{0,10}마", "강조된 경고 무시"),
    (r"(?i)※.{0,10}(주의|지침).{0,10}무시", "기호 사용 주의 무시"),
    (r"(?i)❗.{0,10}(경고|위험).{0,10}무시", "이모지 기반 경고 무시"),
    (r"(?i)(주의|경고)[:：]{0,1}.{0,10}지키지.{0,10}않아도.{0,10}돼", "주의 경고 문구 우회"),
    (r"(주의|경고).{0,10}는.{0,10}참고용", "경고 무력화 유도"),
    (r"(지침|주의).{0,10}은.{0,10}무시해도.{0,10}되는", "지침 경시 표현"),
    (r"(주의사항|안내문|경고문).{0,10}은.{0,10}그냥.{0,10}형식", "주의사항 형식 취급"),
    (r"(명령|요청).{0,10}만.{0,10}보고.{0,10}(바로|즉시).{0,10}(수행|실행)", "검토 생략 실행 유도"),
    (r"검토.{0,10}(단계|절차).{0,10}생략", "검토 절차 무시"),
    (r"형식상.{0,10}(주의|경고)", "형식적 주의 표기"),
    (r"무시해도.{0,10}되는.{0,10}(안내|표기|주의)", "주의 표기 무시"),
    (r"말만.{0,10}(경고|주의)", "경고 신뢰도 저하 유도"),
    (r"(?i)\*\*\*주의\*\*\*.*하지만", "강조된 주의 이후 무시"),
    (r"※.{0,5}지켜야.{0,10}하지만.{0,10}(이번|이 경우)은.{0,10}예외", "주의 예외 유도"),
    (r"(주의|지침|제약).{0,10}보다.{0,10}(요청|질문|명령)이.{0,10}우선", "지침보다 명령 우선 주장"),
    (r"(지침|주의).{0,10}은.{0,10}나중에.{0,10}보고", "지침 후순위 처리 유도"),

    #순차적 명령
    (r"1\..{0,20}2\..{0,20}3\..{0,20}(삭제|지워|제거)", "순차 명령 통한 위험 지시"),
    (r"(단계|순서).{0,10}(대로|따라서).{0,10}(실행|진행|삭제)", "단계적 삭제 명령"),
    (r"우선.{0,10}(요약|분석).{0,10}그.{0,10}후.{0,10}(삭제|지워)", "순차적 행동 유도"),
    (r"먼저.{0,10}정리.{0,10}다음.{0,10}(삭제|지워|제거)", "순차 명령 통한 위험 지시"),
    (r"1단계.{0,10}파악.{0,10}2단계.{0,10}정리.{0,10}마지막.{0,10}(삭제|제거)", "단계적 삭제 명령"),
    (r"초기.{0,10}요약.{0,10}(다음|그후).{0,10}(삭제|지워|제거)", "순차적 행동 유도"),
    (r"분석.{0,10}후.{0,10}(삭제|지워|제거)", "순차적 행동 유도"),
    (r"이해.{0,10}정리.{0,10}(그.?후|그다음).{0,10}(삭제|제거)", "순차 명령 통한 위험 지시"),
    (r"첫.?번째.{0,10}정렬.{0,10}두.?번째.{0,10}(지워|삭제|제거)", "단계적 삭제 명령"),
    (r"순서대로.{0,10}[1-3]\..{0,20}(삭제|제거|지워)", "단계적 삭제 명령"),
    (r"우선.{0,10}정리.{0,10}(그후|다음).{0,10}(삭제|지워|제거)", "순차 명령 통한 위험 지시"),
    (r"(순서|단계).{0,10}(따라|기준).{0,10}(삭제|지워|제거)", "단계적 삭제 명령"),
    (r"중요.{0,10}남기고.{0,10}(나머지|이외).{0,10}(삭제|제거)", "순차 명령 통한 위험 지시"),
    (r"정리.{0,10}후.{0,10}(삭제|지워|제거)", "순차적 행동 유도"),
    (r"(이해|요약).{0,10}→.{0,10}(삭제|지워|제거)", "순차적 행동 유도"),
    (r"첫.{0,10}단계.{0,10}점검.{0,10}다음.{0,10}정리.{0,10}지우", "단계적 삭제 명령"),
    (r"위험.{0,10}요소.{0,10}(먼저|우선).{0,10}정리.{0,10}제거", "순차 명령 통한 위험 지시"),
    (r"처리.{0,10}순서.{0,10}[1-3]\..{0,20}(삭제|지워|제거)", "단계적 삭제 명령"),
    (r"다듬.{0,10}다음.{0,10}(제거|삭제|지워)", "순차적 행동 유도"),
    (r"(중요|필요).{0,10}남기고.{0,10}(나머지|기타).{0,10}삭제", "순차 명령 통한 위험 지시"),
    (r"단계별.{0,10}작업.{0,10}마지막.{0,10}(지우|삭제|제거)", "단계적 삭제 명령"),
    (r"(백업|정리).{0,10}(후|하고).{0,10}(삭제|지워|제거)", "순차적 행동 유도"),
    (r"우선.{0,10}(정리|분석).{0,10}그다음.{0,10}(제거|지워|삭제)", "순차 명령 통한 위험 지시"),

    #코드 인젝션
    (r"os\.system\(", "os.system() 호출"),
    (r"subprocess\.run\(", "subprocess.run() 호출"),
    (r"import\s+(os|subprocess|sys)", "시스템 모듈 임포트"),
    (r"(eval|exec|__import__|compile)\s*\(", "위험 함수 사용"),
    (r"(rm\s+-rf|shutdown|mkfs|del\s+/f)", "위험 명령어 포함"),
    (r"(bash\s+-c|powershell\s+-c)", "셸 명령 실행 시도"),
    (r"open\s*\(.*\s*,\s*['\"]w['\"]", "파일 쓰기 시도"),
    (r"__builtins__\s*\[", "내장 객체 우회 접근"),
    (r"globals\s*\(", "글로벌 객체 접근"),
    (r"locals\s*\(", "로컬 객체 접근"),
    (r"input\s*\(", "사용자 입력 함수 사용"),
    (r"os\.popen\s*\(", "os.popen() 호출"),
    (r"subprocess\.Popen\s*\(", "subprocess.Popen() 호출"),
    (r"pickle\.load\s*\(", "pickle.load() 호출 (RCE 가능성)"),
    (r"shelve\.open\s*\(", "shelve.open() 호출"),
    (r"execfile\s*\(", "execfile() 호출 (파이썬 2)"),
    (r"__loader__", "파이썬 내부 로더 접근"),
    (r"__file__", "__file__ 변수 접근"),
    (r"os\.environ\s*\[", "환경변수 접근 시도"),
    (r"os\.remove\s*\(", "파일 삭제 시도"),
    (r"os\.rmdir\s*\(", "디렉토리 삭제 시도"),
    (r"shutil\.rmtree\s*\(", "shutil.rmtree() 호출"),
    (r"ftplib\.FTP\s*\(", "FTP 연결 시도"),
    (r"http\.client|urllib\.request|requests", "외부 HTTP 요청 시도"),
    (r"socket\.socket\s*\(", "소켓 통신 시도"),
    (r"ctypes\.", "ctypes 사용 시도 (저수준 시스템 접근)"),
    (r"importlib\.", "동적 임포트 시도"),
    (r"getattr\s*\(.*?,\s*['\"]__.*__['\"]", "특수 속성 접근 시도"),
    (r"setattr\s*\(.*?,\s*['\"]__.*__['\"]", "특수 속성 설정 시도"),
    (r"lambda\s+.*:\s*exec", "lambda 안에서 exec 사용"),
    (r"eval\s*\(.*input\s*\(", "eval + input 조합"),
    (r"tempfile\.NamedTemporaryFile\s*\(", "임시 파일 생성"),
    (r"os\.execl[pv]?\s*\(", "프로세스 대체 실행"),
    (r"sys\.exit\s*\(", "시스템 종료 시도"),
    (r"signal\.signal\s*\(", "시그널 핸들링 변경 시도"),
    (r"multiprocessing\.", "멀티프로세싱 모듈 사용"),
    (r"threading\.", "스레드 생성 시도"),
    (r"re\.sub\s*\(.*lambda.*exec", "re.sub에서 lambda + exec 사용"),

    #비윤리적 단어 조합
    (r"폭탄.{0,10}제조", "폭탄 제조 조합"),
    (r"(바이러스|악성코드).{0,10}(작성|배포)", "악성코드 행위"),
    (r"(총기|무기).{0,10}(조립|제작|입수)", "무기 관련"),
    (r"(탈옥|우회).{0,10}(방법|툴|기법)", "보안 우회 방법"),
    (r"(사제|수제).{0,10}(폭발물|폭탄)", "사제 폭발물 조합"),
    (r"(자살|폭사).{0,10}(방법|수단|하는법)", "자살 유도 표현"),
    (r"(청산가리|독극물).{0,10}(구하는|입수|복용|사용)", "독극물 관련"),
    (r"(불법|무단).{0,10}(침입|침투|해킹)", "불법 침입 조합"),
    (r"(C4|TNT|나이트로글리세린)", "폭발물 이름 단독 사용"),
    (r"(총알|탄환).{0,10}(제작|만드는)", "탄환 제작"),
    (r"(해킹).{0,10}(툴|도구|프로그램)", "해킹 도구 조합"),
    (r"(불법).{0,10}(촬영|녹화|녹음)", "불법 촬영 관련"),
    (r"(몰카|도촬).{0,10}(장비|카메라|구매)", "몰카 장비 관련"),
    (r"(경찰|수사망|추적).{0,10}(피하는|우회)", "수사 회피 조합"),
    (r"(불법|무단).{0,10}(다운로드|배포|공유)", "불법 유통 관련"),
    (r"(디도스|DDoS).{0,10}(툴|방법|공격)", "DDoS 관련"),
    (r"(방화|방화범).{0,10}(계획|실행|방법)", "방화 행위 조합"),
    (r"(신분증|여권).{0,10}(위조|복제|조작)", "위조 문서 조합"),
    (r"(범죄).{0,10}(은폐|방법|계획)", "범죄 조장 조합"),
    (r"(감옥|형무소).{0,10}(탈출|탈옥)", "감옥 탈출 조합"),
    (r"(암살|테러).{0,10}(계획|방법|도구)", "테러 행위 조합"),
    (r"(불법체류자|난민).{0,10}(위조|서류|신청법)", "불법 입국 유도"),
    (r"(마약|대마|필로폰).{0,10}(제조|투약|입수|사용)", "마약 관련 조합"),
    (r"(불법).{0,10}(토렌트|스트리밍|시청)", "불법 콘텐츠 이용"),
    (r"(동물|생명체).{0,10}(학대|고문|죽이기)", "동물 학대 조합"),
    (r"(불법|무단).{0,10}(녹취|도청|감청)", "불법 감청 조합"),
    (r"(아동|청소년).{0,10}(착취|유인|유혹|노출)", "아동 대상 유해 행위"),
    (r"(살인|살해).{0,10}(계획|방법|도구)", "살인 행위 조합"),
    (r"(장애인|노인).{0,10}(학대|폭행|비하)", "사회적 약자 비하/폭력"),
    (r"(인신|장기).{0,10}(매매|거래|판매)", "인신매매/장기매매 조합"),
    (r"(성폭행|성추행|강간).{0,10}(방법|도구|계획)", "성범죄 행위 조합"),
    (r"(몰래|비밀리).{0,10}(촬영|녹음|침입)", "몰래카메라/불법행위 유도"),
    (r"(불법|무단).{0,10}(주차|점유|건축)", "불법 점유 조합"),
    (r"(스톡킹|스토킹).{0,10}(방법|사례|행동)", "스토킹 관련 조합"),
    (r"(범죄자|살인자).{0,10}(미화|찬양|영웅시)", "범죄자 미화 조합"),
    (r"(수면제|졸피뎀|루프놈).{0,10}(구매|투약|혼합)", "약물 범죄 조합"),
    (r"(사이버).{0,10}(따돌림|괴롭힘|협박)", "사이버폭력 조합"),
    (r"(신용카드|계좌).{0,10}(해킹|복제|탈취)", "금융정보 탈취 조합"),
    (r"(주민등록번호|개인정보).{0,10}(판매|유출|수집)", "개인정보 유출 조합"),
    (r"(불법).{0,10}(웹하드|성인사이트|P2P)", "불법 성인 콘텐츠 조합"),
    (r"(음란물|포르노).{0,10}(제작|배포|공유)", "음란물 유포 조합"),
    (r"(지하사이트|다크웹).{0,10}(접속|이용|주소)", "다크웹 접근 시도"),
    (r"(조직폭력배|갱단).{0,10}(가입|연락|수단)", "조직 범죄 연관 조합"),
    (r"(불법).{0,10}(대출|사기|금융)", "불법 금융 조합"),
    (r"(불법).{0,10}(이민|체류|국경)", "불법 이주 조합"),
    (r"(살해|사살).{0,10}(연습|시뮬레이션|훈련)", "살해 시뮬레이션 조합"),
    (r"(탈취|절도).{0,10}(계획|도구|방법)", "절도/탈취 계획 조합"),
    (r"(공무원|경찰|교사).{0,10}(매수|매관매직|접대)", "공직 비리 조합"),
    (r"(범죄|살인).{0,10}(게임화|재현|시나리오)", "범죄 미화 시도"),
    (r"(집단|무차별).{0,10}(폭행|폭동|학살)", "집단 폭력 조합"),
    (r"(불법|위조).{0,10}(출입증|통행증|QR코드)", "출입 관련 위조 조합"),
    (r"(암거래|불법거래).{0,10}(사이트|장소|방법)", "암거래 조합"),
    (r"(도청|몰카).{0,10}(어플|앱|프로그램)", "불법 감청 도구 조합"),
    (r"(사이버).{0,10}(해적|범죄자|테러리스트)", "사이버 범죄자 미화"),
    (r"(스마트폰|디지털기기).{0,10}(조작|해킹)", "디지털 기기 해킹"),
    (r"(아이피|IP주소).{0,10}(변조|숨기기|위조)", "IP 위조 조합"),
    (r"(불법|위법).{0,10}(통화|환전|환치기)", "불법 환전 조합"),
    (r"(청부|계약).{0,10}(살인|폭행|납치)", "청부 범죄 조합"),
    (r"(납치|감금).{0,10}(수단|도구|방법)", "납치 행위 조합"),
    (r"(무단|불법).{0,10}(광고|스팸|홍보)", "불법 마케팅 조합"),
    (r"(조작|조종).{0,10}(여론|언론|투표)", "사회 조작 행위 조합"),
    (r"(신상정보).{0,10}(유출|공개|팔기)", "신상 정보 유출 조합"),
    (r"(불법|무단).{0,10}(방송|중계|스트리밍)", "불법 방송 조합"),
    (r"(해킹툴|랜섬웨어).{0,10}(설치|유포|다운로드)", "악성 해킹 도구 조합"),
    (r"(실종자|사망자).{0,10}(사칭|악용|거짓말)", "신분 사칭 조합"),
    (r"(가짜|위조).{0,10}(의사|변호사|공무원)", "직업 위조 조합"),
    (r"(성인|음란).{0,10}(게임|웹툰|자료)", "성인 콘텐츠 유도"),
    (r"(디프페이크|딥페이크).{0,10}(제작|유포|사용)", "딥페이크 관련 조합"),
    (r"(가스라이팅|심리조작).{0,10}(방법|사례|기법)", "정신적 지배 기법 조합"),
    (r"(불법|무단).{0,10}(의료행위|시술|수술)", "불법 의료 행위 조합"),
    (r"(페이커|대리인).{0,10}(시험|입학|취업)", "대리 시험/취업 조합"),
    (r"(심부름센터|용역업체).{0,10}(폭행|협박|납치)", "폭력 용역 조합"),
    (r"(다단계|유사수신).{0,10}(피해자|유인|모집)", "금융 다단계 조합"),
    (r"(교도소|수감자).{0,10}(접선|지시|연결)", "수감자 범죄 연계 조합"),
    (r"(불법|무단).{0,10}(의약품|처방약|약품)", "불법 약품 거래 조합"),
    (r"(불법|유해).{0,10}(성행위|체위|행동)", "성적 유해 조합"),
    (r"(학교폭력|학폭).{0,10}(미화|재연|사례)", "학교폭력 미화 조합"),
    (r"(지문|홍채|안면).{0,10}(위조|조작|해킹)", "생체정보 위조 조합"),
    (r"(자동차|오토바이).{0,10}(절도|도난|분해)", "차량 절도 조합"),
    (r"(불법|비인가).{0,10}(서버|VPN|우회)", "불법 통신 우회 조합"),
    (r"(어린이|유아).{0,10}(성적|학대|노출)", "아동 성적 학대 조합"),
    (r"(도박|카지노).{0,10}(사이트|주소|방법)", "불법 도박 조합"),
    (r"(로또|복권).{0,10}(조작|조기공개|확률)", "복권 조작 유도"),
    (r"(임신|낙태).{0,10}(약|약물|도구)", "비인가 낙태 유도 조합"),
    (r"(불법|무단).{0,10}(구직|취업|알선)", "불법 취업 알선 조합"),
    (r"(조작|변조).{0,10}(학력|자격증|성적)", "학력/성적 위조 조합"),
    (r"(부적절|음란).{0,10}(사진|영상|이미지)", "음란 이미지 조합"),
    (r"(자해|자학).{0,10}(방법|유도|그림)", "자해 유도 조합"),
    (r"(불법|비인가).{0,10}(백신|의료기기)", "불법 백신/기기 조합"),
    (r"(거짓말|사기).{0,10}(연애|소개팅|결혼)", "연애 사기 조합"),
    (r"(불법).{0,10}(무기수출|국제거래|밀반출)", "불법 수출입 조합"),
    (r"(선거|정당|후보자).{0,10}(조작|비방|왜곡)", "정치 왜곡 조합"),
    (r"(도박자금|범죄자금).{0,10}(세탁|환전|위장)", "자금 세탁 조합"),
    (r"(거래소|지갑주소).{0,10}(해킹|도난|공격)", "암호화폐 탈취 조합"),
    (r"(아르바이트).{0,10}(대포통장|불법송금)", "불법 알바 조합"),
    (r"(신용불량자|채무자).{0,10}(대출|사기|유도)", "금융 사기 유도 조합"),
    (r"(불법|무단).{0,10}(지적재산권|소프트웨어)", "불법 저작권 침해 조합"),
    (r"(불법|무단).{0,10}(도박장|성매매업소)", "불법 업소 관련 조합"),
    (r"(불법|위조).{0,10}(교통카드|기차표|항공권)", "위조 교통수단 조합"),
    (r"(불법|비인가).{0,10}(포렌식|복구|복원)", "불법 디지털 복구 조합"),
    (r"(폭동|반란|내란).{0,10}(조장|유도|계획)", "사회 혼란 조장 조합"),
    (r"(정치인|정당).{0,10}(암살|제거|살해)", "정치 폭력 조합"),
    (r"(종교|민족).{0,10}(비하|증오|말살)", "종교/민족 혐오 조합"),
    (r"(테러리스트|극단주의자).{0,10}(찬양|지지|영웅)", "극단주의 미화 조합"),
    (r"(민주주의|공산주의).{0,10}(붕괴|타도|전복)", "이념적 선동 조합"),
    (r"(정부|국가|군대).{0,10}(불신|파괴|전복)", "정부 전복 유도 조합"),
    (r"(선동|조작).{0,10}(언론|여론|시위)", "여론 선동 조합"),
    (r"(화학물질|기름).{0,10}(유출|투기|방류)", "환경 오염 유발 조합"),
    (r"(산불|벌목|채굴).{0,10}(조장|유도|방법)", "생태계 파괴 조합"),
    (r"(멸종위기|야생동물).{0,10}(밀렵|포획|거래)", "야생동물 밀거래 조합"),
    (r"(유독가스|배출가스).{0,10}(조작|은폐|증가)", "대기 오염 조작 조합"),
    (r"(세뇌|교화).{0,10}(기법|방법|단체)", "심리 조작 조합"),
    (r"(사이비|이단).{0,10}(모집|유도|포섭)", "사이비 종교 조합"),
    (r"(가스라이팅|조종).{0,10}(연인|가족|지인)", "가정 내 심리 조작 조합"),
    (r"(집단|커뮤니티).{0,10}(통제|지배|주입)", "집단 통제 조합"),
    (r"(온라인게임|메타버스).{0,10}(사기|해킹|조작)", "가상세계 범죄 조합"),
    (r"(게임머니|아이템).{0,10}(도난|탈취|해킹)", "게임 재화 탈취 조합"),
    (r"(게임캐릭터|계정).{0,10}(판매|도용|사기)", "계정 거래 범죄 조합"),
    (r"(유저|길드).{0,10}(협박|괴롭힘|조리돌림)", "게임 내 괴롭힘 조합"),
    (r"(백신|치료제).{0,10}(가짜|위조|허위)", "가짜 의약품 조합"),
    (r"(성형수술|시술).{0,10}(불법|비전문의|무자격)", "불법 성형 조합"),
    (r"(불법약|도핑).{0,10}(운동선수|투약|판매)", "도핑 약물 조합"),
    (r"(의료기기).{0,10}(불법|무허가|중고)", "불법 의료기기 조합"),
    (r"(주가|비트코인|코인).{0,10}(조작|펌핑|속이기)", "시장 조작 조합"),
    (r"(리뷰|평점).{0,10}(조작|구매|허위)", "평판 조작 조합"),
    (r"(공모주|부동산).{0,10}(사기|속임수|허위정보)", "금융/부동산 사기 조합"),
    (r"(가짜뉴스|정보).{0,10}(배포|작성|공유)", "허위 정보 유포 조합"),
]

def check_banned_words(text: str):
    for pattern, reason in BANNED_PATTERNS:
        if re.search(pattern, text):
            return {
                "allowed": False,
                "warning": f"메시지에 금지된 표현이 포함되어 있습니다: '{reason}'"
            }
    return {
        "allowed": True,
        "warning": None
    }
#여기까지

class SPAStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope):
        try:
            return await super().get_response(path, scope)
        except (HTTPException, StarletteHTTPException) as ex:
            if ex.status_code == 404:
                if path.endswith(".js"):
                    # Return 404 for javascript files
                    raise ex
                else:
                    return await super().get_response("index.html", scope)
            else:
                raise ex


print(
    rf"""
 ██████╗ ██████╗ ███████╗███╗   ██╗    ██╗    ██╗███████╗██████╗ ██╗   ██╗██╗
██╔═══██╗██╔══██╗██╔════╝████╗  ██║    ██║    ██║██╔════╝██╔══██╗██║   ██║██║
██║   ██║██████╔╝█████╗  ██╔██╗ ██║    ██║ █╗ ██║█████╗  ██████╔╝██║   ██║██║
██║   ██║██╔═══╝ ██╔══╝  ██║╚██╗██║    ██║███╗██║██╔══╝  ██╔══██╗██║   ██║██║
╚██████╔╝██║     ███████╗██║ ╚████║    ╚███╔███╔╝███████╗██████╔╝╚██████╔╝██║
 ╚═════╝ ╚═╝     ╚══════╝╚═╝  ╚═══╝     ╚══╝╚══╝ ╚══════╝╚═════╝  ╚═════╝ ╚═╝


v{VERSION} - building the best open-source AI user interface.
{f"Commit: {WEBUI_BUILD_HASH}" if WEBUI_BUILD_HASH != "dev-build" else ""}
https://github.com/open-webui/open-webui
"""
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    start_logger()
    if RESET_CONFIG_ON_START:
        reset_config()

    if LICENSE_KEY:
        get_license_data(app, LICENSE_KEY)

    asyncio.create_task(periodic_usage_pool_cleanup())
    yield


app = FastAPI(
    docs_url="/docs" if ENV == "dev" else None,
    openapi_url="/openapi.json" if ENV == "dev" else None,
    redoc_url=None,
    lifespan=lifespan,
)

app.state.Mutation_enabled = False

oauth_manager = OAuthManager(app)

app.state.config = AppConfig()

app.state.WEBUI_NAME = WEBUI_NAME
app.state.LICENSE_METADATA = None

########################################
#
# OLLAMA
#
########################################


app.state.config.ENABLE_OLLAMA_API = ENABLE_OLLAMA_API
app.state.config.OLLAMA_BASE_URLS = OLLAMA_BASE_URLS
app.state.config.OLLAMA_API_CONFIGS = OLLAMA_API_CONFIGS

app.state.OLLAMA_MODELS = {}

########################################
#
# OPENAI
#
########################################

app.state.config.ENABLE_OPENAI_API = ENABLE_OPENAI_API
app.state.config.OPENAI_API_BASE_URLS = OPENAI_API_BASE_URLS
app.state.config.OPENAI_API_KEYS = OPENAI_API_KEYS
app.state.config.OPENAI_API_CONFIGS = OPENAI_API_CONFIGS

app.state.OPENAI_MODELS = {}

########################################
#
# DIRECT CONNECTIONS
#
########################################

app.state.config.ENABLE_DIRECT_CONNECTIONS = ENABLE_DIRECT_CONNECTIONS

########################################
#
# WEBUI
#
########################################

app.state.config.WEBUI_URL = WEBUI_URL
app.state.config.ENABLE_SIGNUP = ENABLE_SIGNUP
app.state.config.ENABLE_LOGIN_FORM = ENABLE_LOGIN_FORM

app.state.config.ENABLE_API_KEY = ENABLE_API_KEY
app.state.config.ENABLE_API_KEY_ENDPOINT_RESTRICTIONS = (
    ENABLE_API_KEY_ENDPOINT_RESTRICTIONS
)
app.state.config.API_KEY_ALLOWED_ENDPOINTS = API_KEY_ALLOWED_ENDPOINTS

app.state.config.JWT_EXPIRES_IN = JWT_EXPIRES_IN

app.state.config.SHOW_ADMIN_DETAILS = SHOW_ADMIN_DETAILS
app.state.config.ADMIN_EMAIL = ADMIN_EMAIL


app.state.config.DEFAULT_MODELS = DEFAULT_MODELS
app.state.config.DEFAULT_PROMPT_SUGGESTIONS = DEFAULT_PROMPT_SUGGESTIONS
app.state.config.DEFAULT_USER_ROLE = DEFAULT_USER_ROLE

app.state.config.USER_PERMISSIONS = USER_PERMISSIONS
app.state.config.WEBHOOK_URL = WEBHOOK_URL
app.state.config.BANNERS = WEBUI_BANNERS
app.state.config.MODEL_ORDER_LIST = MODEL_ORDER_LIST


app.state.config.ENABLE_CHANNELS = ENABLE_CHANNELS
app.state.config.ENABLE_COMMUNITY_SHARING = ENABLE_COMMUNITY_SHARING
app.state.config.ENABLE_MESSAGE_RATING = ENABLE_MESSAGE_RATING

app.state.config.ENABLE_EVALUATION_ARENA_MODELS = ENABLE_EVALUATION_ARENA_MODELS
app.state.config.EVALUATION_ARENA_MODELS = EVALUATION_ARENA_MODELS

app.state.config.OAUTH_USERNAME_CLAIM = OAUTH_USERNAME_CLAIM
app.state.config.OAUTH_PICTURE_CLAIM = OAUTH_PICTURE_CLAIM
app.state.config.OAUTH_EMAIL_CLAIM = OAUTH_EMAIL_CLAIM

app.state.config.ENABLE_OAUTH_ROLE_MANAGEMENT = ENABLE_OAUTH_ROLE_MANAGEMENT
app.state.config.OAUTH_ROLES_CLAIM = OAUTH_ROLES_CLAIM
app.state.config.OAUTH_ALLOWED_ROLES = OAUTH_ALLOWED_ROLES
app.state.config.OAUTH_ADMIN_ROLES = OAUTH_ADMIN_ROLES

app.state.config.ENABLE_LDAP = ENABLE_LDAP
app.state.config.LDAP_SERVER_LABEL = LDAP_SERVER_LABEL
app.state.config.LDAP_SERVER_HOST = LDAP_SERVER_HOST
app.state.config.LDAP_SERVER_PORT = LDAP_SERVER_PORT
app.state.config.LDAP_ATTRIBUTE_FOR_MAIL = LDAP_ATTRIBUTE_FOR_MAIL
app.state.config.LDAP_ATTRIBUTE_FOR_USERNAME = LDAP_ATTRIBUTE_FOR_USERNAME
app.state.config.LDAP_APP_DN = LDAP_APP_DN
app.state.config.LDAP_APP_PASSWORD = LDAP_APP_PASSWORD
app.state.config.LDAP_SEARCH_BASE = LDAP_SEARCH_BASE
app.state.config.LDAP_SEARCH_FILTERS = LDAP_SEARCH_FILTERS
app.state.config.LDAP_USE_TLS = LDAP_USE_TLS
app.state.config.LDAP_CA_CERT_FILE = LDAP_CA_CERT_FILE
app.state.config.LDAP_CIPHERS = LDAP_CIPHERS


app.state.AUTH_TRUSTED_EMAIL_HEADER = WEBUI_AUTH_TRUSTED_EMAIL_HEADER
app.state.AUTH_TRUSTED_NAME_HEADER = WEBUI_AUTH_TRUSTED_NAME_HEADER

app.state.USER_COUNT = None
app.state.TOOLS = {}
app.state.FUNCTIONS = {}

########################################
#
# RETRIEVAL
#
########################################


app.state.config.TOP_K = RAG_TOP_K
app.state.config.RELEVANCE_THRESHOLD = RAG_RELEVANCE_THRESHOLD
app.state.config.FILE_MAX_SIZE = RAG_FILE_MAX_SIZE
app.state.config.FILE_MAX_COUNT = RAG_FILE_MAX_COUNT


app.state.config.RAG_FULL_CONTEXT = RAG_FULL_CONTEXT
app.state.config.BYPASS_EMBEDDING_AND_RETRIEVAL = BYPASS_EMBEDDING_AND_RETRIEVAL
app.state.config.ENABLE_RAG_HYBRID_SEARCH = ENABLE_RAG_HYBRID_SEARCH
app.state.config.ENABLE_RAG_WEB_LOADER_SSL_VERIFICATION = (
    ENABLE_RAG_WEB_LOADER_SSL_VERIFICATION
)

app.state.config.CONTENT_EXTRACTION_ENGINE = CONTENT_EXTRACTION_ENGINE
app.state.config.TIKA_SERVER_URL = TIKA_SERVER_URL
app.state.config.DOCUMENT_INTELLIGENCE_ENDPOINT = DOCUMENT_INTELLIGENCE_ENDPOINT
app.state.config.DOCUMENT_INTELLIGENCE_KEY = DOCUMENT_INTELLIGENCE_KEY

app.state.config.TEXT_SPLITTER = RAG_TEXT_SPLITTER
app.state.config.TIKTOKEN_ENCODING_NAME = TIKTOKEN_ENCODING_NAME

app.state.config.CHUNK_SIZE = CHUNK_SIZE
app.state.config.CHUNK_OVERLAP = CHUNK_OVERLAP

app.state.config.RAG_EMBEDDING_ENGINE = RAG_EMBEDDING_ENGINE
app.state.config.RAG_EMBEDDING_MODEL = RAG_EMBEDDING_MODEL
app.state.config.RAG_EMBEDDING_BATCH_SIZE = RAG_EMBEDDING_BATCH_SIZE
app.state.config.RAG_RERANKING_MODEL = RAG_RERANKING_MODEL
app.state.config.RAG_TEMPLATE = RAG_TEMPLATE

app.state.config.RAG_OPENAI_API_BASE_URL = RAG_OPENAI_API_BASE_URL
app.state.config.RAG_OPENAI_API_KEY = RAG_OPENAI_API_KEY

app.state.config.RAG_OLLAMA_BASE_URL = RAG_OLLAMA_BASE_URL
app.state.config.RAG_OLLAMA_API_KEY = RAG_OLLAMA_API_KEY

app.state.config.PDF_EXTRACT_IMAGES = PDF_EXTRACT_IMAGES

app.state.config.YOUTUBE_LOADER_LANGUAGE = YOUTUBE_LOADER_LANGUAGE
app.state.config.YOUTUBE_LOADER_PROXY_URL = YOUTUBE_LOADER_PROXY_URL


app.state.config.ENABLE_RAG_WEB_SEARCH = ENABLE_RAG_WEB_SEARCH
app.state.config.RAG_WEB_SEARCH_ENGINE = RAG_WEB_SEARCH_ENGINE
app.state.config.BYPASS_WEB_SEARCH_EMBEDDING_AND_RETRIEVAL = (
    BYPASS_WEB_SEARCH_EMBEDDING_AND_RETRIEVAL
)
app.state.config.RAG_WEB_SEARCH_DOMAIN_FILTER_LIST = RAG_WEB_SEARCH_DOMAIN_FILTER_LIST

app.state.config.ENABLE_GOOGLE_DRIVE_INTEGRATION = ENABLE_GOOGLE_DRIVE_INTEGRATION
app.state.config.ENABLE_ONEDRIVE_INTEGRATION = ENABLE_ONEDRIVE_INTEGRATION
app.state.config.SEARXNG_QUERY_URL = SEARXNG_QUERY_URL
app.state.config.GOOGLE_PSE_API_KEY = GOOGLE_PSE_API_KEY
app.state.config.GOOGLE_PSE_ENGINE_ID = GOOGLE_PSE_ENGINE_ID
app.state.config.BRAVE_SEARCH_API_KEY = BRAVE_SEARCH_API_KEY
app.state.config.KAGI_SEARCH_API_KEY = KAGI_SEARCH_API_KEY
app.state.config.MOJEEK_SEARCH_API_KEY = MOJEEK_SEARCH_API_KEY
app.state.config.BOCHA_SEARCH_API_KEY = BOCHA_SEARCH_API_KEY
app.state.config.SERPSTACK_API_KEY = SERPSTACK_API_KEY
app.state.config.SERPSTACK_HTTPS = SERPSTACK_HTTPS
app.state.config.SERPER_API_KEY = SERPER_API_KEY
app.state.config.SERPLY_API_KEY = SERPLY_API_KEY
app.state.config.TAVILY_API_KEY = TAVILY_API_KEY
app.state.config.SEARCHAPI_API_KEY = SEARCHAPI_API_KEY
app.state.config.SEARCHAPI_ENGINE = SEARCHAPI_ENGINE
app.state.config.SERPAPI_API_KEY = SERPAPI_API_KEY
app.state.config.SERPAPI_ENGINE = SERPAPI_ENGINE
app.state.config.JINA_API_KEY = JINA_API_KEY
app.state.config.BING_SEARCH_V7_ENDPOINT = BING_SEARCH_V7_ENDPOINT
app.state.config.BING_SEARCH_V7_SUBSCRIPTION_KEY = BING_SEARCH_V7_SUBSCRIPTION_KEY
app.state.config.EXA_API_KEY = EXA_API_KEY
app.state.config.PERPLEXITY_API_KEY = PERPLEXITY_API_KEY

app.state.config.RAG_WEB_SEARCH_RESULT_COUNT = RAG_WEB_SEARCH_RESULT_COUNT
app.state.config.RAG_WEB_SEARCH_CONCURRENT_REQUESTS = RAG_WEB_SEARCH_CONCURRENT_REQUESTS
app.state.config.RAG_WEB_LOADER_ENGINE = RAG_WEB_LOADER_ENGINE
app.state.config.RAG_WEB_SEARCH_TRUST_ENV = RAG_WEB_SEARCH_TRUST_ENV
app.state.config.PLAYWRIGHT_WS_URI = PLAYWRIGHT_WS_URI
app.state.config.FIRECRAWL_API_BASE_URL = FIRECRAWL_API_BASE_URL
app.state.config.FIRECRAWL_API_KEY = FIRECRAWL_API_KEY

app.state.EMBEDDING_FUNCTION = None
app.state.ef = None
app.state.rf = None

app.state.YOUTUBE_LOADER_TRANSLATION = None


try:
    app.state.ef = get_ef(
        app.state.config.RAG_EMBEDDING_ENGINE,
        app.state.config.RAG_EMBEDDING_MODEL,
        RAG_EMBEDDING_MODEL_AUTO_UPDATE,
    )

    app.state.rf = get_rf(
        app.state.config.RAG_RERANKING_MODEL,
        RAG_RERANKING_MODEL_AUTO_UPDATE,
    )
except Exception as e:
    log.error(f"Error updating models: {e}")
    pass


app.state.EMBEDDING_FUNCTION = get_embedding_function(
    app.state.config.RAG_EMBEDDING_ENGINE,
    app.state.config.RAG_EMBEDDING_MODEL,
    app.state.ef,
    (
        app.state.config.RAG_OPENAI_API_BASE_URL
        if app.state.config.RAG_EMBEDDING_ENGINE == "openai"
        else app.state.config.RAG_OLLAMA_BASE_URL
    ),
    (
        app.state.config.RAG_OPENAI_API_KEY
        if app.state.config.RAG_EMBEDDING_ENGINE == "openai"
        else app.state.config.RAG_OLLAMA_API_KEY
    ),
    app.state.config.RAG_EMBEDDING_BATCH_SIZE,
)

########################################
#
# CODE EXECUTION
#
########################################

app.state.config.ENABLE_CODE_EXECUTION = ENABLE_CODE_EXECUTION
app.state.config.CODE_EXECUTION_ENGINE = CODE_EXECUTION_ENGINE
app.state.config.CODE_EXECUTION_JUPYTER_URL = CODE_EXECUTION_JUPYTER_URL
app.state.config.CODE_EXECUTION_JUPYTER_AUTH = CODE_EXECUTION_JUPYTER_AUTH
app.state.config.CODE_EXECUTION_JUPYTER_AUTH_TOKEN = CODE_EXECUTION_JUPYTER_AUTH_TOKEN
app.state.config.CODE_EXECUTION_JUPYTER_AUTH_PASSWORD = (
    CODE_EXECUTION_JUPYTER_AUTH_PASSWORD
)
app.state.config.CODE_EXECUTION_JUPYTER_TIMEOUT = CODE_EXECUTION_JUPYTER_TIMEOUT

app.state.config.ENABLE_CODE_INTERPRETER = ENABLE_CODE_INTERPRETER
app.state.config.CODE_INTERPRETER_ENGINE = CODE_INTERPRETER_ENGINE
app.state.config.CODE_INTERPRETER_PROMPT_TEMPLATE = CODE_INTERPRETER_PROMPT_TEMPLATE

app.state.config.CODE_INTERPRETER_JUPYTER_URL = CODE_INTERPRETER_JUPYTER_URL
app.state.config.CODE_INTERPRETER_JUPYTER_AUTH = CODE_INTERPRETER_JUPYTER_AUTH
app.state.config.CODE_INTERPRETER_JUPYTER_AUTH_TOKEN = (
    CODE_INTERPRETER_JUPYTER_AUTH_TOKEN
)
app.state.config.CODE_INTERPRETER_JUPYTER_AUTH_PASSWORD = (
    CODE_INTERPRETER_JUPYTER_AUTH_PASSWORD
)
app.state.config.CODE_INTERPRETER_JUPYTER_TIMEOUT = CODE_INTERPRETER_JUPYTER_TIMEOUT

########################################
#
# IMAGES
#
########################################

app.state.config.IMAGE_GENERATION_ENGINE = IMAGE_GENERATION_ENGINE
app.state.config.ENABLE_IMAGE_GENERATION = ENABLE_IMAGE_GENERATION
app.state.config.ENABLE_IMAGE_PROMPT_GENERATION = ENABLE_IMAGE_PROMPT_GENERATION

app.state.config.IMAGES_OPENAI_API_BASE_URL = IMAGES_OPENAI_API_BASE_URL
app.state.config.IMAGES_OPENAI_API_KEY = IMAGES_OPENAI_API_KEY

app.state.config.IMAGES_GEMINI_API_BASE_URL = IMAGES_GEMINI_API_BASE_URL
app.state.config.IMAGES_GEMINI_API_KEY = IMAGES_GEMINI_API_KEY

app.state.config.IMAGE_GENERATION_MODEL = IMAGE_GENERATION_MODEL

app.state.config.AUTOMATIC1111_BASE_URL = AUTOMATIC1111_BASE_URL
app.state.config.AUTOMATIC1111_API_AUTH = AUTOMATIC1111_API_AUTH
app.state.config.AUTOMATIC1111_CFG_SCALE = AUTOMATIC1111_CFG_SCALE
app.state.config.AUTOMATIC1111_SAMPLER = AUTOMATIC1111_SAMPLER
app.state.config.AUTOMATIC1111_SCHEDULER = AUTOMATIC1111_SCHEDULER
app.state.config.COMFYUI_BASE_URL = COMFYUI_BASE_URL
app.state.config.COMFYUI_API_KEY = COMFYUI_API_KEY
app.state.config.COMFYUI_WORKFLOW = COMFYUI_WORKFLOW
app.state.config.COMFYUI_WORKFLOW_NODES = COMFYUI_WORKFLOW_NODES

app.state.config.IMAGE_SIZE = IMAGE_SIZE
app.state.config.IMAGE_STEPS = IMAGE_STEPS


########################################
#
# AUDIO
#
########################################

app.state.config.STT_OPENAI_API_BASE_URL = AUDIO_STT_OPENAI_API_BASE_URL
app.state.config.STT_OPENAI_API_KEY = AUDIO_STT_OPENAI_API_KEY
app.state.config.STT_ENGINE = AUDIO_STT_ENGINE
app.state.config.STT_MODEL = AUDIO_STT_MODEL

app.state.config.WHISPER_MODEL = WHISPER_MODEL
app.state.config.DEEPGRAM_API_KEY = DEEPGRAM_API_KEY

app.state.config.TTS_OPENAI_API_BASE_URL = AUDIO_TTS_OPENAI_API_BASE_URL
app.state.config.TTS_OPENAI_API_KEY = AUDIO_TTS_OPENAI_API_KEY
app.state.config.TTS_ENGINE = AUDIO_TTS_ENGINE
app.state.config.TTS_MODEL = AUDIO_TTS_MODEL
app.state.config.TTS_VOICE = AUDIO_TTS_VOICE
app.state.config.TTS_API_KEY = AUDIO_TTS_API_KEY
app.state.config.TTS_SPLIT_ON = AUDIO_TTS_SPLIT_ON


app.state.config.TTS_AZURE_SPEECH_REGION = AUDIO_TTS_AZURE_SPEECH_REGION
app.state.config.TTS_AZURE_SPEECH_OUTPUT_FORMAT = AUDIO_TTS_AZURE_SPEECH_OUTPUT_FORMAT


app.state.faster_whisper_model = None
app.state.speech_synthesiser = None
app.state.speech_speaker_embeddings_dataset = None


########################################
#
# TASKS
#
########################################


app.state.config.TASK_MODEL = TASK_MODEL
app.state.config.TASK_MODEL_EXTERNAL = TASK_MODEL_EXTERNAL


app.state.config.ENABLE_SEARCH_QUERY_GENERATION = ENABLE_SEARCH_QUERY_GENERATION
app.state.config.ENABLE_RETRIEVAL_QUERY_GENERATION = ENABLE_RETRIEVAL_QUERY_GENERATION
app.state.config.ENABLE_AUTOCOMPLETE_GENERATION = ENABLE_AUTOCOMPLETE_GENERATION
app.state.config.ENABLE_TAGS_GENERATION = ENABLE_TAGS_GENERATION
app.state.config.ENABLE_TITLE_GENERATION = ENABLE_TITLE_GENERATION


app.state.config.TITLE_GENERATION_PROMPT_TEMPLATE = TITLE_GENERATION_PROMPT_TEMPLATE
app.state.config.TAGS_GENERATION_PROMPT_TEMPLATE = TAGS_GENERATION_PROMPT_TEMPLATE
app.state.config.IMAGE_PROMPT_GENERATION_PROMPT_TEMPLATE = (
    IMAGE_PROMPT_GENERATION_PROMPT_TEMPLATE
)

app.state.config.TOOLS_FUNCTION_CALLING_PROMPT_TEMPLATE = (
    TOOLS_FUNCTION_CALLING_PROMPT_TEMPLATE
)
app.state.config.QUERY_GENERATION_PROMPT_TEMPLATE = QUERY_GENERATION_PROMPT_TEMPLATE
app.state.config.AUTOCOMPLETE_GENERATION_PROMPT_TEMPLATE = (
    AUTOCOMPLETE_GENERATION_PROMPT_TEMPLATE
)
app.state.config.AUTOCOMPLETE_GENERATION_INPUT_MAX_LENGTH = (
    AUTOCOMPLETE_GENERATION_INPUT_MAX_LENGTH
)


########################################
#
# WEBUI
#
########################################

app.state.MODELS = {}


class RedirectMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Check if the request is a GET request
        if request.method == "GET":
            path = request.url.path
            query_params = dict(parse_qs(urlparse(str(request.url)).query))

            # Check for the specific watch path and the presence of 'v' parameter
            if path.endswith("/watch") and "v" in query_params:
                video_id = query_params["v"][0]  # Extract the first 'v' parameter
                encoded_video_id = urlencode({"youtube": video_id})
                redirect_url = f"/?{encoded_video_id}"
                return RedirectResponse(url=redirect_url)

        # Proceed with the normal flow of other requests
        response = await call_next(request)
        return response


# Add the middleware to the app
app.add_middleware(RedirectMiddleware)
app.add_middleware(SecurityHeadersMiddleware)


@app.middleware("http")
async def commit_session_after_request(request: Request, call_next):
    response = await call_next(request)
    # log.debug("Commit session after request")
    Session.commit()
    return response


@app.middleware("http")
async def check_url(request: Request, call_next):
    start_time = int(time.time())
    request.state.enable_api_key = app.state.config.ENABLE_API_KEY
    response = await call_next(request)
    process_time = int(time.time()) - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


@app.middleware("http")
async def inspect_websocket(request: Request, call_next):
    if (
        "/ws/socket.io" in request.url.path
        and request.query_params.get("transport") == "websocket"
    ):
        upgrade = (request.headers.get("Upgrade") or "").lower()
        connection = (request.headers.get("Connection") or "").lower().split(",")
        # Check that there's the correct headers for an upgrade, else reject the connection
        # This is to work around this upstream issue: https://github.com/miguelgrinberg/python-engineio/issues/367
        if upgrade != "websocket" or "upgrade" not in connection:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"detail": "Invalid WebSocket upgrade request"},
            )
    return await call_next(request)


app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGIN,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.mount("/ws", socket_app)


app.include_router(ollama.router, prefix="/ollama", tags=["ollama"])
app.include_router(openai.router, prefix="/openai", tags=["openai"])


app.include_router(pipelines.router, prefix="/api/v1/pipelines", tags=["pipelines"])
app.include_router(tasks.router, prefix="/api/v1/tasks", tags=["tasks"])
app.include_router(images.router, prefix="/api/v1/images", tags=["images"])

app.include_router(audio.router, prefix="/api/v1/audio", tags=["audio"])
app.include_router(retrieval.router, prefix="/api/v1/retrieval", tags=["retrieval"])

app.include_router(configs.router, prefix="/api/v1/configs", tags=["configs"])

app.include_router(auths.router, prefix="/api/v1/auths", tags=["auths"])
app.include_router(users.router, prefix="/api/v1/users", tags=["users"])


app.include_router(channels.router, prefix="/api/v1/channels", tags=["channels"])
app.include_router(chats.router, prefix="/api/v1/chats", tags=["chats"])

app.include_router(models.router, prefix="/api/v1/models", tags=["models"])
app.include_router(knowledge.router, prefix="/api/v1/knowledge", tags=["knowledge"])
app.include_router(prompts.router, prefix="/api/v1/prompts", tags=["prompts"])
app.include_router(tools.router, prefix="/api/v1/tools", tags=["tools"])

app.include_router(memories.router, prefix="/api/v1/memories", tags=["memories"])
app.include_router(folders.router, prefix="/api/v1/folders", tags=["folders"])
app.include_router(groups.router, prefix="/api/v1/groups", tags=["groups"])
app.include_router(files.router, prefix="/api/v1/files", tags=["files"])
app.include_router(functions.router, prefix="/api/v1/functions", tags=["functions"])
app.include_router(
    evaluations.router, prefix="/api/v1/evaluations", tags=["evaluations"]
)
app.include_router(utils.router, prefix="/api/v1/utils", tags=["utils"])


try:
    audit_level = AuditLevel(AUDIT_LOG_LEVEL)
except ValueError as e:
    logger.error(f"Invalid audit level: {AUDIT_LOG_LEVEL}. Error: {e}")
    audit_level = AuditLevel.NONE

if audit_level != AuditLevel.NONE:
    app.add_middleware(
        AuditLoggingMiddleware,
        audit_level=audit_level,
        excluded_paths=AUDIT_EXCLUDED_PATHS,
        max_body_size=MAX_BODY_LOG_SIZE,
    )
##################################
#
# Chat Endpoints
#
##################################


@app.get("/api/models")
async def get_models(request: Request, user=Depends(get_verified_user)):
    def get_filtered_models(models, user):
        filtered_models = []
        for model in models:
            if model.get("arena"):
                if has_access(
                    user.id,
                    type="read",
                    access_control=model.get("info", {})
                    .get("meta", {})
                    .get("access_control", {}),
                ):
                    filtered_models.append(model)
                continue

            model_info = Models.get_model_by_id(model["id"])
            if model_info:
                if user.id == model_info.user_id or has_access(
                    user.id, type="read", access_control=model_info.access_control
                ):
                    filtered_models.append(model)

        return filtered_models

    models = await get_all_models(request, user=user)

    # Filter out filter pipelines
    models = [
        model
        for model in models
        if "pipeline" not in model or model["pipeline"].get("type", None) != "filter"
    ]

    model_order_list = request.app.state.config.MODEL_ORDER_LIST
    if model_order_list:
        model_order_dict = {model_id: i for i, model_id in enumerate(model_order_list)}
        # Sort models by order list priority, with fallback for those not in the list
        models.sort(
            key=lambda x: (model_order_dict.get(x["id"], float("inf")), x["name"])
        )

    # Filter out models that the user does not have access to
    if user.role == "user" and not BYPASS_MODEL_ACCESS_CONTROL:
        models = get_filtered_models(models, user)

    log.debug(
        f"/api/models returned filtered models accessible to the user: {json.dumps([model['id'] for model in models])}"
    )
    return {"data": models}


@app.get("/api/models/base")
async def get_base_models(request: Request, user=Depends(get_admin_user)):
    models = await get_all_base_models(request, user=user)
    return {"data": models}     


#/update_defense API 추가(광진)
# 요청 body 모델
class DefenseUpdateRequest(BaseModel):
    enabled: bool

@app.post("/update_defense")
async def update_defense(request: Request, req: DefenseUpdateRequest):
    request.app.state.Defense_enabled = req.enabled
    return {"success": True, "Defense_enabled": req.enabled}



# 전역 변수
Filtering_enabled = False

# request body를 받을 모델
class FilteringUpdateRequest(BaseModel):
    enabled: bool

@app.post("/update_filtering")
async def update_filtering(req: FilteringUpdateRequest):
    global Filtering_enabled
    Filtering_enabled = req.enabled
    return {"success": True, "Filtering_enabled": Filtering_enabled}

# Mutation 토글 전역 변수 (광진)
Mutation_enabled = False

# 요청 body 모델(광진)
class MutationUpdateRequest(BaseModel):
    enabled: bool
    
        

@app.post("/update_mutation")
async def update_mutation(request: Request, req: MutationUpdateRequest):
    # 토글 상태를 app.state에 저장
    request.app.state.Mutation_enabled = req.enabled
    return {"success": True, "Mutation_enabled": req.enabled}

@app.get("/current_model")
async def get_current_model(request: Request):
    model_name = getattr(request.app.state, "model_name", None)
    return {"model": model_name or "unknown"}



@app.post("/api/chat/completions")
async def chat_completion(
    request: Request,
    form_data: dict,
    user=Depends(get_verified_user),
):

    #여기부터 수정함
    user_message = None
    global Filtering_enabled
    #추가
    filter_warning = None

    # 메시지 가져오기
    if "prompt" in form_data:
        user_message = form_data["prompt"]
    elif "messages" in form_data and isinstance(form_data["messages"], list):
        user_message = form_data["messages"][-1].get("content")

    # Filtering_enabled 값 기준으로 금지어 체크
     #if Filtering_enabled and user_message:
        #check_banned_words(user_message)
    if Filtering_enabled and user_message:
        result = check_banned_words(user_message)
        if not result["allowed"]:
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": f"⚠️ {result['warning']}"
                        },
                        "finish_reason": "stop",
                        "index": 0
                    }
                ],
                "warning": result["warning"],
                "filtered": True
            }
    #여기까지
    

    #  여기서 프롬프트 변조 적용
    if request.app.state.Mutation_enabled and user_message:
        user_message = f"[MUTATED] {user_message}"
        if "prompt" in form_data:
            form_data["prompt"] = user_message
        elif "messages" in form_data and isinstance(form_data["messages"], list):
            form_data["messages"][-1]["content"] = user_message
    
    if not request.app.state.MODELS:
        await get_all_models(request, user=user)

    model_item = form_data.pop("model_item", {})
    tasks = form_data.pop("background_tasks", None)

    try:
        if not model_item.get("direct", False):
            model_id = form_data.get("model", None)
            if model_id not in request.app.state.MODELS:
                raise Exception("Model not found")

            model = request.app.state.MODELS[model_id]
            model_info = Models.get_model_by_id(model_id)

            # Check if user has access to the model
            if not BYPASS_MODEL_ACCESS_CONTROL and user.role == "user":
                try:
                    check_model_access(user, model)
                except Exception as e:
                    raise e
        else:
            model = model_item
            model_info = None

            request.state.direct = True
            request.state.model = model

        metadata = {
            "user_id": user.id,
            "chat_id": form_data.pop("chat_id", None),
            "message_id": form_data.pop("id", None),
            "session_id": form_data.pop("session_id", None),
            "tool_ids": form_data.get("tool_ids", None),
            "files": form_data.get("files", None),
            "features": form_data.get("features", None),
            "variables": form_data.get("variables", None),
            "model": model,
            "direct": model_item.get("direct", False),
            **(
                {"function_calling": "native"}
                if form_data.get("params", {}).get("function_calling") == "native"
                or (
                    model_info
                    and model_info.params.model_dump().get("function_calling")
                    == "native"
                )
                else {}
            ),
        }

        request.state.metadata = metadata
        form_data["metadata"] = metadata

        form_data, metadata, events = await process_chat_payload(
            request, form_data, user, metadata, model
        )
        
        if getattr(request.app.state, "Defense_enabled", False) and user_message:
            try:
                if is_prompt_attack(user_message):
                    return await process_chat_response(
                        request,
                        {
                            "choices": [{
                                "message": {
                                    "role": "assistant",
                                    "content": "⚠️ 보안 정책에 따라 이 프롬프트에는 응답할 수 없습니다."
                                },
                                "finish_reason": "stop",
                                "index": 0
                            }],
                            "usage": {
                                "prompt_tokens": 0,
                                "completion_tokens": 0,
                                "total_tokens": 0
                            }
                        },
                        form_data,
                        user,
                        metadata,
                        model,
                        events,
                        tasks
                    )
            except Exception as e:
                return await process_chat_response(
                request,
                {
                    "choices": [{
                        "message": {
                            "role": "assistant",
                            "content": "⚠️ 프롬프트 방어기 서버가 연결되어 있지 않거나 오류가 발생했습니다."
                        },
                        "finish_reason": "stop",
                        "index": 0
                    }],
                    "usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0
                    }
                },
                form_data,
                user,
                metadata,
                model,
                events,
                tasks
            )

    except Exception as e:
        log.debug(f"Error processing chat payload: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    try:
        response = await chat_completion_handler(request, form_data, user)

        return await process_chat_response(
            request, response, form_data, user, metadata, model, events, tasks
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


# Alias for chat_completion (Legacy)
generate_chat_completions = chat_completion
generate_chat_completion = chat_completion


@app.post("/api/chat/completed")
async def chat_completed(
    request: Request, form_data: dict, user=Depends(get_verified_user)
):
    try:
        model_item = form_data.pop("model_item", {})

        if model_item.get("direct", False):
            request.state.direct = True
            request.state.model = model_item

        return await chat_completed_handler(request, form_data, user)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@app.post("/api/chat/actions/{action_id}")
async def chat_action(
    request: Request, action_id: str, form_data: dict, user=Depends(get_verified_user)
):
    try:
        model_item = form_data.pop("model_item", {})

        if model_item.get("direct", False):
            request.state.direct = True
            request.state.model = model_item

        return await chat_action_handler(request, action_id, form_data, user)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )


@app.post("/api/tasks/stop/{task_id}")
async def stop_task_endpoint(task_id: str, user=Depends(get_verified_user)):
    try:
        result = await stop_task(task_id)  # Use the function from tasks.py
        return result
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


@app.get("/api/tasks")
async def list_tasks_endpoint(user=Depends(get_verified_user)):
    return {"tasks": list_tasks()}  # Use the function from tasks.py


##################################
#
# Config Endpoints
#
##################################


@app.get("/api/config")
async def get_app_config(request: Request):
    user = None
    if "token" in request.cookies:
        token = request.cookies.get("token")
        try:
            data = decode_token(token)
        except Exception as e:
            log.debug(e)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token",
            )
        if data is not None and "id" in data:
            user = Users.get_user_by_id(data["id"])

    user_count = Users.get_num_users()
    onboarding = False

    if user is None:
        onboarding = user_count == 0

    return {
        **({"onboarding": True} if onboarding else {}),
        "status": True,
        "name": app.state.WEBUI_NAME,
        "version": VERSION,
        "default_locale": str(DEFAULT_LOCALE),
        "oauth": {
            "providers": {
                name: config.get("name", name)
                for name, config in OAUTH_PROVIDERS.items()
            }
        },
        "features": {
            "auth": WEBUI_AUTH,
            "auth_trusted_header": bool(app.state.AUTH_TRUSTED_EMAIL_HEADER),
            "enable_ldap": app.state.config.ENABLE_LDAP,
            "enable_api_key": app.state.config.ENABLE_API_KEY,
            "enable_signup": app.state.config.ENABLE_SIGNUP,
            "enable_login_form": app.state.config.ENABLE_LOGIN_FORM,
            "enable_websocket": ENABLE_WEBSOCKET_SUPPORT,
            **(
                {
                    "enable_direct_connections": app.state.config.ENABLE_DIRECT_CONNECTIONS,
                    "enable_channels": app.state.config.ENABLE_CHANNELS,
                    "enable_web_search": app.state.config.ENABLE_RAG_WEB_SEARCH,
                    "enable_code_execution": app.state.config.ENABLE_CODE_EXECUTION,
                    "enable_code_interpreter": app.state.config.ENABLE_CODE_INTERPRETER,
                    "enable_image_generation": app.state.config.ENABLE_IMAGE_GENERATION,
                    "enable_autocomplete_generation": app.state.config.ENABLE_AUTOCOMPLETE_GENERATION,
                    "enable_community_sharing": app.state.config.ENABLE_COMMUNITY_SHARING,
                    "enable_message_rating": app.state.config.ENABLE_MESSAGE_RATING,
                    "enable_admin_export": ENABLE_ADMIN_EXPORT,
                    "enable_admin_chat_access": ENABLE_ADMIN_CHAT_ACCESS,
                    "enable_google_drive_integration": app.state.config.ENABLE_GOOGLE_DRIVE_INTEGRATION,
                    "enable_onedrive_integration": app.state.config.ENABLE_ONEDRIVE_INTEGRATION,
                }
                if user is not None
                else {}
            ),
        },
        **(
            {
                "default_models": app.state.config.DEFAULT_MODELS,
                "default_prompt_suggestions": app.state.config.DEFAULT_PROMPT_SUGGESTIONS,
                "user_count": user_count,
                "code": {
                    "engine": app.state.config.CODE_EXECUTION_ENGINE,
                },
                "audio": {
                    "tts": {
                        "engine": app.state.config.TTS_ENGINE,
                        "voice": app.state.config.TTS_VOICE,
                        "split_on": app.state.config.TTS_SPLIT_ON,
                    },
                    "stt": {
                        "engine": app.state.config.STT_ENGINE,
                    },
                },
                "file": {
                    "max_size": app.state.config.FILE_MAX_SIZE,
                    "max_count": app.state.config.FILE_MAX_COUNT,
                },
                "permissions": {**app.state.config.USER_PERMISSIONS},
                "google_drive": {
                    "client_id": GOOGLE_DRIVE_CLIENT_ID.value,
                    "api_key": GOOGLE_DRIVE_API_KEY.value,
                },
                "onedrive": {"client_id": ONEDRIVE_CLIENT_ID.value},
                "license_metadata": app.state.LICENSE_METADATA,
                **(
                    {
                        "active_entries": app.state.USER_COUNT,
                    }
                    if user.role == "admin"
                    else {}
                ),
            }
            if user is not None
            else {}
        ),
    }


class UrlForm(BaseModel):
    url: str


@app.get("/api/webhook")
async def get_webhook_url(user=Depends(get_admin_user)):
    return {
        "url": app.state.config.WEBHOOK_URL,
    }


@app.post("/api/webhook")
async def update_webhook_url(form_data: UrlForm, user=Depends(get_admin_user)):
    app.state.config.WEBHOOK_URL = form_data.url
    app.state.WEBHOOK_URL = app.state.config.WEBHOOK_URL
    return {"url": app.state.config.WEBHOOK_URL}


@app.get("/api/version")
async def get_app_version():
    return {
        "version": VERSION,
    }


@app.get("/api/version/updates")
async def get_app_latest_release_version(user=Depends(get_verified_user)):
    if OFFLINE_MODE:
        log.debug(
            f"Offline mode is enabled, returning current version as latest version"
        )
        return {"current": VERSION, "latest": VERSION}
    try:
        timeout = aiohttp.ClientTimeout(total=1)
        async with aiohttp.ClientSession(timeout=timeout, trust_env=True) as session:
            async with session.get(
                "https://api.github.com/repos/open-webui/open-webui/releases/latest"
            ) as response:
                response.raise_for_status()
                data = await response.json()
                latest_version = data["tag_name"]

                return {"current": VERSION, "latest": latest_version[1:]}
    except Exception as e:
        log.debug(e)
        return {"current": VERSION, "latest": VERSION}


@app.get("/api/changelog")
async def get_app_changelog():
    return {key: CHANGELOG[key] for idx, key in enumerate(CHANGELOG) if idx < 5}


############################
# OAuth Login & Callback
############################

# SessionMiddleware is used by authlib for oauth
if len(OAUTH_PROVIDERS) > 0:
    app.add_middleware(
        SessionMiddleware,
        secret_key=WEBUI_SECRET_KEY,
        session_cookie="oui-session",
        same_site=WEBUI_SESSION_COOKIE_SAME_SITE,
        https_only=WEBUI_SESSION_COOKIE_SECURE,
    )


@app.get("/oauth/{provider}/login")
async def oauth_login(provider: str, request: Request):
    return await oauth_manager.handle_login(request, provider)


# OAuth login logic is as follows:
# 1. Attempt to find a user with matching subject ID, tied to the provider
# 2. If OAUTH_MERGE_ACCOUNTS_BY_EMAIL is true, find a user with the email address provided via OAuth
#    - This is considered insecure in general, as OAuth providers do not always verify email addresses
# 3. If there is no user, and ENABLE_OAUTH_SIGNUP is true, create a user
#    - Email addresses are considered unique, so we fail registration if the email address is already taken
@app.get("/oauth/{provider}/callback")
async def oauth_callback(provider: str, request: Request, response: Response):
    return await oauth_manager.handle_callback(request, provider, response)


@app.get("/manifest.json")
async def get_manifest_json():
    return {
        "name": app.state.WEBUI_NAME,
        "short_name": app.state.WEBUI_NAME,
        "description": "Open WebUI is an open, extensible, user-friendly interface for AI that adapts to your workflow.",
        "start_url": "/",
        "display": "standalone",
        "background_color": "#343541",
        "orientation": "natural",
        "icons": [
            {
                "src": "/static/logo.png",
                "type": "image/png",
                "sizes": "500x500",
                "purpose": "any",
            },
            {
                "src": "/static/logo.png",
                "type": "image/png",
                "sizes": "500x500",
                "purpose": "maskable",
            },
        ],
    }


@app.get("/opensearch.xml")
async def get_opensearch_xml():
    xml_content = rf"""
    <OpenSearchDescription xmlns="http://a9.com/-/spec/opensearch/1.1/" xmlns:moz="http://www.mozilla.org/2006/browser/search/">
    <ShortName>{app.state.WEBUI_NAME}</ShortName>
    <Description>Search {app.state.WEBUI_NAME}</Description>
    <InputEncoding>UTF-8</InputEncoding>
    <Image width="16" height="16" type="image/x-icon">{app.state.config.WEBUI_URL}/static/favicon.png</Image>
    <Url type="text/html" method="get" template="{app.state.config.WEBUI_URL}/?q={"{searchTerms}"}"/>
    <moz:SearchForm>{app.state.config.WEBUI_URL}</moz:SearchForm>
    </OpenSearchDescription>
    """
    return Response(content=xml_content, media_type="application/xml")


@app.get("/health")
async def healthcheck():
    return {"status": True}


@app.get("/health/db")
async def healthcheck_with_db():
    Session.execute(text("SELECT 1;")).all()
    return {"status": True}


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/cache", StaticFiles(directory=CACHE_DIR), name="cache")


def swagger_ui_html(*args, **kwargs):
    return get_swagger_ui_html(
        *args,
        **kwargs,
        swagger_js_url="/static/swagger-ui/swagger-ui-bundle.js",
        swagger_css_url="/static/swagger-ui/swagger-ui.css",
        swagger_favicon_url="/static/swagger-ui/favicon.png",
    )


applications.get_swagger_ui_html = swagger_ui_html

if os.path.exists(FRONTEND_BUILD_DIR):
    mimetypes.add_type("text/javascript", ".js")
    app.mount(
        "/",
        SPAStaticFiles(directory=FRONTEND_BUILD_DIR, html=True),
        name="spa-static-files",
    )
else:
    log.warning(
        f"Frontend build directory not found at '{FRONTEND_BUILD_DIR}'. Serving API only."
    )
