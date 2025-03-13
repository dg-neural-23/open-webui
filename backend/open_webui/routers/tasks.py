import json
from fastapi import APIRouter, Depends, HTTPException, Response, status, Request
from fastapi.responses import JSONResponse, RedirectResponse

from pydantic import BaseModel
from typing import Optional
import logging
import re
from open_webui.utils.misc import get_last_user_message
from open_webui.utils.chat import generate_chat_completion
from open_webui.utils.task import (
    title_generation_template,
    query_generation_template,
    image_prompt_generation_template,
    autocomplete_generation_template,
    tags_generation_template,
    emoji_generation_template,
    moa_response_generation_template,
    auto_memory_template
)

from open_webui.routers.memories import Memories
from open_webui.utils.auth import get_admin_user, get_verified_user
from open_webui.constants import TASKS
from open_webui.retrieval.vector.connector import VECTOR_DB_CLIENT
from open_webui.routers.pipelines import process_pipeline_inlet_filter
from open_webui.utils.filter import (
    get_sorted_filter_ids,
    process_filter_functions,
)
from open_webui.utils.task import get_task_model_id
from open_webui.utils.deduplication import compute_cosine_similarity
from open_webui.config import (
    DEFAULT_TITLE_GENERATION_PROMPT_TEMPLATE,
    DEFAULT_TAGS_GENERATION_PROMPT_TEMPLATE,
    DEFAULT_IMAGE_PROMPT_GENERATION_PROMPT_TEMPLATE,
    DEFAULT_QUERY_GENERATION_PROMPT_TEMPLATE,
    DEFAULT_AUTOCOMPLETE_GENERATION_PROMPT_TEMPLATE,
    DEFAULT_EMOJI_GENERATION_PROMPT_TEMPLATE,
    DEFAULT_MOA_GENERATION_PROMPT_TEMPLATE,
    DEFAULT_AUTO_MEMORY_PROMPT_TEMPLATE,
    
)
from open_webui.env import SRC_LOG_LEVELS


log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["MODELS"])

router = APIRouter()


##################################
#
# Task Endpoints
#
##################################


@router.get("/config")
async def get_task_config(request: Request, user=Depends(get_verified_user)):
    return {
        "TASK_MODEL": request.app.state.config.TASK_MODEL,
        "TASK_MODEL_EXTERNAL": request.app.state.config.TASK_MODEL_EXTERNAL,
        "TITLE_GENERATION_PROMPT_TEMPLATE": request.app.state.config.TITLE_GENERATION_PROMPT_TEMPLATE,
        "IMAGE_PROMPT_GENERATION_PROMPT_TEMPLATE": request.app.state.config.IMAGE_PROMPT_GENERATION_PROMPT_TEMPLATE,
        "ENABLE_AUTOCOMPLETE_GENERATION": request.app.state.config.ENABLE_AUTOCOMPLETE_GENERATION,
        "AUTOCOMPLETE_GENERATION_INPUT_MAX_LENGTH": request.app.state.config.AUTOCOMPLETE_GENERATION_INPUT_MAX_LENGTH,
        "TAGS_GENERATION_PROMPT_TEMPLATE": request.app.state.config.TAGS_GENERATION_PROMPT_TEMPLATE,
        "ENABLE_TAGS_GENERATION": request.app.state.config.ENABLE_TAGS_GENERATION,
        "ENABLE_TITLE_GENERATION": request.app.state.config.ENABLE_TITLE_GENERATION,
        "ENABLE_SEARCH_QUERY_GENERATION": request.app.state.config.ENABLE_SEARCH_QUERY_GENERATION,
        "ENABLE_RETRIEVAL_QUERY_GENERATION": request.app.state.config.ENABLE_RETRIEVAL_QUERY_GENERATION,
        "QUERY_GENERATION_PROMPT_TEMPLATE": request.app.state.config.QUERY_GENERATION_PROMPT_TEMPLATE,
        "TOOLS_FUNCTION_CALLING_PROMPT_TEMPLATE": request.app.state.config.TOOLS_FUNCTION_CALLING_PROMPT_TEMPLATE,
        "ENABLE_AUTO_MEMORY": request.app.state.config.ENABLE_AUTO_MEMORY,
        "AUTO_MEMORY_PROMPT_TEMPLATE": request.app.state.config.AUTO_MEMORY_PROMPT_TEMPLATE,
    }


class TaskConfigForm(BaseModel):
    TASK_MODEL: Optional[str]
    TASK_MODEL_EXTERNAL: Optional[str]
    ENABLE_TITLE_GENERATION: bool
    TITLE_GENERATION_PROMPT_TEMPLATE: str
    IMAGE_PROMPT_GENERATION_PROMPT_TEMPLATE: str
    ENABLE_AUTOCOMPLETE_GENERATION: bool
    AUTOCOMPLETE_GENERATION_INPUT_MAX_LENGTH: int
    TAGS_GENERATION_PROMPT_TEMPLATE: str
    ENABLE_TAGS_GENERATION: bool
    ENABLE_SEARCH_QUERY_GENERATION: bool
    ENABLE_RETRIEVAL_QUERY_GENERATION: bool
    QUERY_GENERATION_PROMPT_TEMPLATE: str
    TOOLS_FUNCTION_CALLING_PROMPT_TEMPLATE: str
    ENABLE_AUTO_MEMORY: bool
    AUTO_MEMORY_PROMPT_TEMPLATE: str


@router.post("/config/update")
async def update_task_config(
    request: Request, form_data: TaskConfigForm, user=Depends(get_admin_user)
):
    request.app.state.config.TASK_MODEL = form_data.TASK_MODEL
    request.app.state.config.TASK_MODEL_EXTERNAL = form_data.TASK_MODEL_EXTERNAL
    request.app.state.config.ENABLE_TITLE_GENERATION = form_data.ENABLE_TITLE_GENERATION
    request.app.state.config.TITLE_GENERATION_PROMPT_TEMPLATE = (
        form_data.TITLE_GENERATION_PROMPT_TEMPLATE
    )

    request.app.state.config.IMAGE_PROMPT_GENERATION_PROMPT_TEMPLATE = (
        form_data.IMAGE_PROMPT_GENERATION_PROMPT_TEMPLATE
    )

    request.app.state.config.ENABLE_AUTOCOMPLETE_GENERATION = (
        form_data.ENABLE_AUTOCOMPLETE_GENERATION
    )
    request.app.state.config.AUTOCOMPLETE_GENERATION_INPUT_MAX_LENGTH = (
        form_data.AUTOCOMPLETE_GENERATION_INPUT_MAX_LENGTH
    )

    request.app.state.config.TAGS_GENERATION_PROMPT_TEMPLATE = (
        form_data.TAGS_GENERATION_PROMPT_TEMPLATE
    )
    request.app.state.config.ENABLE_TAGS_GENERATION = form_data.ENABLE_TAGS_GENERATION
    request.app.state.config.ENABLE_SEARCH_QUERY_GENERATION = (
        form_data.ENABLE_SEARCH_QUERY_GENERATION
    )
    request.app.state.config.ENABLE_RETRIEVAL_QUERY_GENERATION = (
        form_data.ENABLE_RETRIEVAL_QUERY_GENERATION
    )

    request.app.state.config.QUERY_GENERATION_PROMPT_TEMPLATE = (
        form_data.QUERY_GENERATION_PROMPT_TEMPLATE
    )
    request.app.state.config.TOOLS_FUNCTION_CALLING_PROMPT_TEMPLATE = (
        form_data.TOOLS_FUNCTION_CALLING_PROMPT_TEMPLATE
    )
    request.app.state.config.ENABLE_AUTO_MEMORY = (
        form_data.ENABLE_AUTO_MEMORY
    )

    return {
        "TASK_MODEL": request.app.state.config.TASK_MODEL,
        "TASK_MODEL_EXTERNAL": request.app.state.config.TASK_MODEL_EXTERNAL,
        "ENABLE_TITLE_GENERATION": request.app.state.config.ENABLE_TITLE_GENERATION,
        "TITLE_GENERATION_PROMPT_TEMPLATE": request.app.state.config.TITLE_GENERATION_PROMPT_TEMPLATE,
        "IMAGE_PROMPT_GENERATION_PROMPT_TEMPLATE": request.app.state.config.IMAGE_PROMPT_GENERATION_PROMPT_TEMPLATE,
        "ENABLE_AUTOCOMPLETE_GENERATION": request.app.state.config.ENABLE_AUTOCOMPLETE_GENERATION,
        "AUTOCOMPLETE_GENERATION_INPUT_MAX_LENGTH": request.app.state.config.AUTOCOMPLETE_GENERATION_INPUT_MAX_LENGTH,
        "TAGS_GENERATION_PROMPT_TEMPLATE": request.app.state.config.TAGS_GENERATION_PROMPT_TEMPLATE,
        "ENABLE_TAGS_GENERATION": request.app.state.config.ENABLE_TAGS_GENERATION,
        "ENABLE_SEARCH_QUERY_GENERATION": request.app.state.config.ENABLE_SEARCH_QUERY_GENERATION,
        "ENABLE_RETRIEVAL_QUERY_GENERATION": request.app.state.config.ENABLE_RETRIEVAL_QUERY_GENERATION,
        "QUERY_GENERATION_PROMPT_TEMPLATE": request.app.state.config.QUERY_GENERATION_PROMPT_TEMPLATE,
        "TOOLS_FUNCTION_CALLING_PROMPT_TEMPLATE": request.app.state.config.TOOLS_FUNCTION_CALLING_PROMPT_TEMPLATE,
        "ENABLE_AUTO_MEMORY": request.app.state.config.ENABLE_AUTO_MEMORY 
    }


@router.post("/title/completions")
async def generate_title(
    request: Request, form_data: dict, user=Depends(get_verified_user)
):

    if not request.app.state.config.ENABLE_TITLE_GENERATION:
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"detail": "Title generation is disabled"},
        )

    if getattr(request.state, "direct", False) and hasattr(request.state, "model"):
        models = {
            request.state.model["id"]: request.state.model,
        }
    else:
        models = request.app.state.MODELS

    model_id = form_data["model"]
    if model_id not in models:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model not found",
        )

    # Check if the user has a custom task model
    # If the user has a custom task model, use that model
    task_model_id = get_task_model_id(
        model_id,
        request.app.state.config.TASK_MODEL,
        request.app.state.config.TASK_MODEL_EXTERNAL,
        models,
    )

    log.debug(
        f"generating chat title using model {task_model_id} for user {user.email} "
    )

    if request.app.state.config.TITLE_GENERATION_PROMPT_TEMPLATE != "":
        template = request.app.state.config.TITLE_GENERATION_PROMPT_TEMPLATE
    else:
        template = DEFAULT_TITLE_GENERATION_PROMPT_TEMPLATE

    messages = form_data["messages"]

    # Remove reasoning details from the messages
    for message in messages:
        message["content"] = re.sub(
            r"<details\s+type=\"reasoning\"[^>]*>.*?<\/details>",
            "",
            message["content"],
            flags=re.S,
        ).strip()

    content = title_generation_template(
        template,
        messages,
        {
            "name": user.name,
            "location": user.info.get("location") if user.info else None,
        },
    )

    payload = {
        "model": task_model_id,
        "messages": [{"role": "user", "content": content}],
        "stream": False,
        **(
            {"max_tokens": 1000}
            if models[task_model_id].get("owned_by") == "ollama"
            else {
                "max_completion_tokens": 1000,
            }
        ),
        "metadata": {
            **(request.state.metadata if hasattr(request.state, "metadata") else {}),
            "task": str(TASKS.TITLE_GENERATION),
            "task_body": form_data,
            "chat_id": form_data.get("chat_id", None),
        },
    }

    # Process the payload through the pipeline
    try:
        payload = await process_pipeline_inlet_filter(request, payload, user, models)
    except Exception as e:
        raise e

    try:
        return await generate_chat_completion(request, form_data=payload, user=user)
    except Exception as e:
        log.error("Exception occurred", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"detail": "An internal error has occurred."},
        )


@router.post("/tags/completions")
async def generate_chat_tags(
    request: Request, form_data: dict, user=Depends(get_verified_user)
):

    if not request.app.state.config.ENABLE_TAGS_GENERATION:
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"detail": "Tags generation is disabled"},
        )

    if getattr(request.state, "direct", False) and hasattr(request.state, "model"):
        models = {
            request.state.model["id"]: request.state.model,
        }
    else:
        models = request.app.state.MODELS

    model_id = form_data["model"]
    if model_id not in models:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model not found",
        )

    # Check if the user has a custom task model
    # If the user has a custom task model, use that model
    task_model_id = get_task_model_id(
        model_id,
        request.app.state.config.TASK_MODEL,
        request.app.state.config.TASK_MODEL_EXTERNAL,
        models,
    )

    log.debug(
        f"generating chat tags using model {task_model_id} for user {user.email} "
    )

    if request.app.state.config.TAGS_GENERATION_PROMPT_TEMPLATE != "":
        template = request.app.state.config.TAGS_GENERATION_PROMPT_TEMPLATE
    else:
        template = DEFAULT_TAGS_GENERATION_PROMPT_TEMPLATE

    content = tags_generation_template(
        template, form_data["messages"], {"name": user.name}
    )

    payload = {
        "model": task_model_id,
        "messages": [{"role": "user", "content": content}],
        "stream": False,
        "metadata": {
            **(request.state.metadata if hasattr(request.state, "metadata") else {}),
            "task": str(TASKS.TAGS_GENERATION),
            "task_body": form_data,
            "chat_id": form_data.get("chat_id", None),
        },
    }

    # Process the payload through the pipeline
    try:
        payload = await process_pipeline_inlet_filter(request, payload, user, models)
    except Exception as e:
        raise e

    try:
        return await generate_chat_completion(request, form_data=payload, user=user)
    except Exception as e:
        log.error(f"Error generating chat completion: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "An internal error has occurred."},
        )


@router.post("/image_prompt/completions")
async def generate_image_prompt(
    request: Request, form_data: dict, user=Depends(get_verified_user)
):
    if getattr(request.state, "direct", False) and hasattr(request.state, "model"):
        models = {
            request.state.model["id"]: request.state.model,
        }
    else:
        models = request.app.state.MODELS

    model_id = form_data["model"]
    if model_id not in models:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model not found",
        )

    # Check if the user has a custom task model
    # If the user has a custom task model, use that model
    task_model_id = get_task_model_id(
        model_id,
        request.app.state.config.TASK_MODEL,
        request.app.state.config.TASK_MODEL_EXTERNAL,
        models,
    )

    log.debug(
        f"generating image prompt using model {task_model_id} for user {user.email} "
    )

    if request.app.state.config.IMAGE_PROMPT_GENERATION_PROMPT_TEMPLATE != "":
        template = request.app.state.config.IMAGE_PROMPT_GENERATION_PROMPT_TEMPLATE
    else:
        template = DEFAULT_IMAGE_PROMPT_GENERATION_PROMPT_TEMPLATE

    content = image_prompt_generation_template(
        template,
        form_data["messages"],
        user={
            "name": user.name,
        },
    )

    payload = {
        "model": task_model_id,
        "messages": [{"role": "user", "content": content}],
        "stream": False,
        "metadata": {
            **(request.state.metadata if hasattr(request.state, "metadata") else {}),
            "task": str(TASKS.IMAGE_PROMPT_GENERATION),
            "task_body": form_data,
            "chat_id": form_data.get("chat_id", None),
        },
    }

    # Process the payload through the pipeline
    try:
        payload = await process_pipeline_inlet_filter(request, payload, user, models)
    except Exception as e:
        raise e

    try:
        return await generate_chat_completion(request, form_data=payload, user=user)
    except Exception as e:
        log.error("Exception occurred", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"detail": "An internal error has occurred."},
        )


@router.post("/queries/completions")
async def generate_queries(
    request: Request, form_data: dict, user=Depends(get_verified_user)
):

    type = form_data.get("type")
    if type == "web_search":
        if not request.app.state.config.ENABLE_SEARCH_QUERY_GENERATION:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Search query generation is disabled",
            )
    elif type == "retrieval":
        if not request.app.state.config.ENABLE_RETRIEVAL_QUERY_GENERATION:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Query generation is disabled",
            )

    if getattr(request.state, "direct", False) and hasattr(request.state, "model"):
        models = {
            request.state.model["id"]: request.state.model,
        }
    else:
        models = request.app.state.MODELS

    model_id = form_data["model"]
    if model_id not in models:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model not found",
        )

    # Check if the user has a custom task model
    # If the user has a custom task model, use that model
    task_model_id = get_task_model_id(
        model_id,
        request.app.state.config.TASK_MODEL,
        request.app.state.config.TASK_MODEL_EXTERNAL,
        models,
    )

    log.debug(
        f"generating {type} queries using model {task_model_id} for user {user.email}"
    )

    if (request.app.state.config.QUERY_GENERATION_PROMPT_TEMPLATE).strip() != "":
        template = request.app.state.config.QUERY_GENERATION_PROMPT_TEMPLATE
    else:
        template = DEFAULT_QUERY_GENERATION_PROMPT_TEMPLATE

    content = query_generation_template(
        template, form_data["messages"], {"name": user.name}
    )

    payload = {
        "model": task_model_id,
        "messages": [{"role": "user", "content": content}],
        "stream": False,
        "metadata": {
            **(request.state.metadata if hasattr(request.state, "metadata") else {}),
            "task": str(TASKS.QUERY_GENERATION),
            "task_body": form_data,
            "chat_id": form_data.get("chat_id", None),
        },
    }

    # Process the payload through the pipeline
    try:
        payload = await process_pipeline_inlet_filter(request, payload, user, models)
    except Exception as e:
        raise e

    try:
        return await generate_chat_completion(request, form_data=payload, user=user)
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"detail": str(e)},
        )


@router.post("/auto/completions")
async def generate_autocompletion(
    request: Request, form_data: dict, user=Depends(get_verified_user)
):
    if not request.app.state.config.ENABLE_AUTOCOMPLETE_GENERATION:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Autocompletion generation is disabled",
        )

    type = form_data.get("type")
    prompt = form_data.get("prompt")
    messages = form_data.get("messages")

    if request.app.state.config.AUTOCOMPLETE_GENERATION_INPUT_MAX_LENGTH > 0:
        if (
            len(prompt)
            > request.app.state.config.AUTOCOMPLETE_GENERATION_INPUT_MAX_LENGTH
        ):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Input prompt exceeds maximum length of {request.app.state.config.AUTOCOMPLETE_GENERATION_INPUT_MAX_LENGTH}",
            )

    if getattr(request.state, "direct", False) and hasattr(request.state, "model"):
        models = {
            request.state.model["id"]: request.state.model,
        }
    else:
        models = request.app.state.MODELS

    model_id = form_data["model"]
    if model_id not in models:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model not found",
        )

    # Check if the user has a custom task model
    # If the user has a custom task model, use that model
    task_model_id = get_task_model_id(
        model_id,
        request.app.state.config.TASK_MODEL,
        request.app.state.config.TASK_MODEL_EXTERNAL,
        models,
    )

    log.debug(
        f"generating autocompletion using model {task_model_id} for user {user.email}"
    )

    if (request.app.state.config.AUTOCOMPLETE_GENERATION_PROMPT_TEMPLATE).strip() != "":
        template = request.app.state.config.AUTOCOMPLETE_GENERATION_PROMPT_TEMPLATE
    else:
        template = DEFAULT_AUTOCOMPLETE_GENERATION_PROMPT_TEMPLATE

    content = autocomplete_generation_template(
        template, prompt, messages, type, {"name": user.name}
    )

    payload = {
        "model": task_model_id,
        "messages": [{"role": "user", "content": content}],
        "stream": False,
        "metadata": {
            **(request.state.metadata if hasattr(request.state, "metadata") else {}),
            "task": str(TASKS.AUTOCOMPLETE_GENERATION),
            "task_body": form_data,
            "chat_id": form_data.get("chat_id", None),
        },
    }

    # Process the payload through the pipeline
    try:
        payload = await process_pipeline_inlet_filter(request, payload, user, models)
    except Exception as e:
        raise e

    try:
        return await generate_chat_completion(request, form_data=payload, user=user)
    except Exception as e:
        log.error(f"Error generating chat completion: {e}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "An internal error has occurred."},
        )


@router.post("/emoji/completions")
async def generate_emoji(
    request: Request, form_data: dict, user=Depends(get_verified_user)
):

    if getattr(request.state, "direct", False) and hasattr(request.state, "model"):
        models = {
            request.state.model["id"]: request.state.model,
        }
    else:
        models = request.app.state.MODELS

    model_id = form_data["model"]
    if model_id not in models:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model not found",
        )

    # Check if the user has a custom task model
    # If the user has a custom task model, use that model
    task_model_id = get_task_model_id(
        model_id,
        request.app.state.config.TASK_MODEL,
        request.app.state.config.TASK_MODEL_EXTERNAL,
        models,
    )

    log.debug(f"generating emoji using model {task_model_id} for user {user.email} ")

    template = DEFAULT_EMOJI_GENERATION_PROMPT_TEMPLATE

    content = emoji_generation_template(
        template,
        form_data["prompt"],
        {
            "name": user.name,
            "location": user.info.get("location") if user.info else None,
        },
    )

    payload = {
        "model": task_model_id,
        "messages": [{"role": "user", "content": content}],
        "stream": False,
        **(
            {"max_tokens": 4}
            if models[task_model_id].get("owned_by") == "ollama"
            else {
                "max_completion_tokens": 4,
            }
        ),
        "chat_id": form_data.get("chat_id", None),
        "metadata": {
            **(request.state.metadata if hasattr(request.state, "metadata") else {}),
            "task": str(TASKS.EMOJI_GENERATION),
            "task_body": form_data,
        },
    }

    # Process the payload through the pipeline
    try:
        payload = await process_pipeline_inlet_filter(request, payload, user, models)
    except Exception as e:
        raise e

    try:
        return await generate_chat_completion(request, form_data=payload, user=user)
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"detail": str(e)},
        )


@router.post("/moa/completions")
async def generate_moa_response(
    request: Request, form_data: dict, user=Depends(get_verified_user)
):

    if getattr(request.state, "direct", False) and hasattr(request.state, "model"):
        models = {
            request.state.model["id"]: request.state.model,
        }
    else:
        models = request.app.state.MODELS

    model_id = form_data["model"]

    if model_id not in models:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model not found",
        )

    # Check if the user has a custom task model
    # If the user has a custom task model, use that model
    task_model_id = get_task_model_id(
        model_id,
        request.app.state.config.TASK_MODEL,
        request.app.state.config.TASK_MODEL_EXTERNAL,
        models,
    )

    log.debug(f"generating MOA model {task_model_id} for user {user.email} ")

    template = DEFAULT_MOA_GENERATION_PROMPT_TEMPLATE

    content = moa_response_generation_template(
        template,
        form_data["prompt"],
        form_data["responses"],
    )

    payload = {
        "model": task_model_id,
        "messages": [{"role": "user", "content": content}],
        "stream": form_data.get("stream", False),
        "metadata": {
            **(request.state.metadata if hasattr(request.state, "metadata") else {}),
            "chat_id": form_data.get("chat_id", None),
            "task": str(TASKS.MOA_RESPONSE_GENERATION),
            "task_body": form_data,
        },
    }

    # Process the payload through the pipeline
    try:
        payload = await process_pipeline_inlet_filter(request, payload, user, models)
    except Exception as e:
        raise e

    try:
        return await generate_chat_completion(request, form_data=payload, user=user)
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"detail": str(e)},
        )


@router.post("/auto_memory/completions")
async def generate_auto_memory(
    request: Request, form_data: dict, user=Depends(get_verified_user)
):
    log.error("TEST: Auto Memory endpoint called") 
    log.info(f"Auto Memory endpoint called with data: {form_data}")

    if not request.app.state.config.ENABLE_AUTO_MEMORY:
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"detail": "Auto memory is disabled"},
        )
  
    if getattr(request.state, "direct", False) and hasattr(request.state, "model"):
        models = {
            request.state.model["id"]: request.state.model,
        }
    else:
        models = request.app.state.MODELS

    if "model" not in form_data or not form_data["model"]:
         # Pobierz pierwszy dostępny model lub domyślny model zadań
         model_id = request.app.state.config.DEFAULT_MODEL
         if not model_id and models:
             model_id = next(iter(models.keys()))
         if not model_id:
             return JSONResponse(
                 status_code=status.HTTP_400_BAD_REQUEST,
                 content={"detail": "No model specified and no default model available"},
             )
         log.info(f"Using default model {model_id} for auto memory generation")
    else:
         model_id = form_data["model"]

     # Sprawdź czy model istnieje
    if model_id not in models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    memory_tool = {
        "type": "function",
        "function": {
            "name": "extract_memories",
            "description": "Extract important facts about the user from conversation",
            "parameters": {
                "type": "object",
                "properties": {
                    "memories": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "description": "a clear factual statement about the user"
                        },
                        "description": " List of extracted memories about the user"
                    }
                },
                "required": ["memories"]
            }
        }
    }

    template = request.app.state.config.AUTO_MEMORY_PROMPT_TEMPLATE
    if not template:
        template = DEFAULT_AUTO_MEMORY_PROMPT_TEMPLATE  # Z podkreślnikiem

    content = auto_memory_template(
        template,
        form_data["messages"],
        {
            "name": user.name,
            "location": user.info.get("location") if user.info else None,
        },
    )

    model_supports_functions = request.app.state.MODELS[model_id].get("supports_tools", False)
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": content}],
        "stream": False,
        "temperature": 0.4,
        "tools": [memory_tool] if model_supports_functions else None,
        "tool_choice": {"type": "function", "function": {"name": "extract_memories"}}
            if model_supports_functions else None,
        "metadata": {
            **(request.state.metadata if hasattr(request.state, "metadata") else {}),
            "task": str(TASKS.AUTO_MEMORY),
            "chat_id": form_data.get("chat_id", None),
        }
    }
    try:
        payload = await process_pipeline_inlet_filter(request, payload, user, models)
    except Exception as e:
        raise e
    try:
        response = await generate_chat_completion(request, form_data=payload, user=user)
            
        memory_items = []
        added_items = []
        duplicates = []
        if response and "choices" in response and len(response["choices"]):
            message = response["choices"][0]["message"]
            if "tool_calls" in message and message["tool_calls"]:
                for tool_call in message["tool_calls"]:
                    if tool_call["function"]["name"] == "extract_memories":
                        try:
                            args = json.loads(tool_call["function"]["arguments"])
                            if "memories" in args:
                                memory_items = args["memories"]
                                log.info(f"Extracted {len(memory_items)} memories from function call")
                        except Exception as e:
                            log.error(f"Error parsing function arguments: {e}", exc_info=True)
            elif "content" in message and message["content"]:
                content = message["content"]
                log.debug(f"Processing text response: {content[:100]}")
                
                try:
                    json_match = re.search(r'```(?:json)?\s*({.*?}|[\[{].*?[\]}])\s*```', content, re.DOTALL)
                    if json_match:
                        data = json.loads(json_match.group(1))
                        if isinstance(data, dict) and "memories" in data:
                            memory_items = data["memories"]
                            log.info(f"extracted memories from json in code block")
                    else:
                        if (content.strip().startswith('[') and content.strip().endswith(']')) or \
                            (content.strip().startswith('{') and content.strip().endswith('}')):
                            data = json.loads(content.strip())
                            if isinstance(data, dict) and "memories" in data:
                                memory_items = data["memories"]
                            elif isinstance(data, list):
                                memory_items = data
                            log.info(f"extracted memories from direct JSON")
                        else:
                            # Szukaj listy numerowanej lub z punktorami
                            lines = [line.strip() for line in content.split('\n') if line.strip()]
                            # Usuń numerację lub punktory na początku
                            memory_items = [re.sub(r'^(\d+[\.\)]-?|\*\s*|-\s*)', '', line).strip() 
                                        for line in lines if len(line) > 5]
                            log.info(f"Extracted {len(memory_items)} memories from text lines")
                except Exception as e:
                    log.error(f"Error parsing response: {e}", exc_info=True)
                    memory_items = [line.strip() for line in content.split("\n")
                                   if len(line.strip()) > 5 and not line.strip().startswith('```')]
                                    
            memory_items = [item.strip('"\'- ').rstrip('.') for item in memory_items                                
                          if item and isinstance(item, str) and len(item.strip()) > 5]

            existing_memories = Memories.get_memories_by_user_id(user.id)
            for item in memory_items:
                if not item or len(item) < 5:
                    continue
                try:
                    is_duplicate = False
                    new_memory_embedding = None
                    try:
                        if hasattr(request.app.state, "EMBEDDING_FUNCTION"):
                            new_memory_embedding = request.app.state.EMBEDDING_FUNCTION(item)
                    except Exception as e:
                        log.warning(f"Failed to generate embedding for memory: {e}")
                    if new_memory_embedding:
                        similar_results = VECTOR_DB_CLIENT.search(
                            collection_name=f"user-memory-{user.id}",
                            query_vector=new_memory_embedding,
                            limit=5,
                            include_metadata=True
                        )
                        for result in similar_results:
                            if result.score > 0.9:
                                is_duplicate = True
                                duplicates.append({
                                    "new": item,
                                    "existing": result.payload["text"],
                                    "similarity": result.score
                                })
                                break
                except Exception as e:
                    log.warning(f"Vector search failed, falling back to manual comparison: {e}")
                    for existing_memory in existing_memories:
                        if hasattr(existing_memory, "data") and existing_memory.data and \
                           "embedding" in existing_memory.data:
                            similarity = compute_cosine_similarity(new_memory_embedding, existing_memory.data["embedding"],
                                                                handle_different_lengths="interpolate")
                            if similarity > 0.9:
                                is_duplicate = True
                                duplicates.append({
                                    "new": item,
                                    "existing": existing_memory.content,
                                    "similarity": similarity
                                }) 
                                break
                if not is_duplicate:
                    memory = Memories.insert_new_memory(user.id, item)
                    if memory and new_memory_embedding:
                        try:
                            VECTOR_DB_CLIENT.upsert(
                                collection_name=f"user-memory-{user.id}",
                                items=[
                                    {
                                        "id": memory.id,
                                        "text": memory.content,
                                        "vector": new_memory_embedding,
                                        "metadata": {"created_at": memory.created_at}
                                    }
                                ]
                            )
                        except Exception as e:
                            log.error(f"Error adding to vector DB: {e}")
                    if memory:
                        added_items.append(memory.model_dump())

            return {
                "success": True,
                "added": added_items,
                "duplicates": duplicates,
                "total_added": len(added_items),
                "model_used": model_id,
                "function_called": "tool_calls" in message and len(message["tool_calls"]) > 0
            }
            
     # Przypadek gdy nie znaleziono wspomnień
        return {
            "success": False,
            "added": [],
            "duplicates": [],
            "total_added": 0,
            "model_used": model_id
        }

    except Exception as e:
        log.error(f"Exception occurred: {e}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"detail": f"Error: {str(e)}"},
        )