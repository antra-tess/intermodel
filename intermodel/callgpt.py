#!/usr/bin/env python
import dataclasses
import json
import re
import sys
import warnings
import traceback
import asyncio
import cmd
import datetime
import os
import uuid
import hashlib
from typing import Union, List, Optional, Iterable
import base64
from io import BytesIO
from PIL import Image
import datetime
import requests
from mimetypes import guess_type
import time
from typing import Dict, Tuple

import aiohttp
import tenacity
from huggingface_hub.utils import LocalEntryNotFoundError

import intermodel.callgpt_faker

from dotenv import load_dotenv

from intermodel.hf_token import get_hf_tokenizer, get_hf_auth_token

load_dotenv()

# URL validation cache: stores (is_valid, last_checked_timestamp) for each URL
_url_validation_cache: Dict[str, Tuple[bool, float]] = {}
_URL_CACHE_DURATION = 3600  # 1 hour in seconds

async def validate_image_url(url: str, session: Optional[aiohttp.ClientSession] = None) -> bool:
    """Validate that an image URL is accessible and returns a valid image.
    
    Args:
        url: The URL to validate
        session: Optional aiohttp session to reuse
        
    Returns:
        bool: True if the URL is valid and accessible, False otherwise
    """
    # Check cache first
    current_time = time.time()
    if url in _url_validation_cache:
        is_valid, last_checked = _url_validation_cache[url]
        if current_time - last_checked < _URL_CACHE_DURATION:
            print(f"[DEBUG] URL validation cache hit for {url[:100]}{'...' if len(url) > 100 else ''}: {'valid' if is_valid else 'invalid'}", file=sys.stderr)
            return is_valid
    
    # Validate the URL
    print(f"[DEBUG] Validating image URL: {url[:100]}{'...' if len(url) > 100 else ''}", file=sys.stderr)
    
    try:
        # Create session if not provided
        if session is None:
            async with aiohttp.ClientSession() as new_session:
                return await _validate_url_with_session(url, new_session, current_time)
        else:
            return await _validate_url_with_session(url, session, current_time)
    except Exception as e:
        print(f"[DEBUG] Error validating URL {url[:100]}{'...' if len(url) > 100 else ''}: {str(e)}", file=sys.stderr)
        _url_validation_cache[url] = (False, current_time)
        return False

async def _validate_url_with_session(url: str, session: aiohttp.ClientSession, current_time: float) -> bool:
    """Internal function to validate URL with a given session."""
    try:
        # Use HEAD request first to check if resource exists without downloading
        async with session.head(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
            if response.status != 200:
                # Try GET if HEAD fails (some servers don't support HEAD)
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as get_response:
                    if get_response.status != 200:
                        print(f"[DEBUG] URL returned status {get_response.status}", file=sys.stderr)
                        _url_validation_cache[url] = (False, current_time)
                        return False
                    
                    # Check content type
                    content_type = get_response.headers.get('Content-Type', '').lower()
                    if not any(img_type in content_type for img_type in ['image/jpeg', 'image/png', 'image/gif', 'image/webp', 'image/']):
                        print(f"[DEBUG] URL content type '{content_type}' is not an image", file=sys.stderr)
                        _url_validation_cache[url] = (False, current_time)
                        return False
                    
                    # Check content length if available
                    content_length = get_response.headers.get('Content-Length')
                    if content_length:
                        size_mb = int(content_length) / (1024 * 1024)
                        if size_mb > 5:
                            print(f"[DEBUG] Image size {size_mb:.2f}MB exceeds 5MB limit", file=sys.stderr)
                            _url_validation_cache[url] = (False, current_time)
                            return False
            else:
                # HEAD request succeeded, check headers
                content_type = response.headers.get('Content-Type', '').lower()
                if not any(img_type in content_type for img_type in ['image/jpeg', 'image/png', 'image/gif', 'image/webp', 'image/']):
                    print(f"[DEBUG] URL content type '{content_type}' is not an image", file=sys.stderr)
                    _url_validation_cache[url] = (False, current_time)
                    return False
                
                content_length = response.headers.get('Content-Length')
                if content_length:
                    size_mb = int(content_length) / (1024 * 1024)
                    if size_mb > 5:
                        print(f"[DEBUG] Image size {size_mb:.2f}MB exceeds 5MB limit", file=sys.stderr)
                        _url_validation_cache[url] = (False, current_time)
                        return False
        
        print(f"[DEBUG] URL validation successful", file=sys.stderr)
        _url_validation_cache[url] = (True, current_time)
        return True
        
    except asyncio.TimeoutError:
        print(f"[DEBUG] URL validation timed out", file=sys.stderr)
        _url_validation_cache[url] = (False, current_time)
        return False
    except Exception as e:
        print(f"[DEBUG] URL validation error: {str(e)}", file=sys.stderr)
        _url_validation_cache[url] = (False, current_time)
        return False

def guess_mime_type(url: str) -> Optional[str]:
    """Get the MIME type from a URL or file path.
    
    Args:
        url: URL or file path to guess the MIME type for
        
    Returns:
        str or None: The MIME type if it can be determined, or None if not
    """
    mime_type, _ = guess_type(url)
    if not mime_type and url.lower().endswith(('.jpg', '.jpeg')):
        return 'image/jpeg'
    elif not mime_type and url.lower().endswith('.png'):
        return 'image/png'
    elif not mime_type and url.lower().endswith('.gif'):
        return 'image/gif'
    elif not mime_type and url.lower().endswith('.webp'):
        return 'image/webp'
    return mime_type

MODEL_ALIASES = {}
untokenizable = set()

session: Optional[aiohttp.ClientSession] = None

@dataclasses.dataclass
class MessagePart:
    type: str  # "text" or "image"
    content: str  # text content or image URL
    mime_type: Optional[str] = None  # for images
    cache_control: Optional[dict] = None  # for Anthropic prompt caching

@dataclasses.dataclass
class ProcessedMessage:
    role: str
    parts: List[MessagePart]

def convert_to_openai_format(processed_msg: ProcessedMessage) -> dict:
    content = []
    for part in processed_msg.parts:
        if part.type == "text":
            content.append({"type": "text", "text": part.content})
        elif part.type == "image":
            content.append({
                "type": "image_url",
                "image_url": {"url": part.content}
            })
    return {"role": processed_msg.role, "content": content}

def convert_to_gemini_format(processed_msg: ProcessedMessage) -> "types.Content":
    from google.genai import types
    parts = []
    for part in processed_msg.parts:
        if part.type == "text":
            parts.append(types.Part(text=part.content))
        elif part.type == "image":
            # Handle image conversion for Gemini
            parts.append(types.Part(
                inline_data=types.Blob(
                    mime_type=part.mime_type,
                    data=download_and_process_image(part.content)
                )
            ))
    # Map roles to Gemini's expected format ('user' or 'model')
    gemini_role = processed_msg.role
    if processed_msg.role == 'assistant' or processed_msg.role == 'system':
        gemini_role = 'model'
    elif processed_msg.role != 'user': # Treat any other unknown roles as user
        gemini_role = 'user'
    
    return types.Content(parts=parts, role=gemini_role)

async def convert_to_bedrock_format(processed_msg: ProcessedMessage, session: Optional[aiohttp.ClientSession] = None) -> dict:
    """Convert the intermediate message format to Bedrock's API format.
    
    Similar to Anthropic format but requires all images to be base64-encoded.
    Bedrock doesn't support image URLs - only base64 data.
    
    Args:
        processed_msg: A ProcessedMessage object containing text and/or images
        session: Optional aiohttp session for downloading images
        
    Returns:
        dict: Message formatted for Bedrock's API
    """
    content = []
    
    for part in processed_msg.parts:
        if part.type == "text":
            content.append({
                "type": "text",
                "text": part.content
            })
        elif part.type == "image":
            # For Bedrock, all images must be base64-encoded
            if part.content.startswith(('http://', 'https://')):
                # Download and convert URL to base64
                try:
                    if session is None:
                        import aiohttp
                        async with aiohttp.ClientSession() as new_session:
                            base64_data, mime_type = await download_and_encode_image_for_bedrock(part.content, new_session)
                    else:
                        base64_data, mime_type = await download_and_encode_image_for_bedrock(part.content, session)
                    
                    if base64_data is None:
                        print(f"[WARNING] Skipping invalid/inaccessible image URL for Bedrock: {part.content[:100]}{'...' if len(part.content) > 100 else ''}", file=sys.stderr)
                        content.append({
                            "type": "text",
                            "text": f"[Image URL was not accessible and has been skipped]"
                        })
                        continue
                    
                    content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": mime_type or "image/jpeg",
                            "data": base64_data
                        }
                    })
                except Exception as e:
                    print(f"[ERROR] Failed to download image for Bedrock: {str(e)}", file=sys.stderr)
                    content.append({
                        "type": "text",
                        "text": f"[Error processing image URL: {part.content}]"
                    })
            else:
                # Already base64 data, use as-is
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": part.mime_type or "image/jpeg",
                        "data": part.content
                    }
                })
    
    return {
        "role": "user" if processed_msg.role in ["user", "system"] else "assistant",
        "content": content
    }


async def convert_to_anthropic_format(processed_msg: ProcessedMessage, session: Optional[aiohttp.ClientSession] = None) -> dict:
    """Convert the intermediate message format to Anthropic's API format.
    
    Args:
        processed_msg: A ProcessedMessage object containing text and/or images
        session: Optional aiohttp session for URL validation
        
    Returns:
        dict: Message formatted for Anthropic's API
    
    Notes:
        - Anthropic supports both URL and base64-encoded images
        - Images must be JPEG, PNG, GIF, or WebP format
        - Maximum 100 images per API request
        - Maximum 5MB per image
        - If image is larger than 8000x8000px (or 2000x2000px for >20 images), it will be rejected
        - URLs are validated before being sent to ensure they're accessible
        - Supports cache_control for prompt caching (beta feature)
    """
    content = []
    
    for part in processed_msg.parts:
        if part.type == "text":
            text_block = {
                "type": "text",
                "text": part.content
            }
            # Add cache_control if present
            if part.cache_control:
                text_block["cache_control"] = part.cache_control
            content.append(text_block)
        elif part.type == "image":
            # For URLs, validate and use the URL source type
            if part.content.startswith(('http://', 'https://')):
                # Validate the URL is accessible
                is_valid = await validate_image_url(part.content, session)
                if not is_valid:
                    print(f"[WARNING] Skipping invalid/inaccessible image URL: {part.content[:100]}{'...' if len(part.content) > 100 else ''}", file=sys.stderr)
                    # Add a text message explaining the skipped image
                    content.append({
                        "type": "text",
                        "text": f"[Image URL was not accessible and has been skipped]"
                    })
                    continue
                
                content.append({
                    "type": "image",
                    "source": {
                        "type": "url",
                        "url": part.content
                    }
                })
            else:
                # For base64 data, include the mime type
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": part.mime_type or "image/jpeg",  # Default to JPEG if not specified
                        "data": part.content
                    }
                })
    
    return {
        "role": "user" if processed_msg.role in ["user", "system"] else "assistant",
        "content": content
    }


@tenacity.retry(
    retry=tenacity.retry_if_exception(
        lambda e: isinstance(e, aiohttp.ClientResponseError)
        and e.status in (429,)
        or isinstance(e, ValueError)
    ),
    wait=tenacity.wait_random_exponential(min=1, max=60),
    stop=tenacity.stop_after_attempt(6),
)
async def complete(
    model,
    prompt=None,
    temperature=None,
    top_p=None,
    max_tokens=None,
    stop: Optional[List[str]] = None,
    frequency_penalty: Union[float, int] = 0,
    presence_penalty: Union[float, int] = 0,
    num_completions: int = None,
    top_k=None,
    repetition_penalty: Union[float, int] = 1,
    tfs=1,
    user_id=None,
    logit_bias=None,
    vendor=None,
    vendor_config=None,
    voice: Optional[str] = None,
    force_api_mode=None,
    log_dir="intermodel_logs",  # Add log_dir parameter with a default
    cache_breakpoints: bool = False,  # Enable cache breakpoint processing for Anthropic
    cache_type: str = "ephemeral",  # Type of cache control: "ephemeral", "5m", "1h"
    min_p: Optional[float] = None,  # Minimum probability for sampling (used by K2/K3 models)
    add_bos_token: bool = True,  # Whether to add [BOS] token for K2/K3 models (sglang doesn't add by default)
    **kwargs,
):
    modalities = kwargs.pop("modalities", None)
    audio_settings = kwargs.pop("audio_settings", None)
    message_history_format = kwargs.get("message_history_format", None)
    messages = kwargs.get("messages", None)
    name = kwargs.get("name", None)
    max_images = kwargs.get("max_images", 10)  # Default to 10 images
    reasoning_effort = kwargs.get("reasoning_effort", None)  # Extract reasoning_effort before kwargs is reassigned
    kwargs = kwargs.get("continuation_options", {})
    tokenize_as = parse_model_string(MODEL_ALIASES.get(model, model)).tokenize_as
    model = parse_model_string(MODEL_ALIASES.get(model, model)).model
    # todo: multiple completions, top k, logit bias for all vendors
    # todo: detect model not found on all vendors and throw the same exception
    if vendor is None:
        vendor = pick_vendor(model, vendor_config)
    if vendor_config is not None and vendor in vendor_config:
        kwargs = {**vendor_config[vendor]["config"], **kwargs}
    if vendor.startswith("openai"):
        import sys
        import os
        import re

        global session
        if session is None:
            session = aiohttp.ClientSession()
        if user_id is None:
            hashed_user_id = None
        else:
            hash_object = hashlib.sha256()
            hash_object.update(os.getenv("INTERMODEL_HASH_SALT", "").encode("utf-8"))
            hash_object.update(str(user_id).encode("utf-8"))
            hashed_user_id = hash_object.hexdigest()

        reasoning_content_key = None

        # Initialize common OpenAI variables first
        if "openai_api_key" not in kwargs:
            kwargs["openai_api_key"] = os.getenv("OPENAI_API_KEY")
        
        rest = dict(kwargs) # Create a copy to pop from for api_base etc.
        headers = {
            "Content-Type": "application/json",
        }
        if (api_key := rest.pop("openai_api_key", None)) is not None:
            headers["Authorization"] = f"Bearer {api_key}"
        api_base = rest.pop("api_base", "https://api.openai.com/v1")
        
        # Special handling for NousResearch K2/K3 models on experimental endpoint
        is_k3_model = model.startswith("NousResearch/K3-") or model.startswith("NousResearch/K2-")
        if is_k3_model:
            # api_base = "https://api.nousresearch.com/v1" # custom base via config
            # Remove authorization header for K2/K3 models (no API key required)
            if "Authorization" in headers:
                del headers["Authorization"]

        # Determine if this is an image generation model
        is_image_generation_model = model.startswith("dall-e") or model == "gpt-image-1"

        if is_image_generation_model:
            api_suffix = "/images/generations"

            # Construct the text prompt for image generation
            actual_prompt_for_image_gen = ""
            if messages:  # messages kwarg from complete()
                prompt_parts = []
                for msg_obj in messages:
                    content = msg_obj.get("content")
                    if isinstance(content, str):
                        prompt_parts.append(strip_cache_breakpoints(content))
                    elif isinstance(content, list):  # OpenAI multimodal message format
                        for part in content:
                            if part.get("type") == "text":
                                prompt_parts.append(strip_cache_breakpoints(part.get("text", "")))
                actual_prompt_for_image_gen = "\\n".join(prompt_parts).strip()
            elif prompt:  # prompt string kwarg from complete()
                actual_prompt_for_image_gen = strip_cache_breakpoints(prompt)
            
            # Truncate prompt for gpt-image-1 if it exceeds 32000 characters
            if model == "gpt-image-1" and len(actual_prompt_for_image_gen) > 32000:
                print(f"[DEBUG] Truncating gpt-image-1 prompt from {len(actual_prompt_for_image_gen)} to 32000 characters.", file=sys.stderr)
                actual_prompt_for_image_gen = actual_prompt_for_image_gen[-32000:]

            if not actual_prompt_for_image_gen:
                raise ValueError("A non-empty prompt is required for OpenAI image generation.")

            api_arguments_img = {
                "model": model,
                "prompt": actual_prompt_for_image_gen,
                "n": num_completions if num_completions is not None else 1,
                "size": kwargs.get("size", "1024x1024"),  # Default size, overridable
                "user": hashed_user_id,
                "moderation": "low",
                "quality": "auto",

            }
            # Add DALL-E 3 specific parameters if provided
            if model == "dall-e-3":
                if "quality" in kwargs: api_arguments_img["quality"] = kwargs["quality"]
                if "style" in kwargs: api_arguments_img["style"] = kwargs["style"]
            
            # Add gpt-image-1 specific parameters - default to no transparency
            if model == "gpt-image-1":
                # Default to no transparency unless explicitly overridden
                api_arguments_img["background"] = kwargs.get("background", "opaque")
            
            # Add response_format only for specific DALL-E models that support it
            if model == "dall-e-2" or model == "dall-e-3":
                api_arguments_img["response_format"] = "b64_json"

            # Remove None values
            api_arguments_img = {k: v for k, v in api_arguments_img.items() if v is not None}

            request_log_file = _log_openai_request({
                "url": api_base + api_suffix,
                "headers": headers,
                "body": api_arguments_img
            }, log_dir)

            async with session.post(api_base + api_suffix, headers=headers, json=api_arguments_img) as response:
                api_response = await response.json()
                _log_openai_response(api_response, response.status, log_dir, request_log_file)
                if response.status >= 400:
                    error_info = {
                        "request": {
                            "url": api_base + api_suffix,
                            "headers": {k: v if k.lower() != "authorization" else "Bearer [REDACTED]" for k, v in headers.items()},
                            "body": api_arguments_img
                        },
                        "response": api_response,
                        "status_code": response.status
                    }
                    _log_error(error_info)
                response.raise_for_status()
            
            return {
                "prompt": {"text": actual_prompt_for_image_gen},
                "completions": [
                    {
                        "text": img_item.get("revised_prompt", ""),  # DALL-E 3 can return revised_prompt
                        "image_data": img_item.get("b64_json"),
                        "finish_reason": {"reason": "stop"}, # Assuming success
                    }
                    for img_item in api_response.get("data", [])
                ],
                "model": model, # Actual model used
                "id": str(uuid.uuid4()), # Image API doesn't provide a top-level UUID, use created timestamp or new UUID
                "created": api_response.get("created"), # Timestamp from response
                "usage": { # Image API doesn't provide token usage
                    "vendor": vendor,
                },
            }

        if model.startswith("o1"):
            stop = []
        api_arguments = {
            "model": model,
            "prompt": "<|endoftext|>" if prompt is None else prompt,
            "temperature": temperature,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "stop": stop if stop != [] else None,
            "user": hashed_user_id,
            "logit_bias": logit_bias,
            "n": num_completions,
            **rest,
        }
        if not model.startswith("o1") and not model.startswith("o3") and not model.startswith("o4-mini") and model != "hermes-4-cot" and not model.startswith("gpt-5"):
            api_arguments["max_tokens"] = max_tokens
        elif model.startswith("gpt-5"):
            api_arguments["max_completion_tokens"] = max_tokens
            # Add verbosity parameter for GPT-5 models (except gpt-5-chat variants)
            if not model.startswith("gpt-5-chat") and "verbosity" not in api_arguments:
                api_arguments["verbosity"] = "medium"
            # Add reasoning_effort parameter for GPT-5 if provided
            if reasoning_effort is not None:
                api_arguments["reasoning_effort"] = reasoning_effort

        # Only add audio-related parameters for models that support them
        if is_openai_audio_model(model):
            if modalities:
                api_arguments["modalities"] = modalities

            # Add the 'audio' parameter block if audio is requested via modalities or explicit settings
            if (modalities and "audio" in modalities) or audio_settings:
                final_audio_settings = audio_settings or {}
                if voice:
                    final_audio_settings["voice"] = voice
                elif "voice" not in final_audio_settings:
                    final_audio_settings["voice"] = "ballad"

                if "format" not in final_audio_settings:
                    final_audio_settings["format"] = "mp3"
                
                if final_audio_settings:
                    api_arguments["audio"] = final_audio_settings
        
        # For K2/K3 models, strip optional parameters - keep only basics
        if is_k3_model:
            # Prepend BOS token to prompt (sglang doesn't add it by default)
            if add_bos_token:
                current_prompt = api_arguments.get("prompt", "")
                if current_prompt and not current_prompt.startswith("[BOS]"):
                    api_arguments["prompt"] = "[BOS]" + current_prompt
                    print(f"[DEBUG] Added [BOS] token to K2/K3 prompt", file=sys.stderr)
            else:
                print(f"[DEBUG] Skipping [BOS] token addition (add_bos_token=False)", file=sys.stderr)
            
            minimal_args = {
                "model": api_arguments.get("model"),
                "prompt": api_arguments.get("prompt"),
            }
            # Only add max_tokens, temperature, top_p, min_p, repetition_penalty, and stop if they were explicitly provided
            if max_tokens is not None:
                minimal_args["max_tokens"] = max_tokens
            if temperature is not None:
                minimal_args["temperature"] = temperature
            if top_p is not None:
                minimal_args["top_p"] = top_p
                print(f"[DEBUG] K2/K3: Adding top_p={top_p}", file=sys.stderr)
            # Support min_p parameter (minimum probability for sampling)
            if min_p is not None:
                minimal_args["min_p"] = min_p
                print(f"[DEBUG] K2/K3: Adding min_p={min_p}", file=sys.stderr)
            # Support repetition_penalty parameter (controls repetition in generation)
            if repetition_penalty is not None and repetition_penalty != 1:
                minimal_args["repetition_penalty"] = repetition_penalty
                print(f"[DEBUG] K2/K3: Adding repetition_penalty={repetition_penalty}", file=sys.stderr)
            # Support stop sequences
            if stop is not None and stop:
                minimal_args["stop"] = stop
                print(f"[DEBUG] K2/K3: Adding stop sequences: {stop}", file=sys.stderr)
            api_arguments = minimal_args
            print(f"[DEBUG] NousResearch K2/K3 model detected, using minimal parameters: {list(api_arguments.keys())}", file=sys.stderr)
        
        # remove None values, OpenAI API doesn't like them
        for key, value in dict(api_arguments).items():
            if value is None:
                del api_arguments[key]
        # Limit stop sequences for OpenAI
        if "stop" in api_arguments and isinstance(api_arguments["stop"], list):
            if not model.startswith("o3") and not model.startswith("o4-mini") and model != "hermes-4-cot" and not model.startswith("gpt-5"):
                if len(api_arguments["stop"]) > 4:
                    print(f"[DEBUG] OpenAI only supports up to 4 stop sequences. Truncating from {len(api_arguments['stop'])}.", file=sys.stderr)
                    api_arguments["stop"] = api_arguments["stop"][:4]
                # Ensure stop sequences are not empty strings, which OpenAI rejects
                api_arguments["stop"] = [s for s in api_arguments["stop"] if s]
                if not api_arguments["stop"]: # If list becomes empty after removing empty strings
                    del api_arguments["stop"]
            else:
                del api_arguments["stop"]

        # Helper function to check if force_api_mode is effectively not set
        # Helper functions for specific API modes
        def is_force_api_mode_chat(mode):
            return mode.lower() == "chat"
            
        def is_force_api_mode_completions(mode):
            return mode.lower() == "completions"
                
        print(f"[DEBUG] force_api_mode: {force_api_mode}")
        print(f"[DEBUG] is_force_api_mode_chat(force_api_mode): {is_force_api_mode_chat(force_api_mode)}")
        print(f"[DEBUG] is_force_api_mode_completions(force_api_mode): {is_force_api_mode_completions(force_api_mode)}")
        
        # Explicitly check for base models that should use completions endpoint
        is_base_model = model.endswith("-base") or model == "gpt-4-base"
        print(f"[DEBUG] is_base_model: {is_base_model}")
        
        # Check if this is an anthropic model going through OpenRouter (special handling needed)
        is_anthropic_openrouter = model.startswith("anthropic/claude-")
        print(f"[DEBUG] is_anthropic_openrouter: {is_anthropic_openrouter}")
        
        # Check if this is a nousresearch.com endpoint (requires completions mode for Hermes models)
        is_nousresearch_endpoint = "nousresearch.com" in api_base
        print(f"[DEBUG] is_nousresearch_endpoint: {is_nousresearch_endpoint}")
        
        if (
            is_force_api_mode_chat(force_api_mode) or
            (not is_force_api_mode_completions(force_api_mode)) and (
                (message_history_format is not None and message_history_format.is_chat()) or
                model.startswith("gpt-3.5") or
                model.startswith("gpt-4") or
                model.startswith("gpt-5") or
                model.startswith("o1") or
                model.startswith("openpipe:") or
                model.startswith("gpt4") or
                model.startswith("chatgpt-4o") or
                model.startswith("gpt-4.1") or
                model.startswith("grok") or
                model.startswith("deepseek-reasoner") or
                model.startswith("deepseek/deepseek-chat") or
                model.startswith("deepseek/deepseek-r1") or
                model.startswith("deepseek-ai/DeepSeek-R1-Zero") or
                model.startswith("tngtech/deepseek") or
                model.startswith("aion") or
                model.startswith("google/gemma-3-27b-it") or
                model.startswith("DeepHermes-3-Mistral-24B-Preview") or
                api_base.startswith("https://integrate.api.nvidia.com") or
                model.startswith("o3") or
                model.startswith("o4-mini") or
                model.startswith("moonshotai/") or
                (model.startswith("hermes-4") or model == "Hermes-4-405B") and not is_nousresearch_endpoint or
                is_anthropic_openrouter
            )
        ) and not is_base_model and not is_k3_model:
        
            if messages is None:
                if (
                    message_history_format is not None
                    and message_history_format.is_chat() 
                ):
                    # Format messages and strip cache breakpoints for non-Anthropic models
                    messages = message_history_format.format_messages(api_arguments["prompt"], "user")
                    # Strip cache breakpoints from all message content
                    for message in messages:
                        content = message.get("content")
                        if isinstance(content, str):
                            message["content"] = strip_cache_breakpoints(content)
                        elif isinstance(content, list):
                            # Handle multimodal messages with content as list of parts
                            for part in content:
                                if part.get("type") == "text" and "text" in part:
                                    part["text"] = strip_cache_breakpoints(part["text"])
                    api_arguments["messages"] = messages
                    print(f"[DEBUG] used format_messages, message count: {len(api_arguments['messages'])}")
                else:
                    prompt_text = strip_cache_breakpoints(api_arguments.get("prompt", ""))
                    if "<|begin_of_img_url|>" in prompt_text:
                        # Multimodal message
                        content_parts = []
                        sections = re.split(r"<\|(?:begin|end)_of_img_url\|>", prompt_text)
                        for i, section in enumerate(sections):
                            if i % 2 == 0: # text
                                if section.strip():
                                    content_parts.append({"type": "text", "text": section.strip()})
                            else: # image url
                                url = section.strip()
                                if url:
                                    content_parts.append({"type": "image_url", "image_url": {"url": url}})
                        
                        api_arguments["messages"] = [{"role": "user", "content": content_parts}]
                        print(f"[DEBUG] chat history sent as a single multimodal user message")
                    else:
                        # Original logic for text-only prompt
                        api_arguments["messages"] = [
                            {
                                "role": "system",
                                "content": f"Respond to the chat, where your username is shown as {name}. Only respond with the content of your message, without including your username.",
                            },
                            {"role": "user", "content": prompt_text},
                        ]
                        print(f"[DEBUG] chat history sent as a single user message")
            else:
                # Strip cache breakpoints from provided messages for non-Anthropic models
                if messages:
                    for message in messages:
                        content = message.get("content")
                        if isinstance(content, str):
                            message["content"] = strip_cache_breakpoints(content)
                        elif isinstance(content, list):
                            # Handle multimodal messages with content as list of parts
                            for part in content:
                                if part.get("type") == "text" and "text" in part:
                                    part["text"] = strip_cache_breakpoints(part["text"])
                api_arguments["messages"] = messages
                print(f"[DEBUG] messages sent as is, message count: {len(api_arguments['messages'])}")
            if "prompt" in api_arguments:
                del api_arguments["prompt"]
            if "logprobs" in api_arguments:
                del api_arguments["logprobs"]
            if model.startswith("o1") or model.startswith("deepseek") or api_base.startswith("https://integrate.api.nvidia.com") or model.startswith("aion") or model.startswith("grok") or model.startswith("o3") or model.startswith("o4-mini") or model.startswith("gpt-5") or model == "Hermes-4-405B":
                if "logit_bias" in api_arguments:
                    del api_arguments["logit_bias"]
                # Remove presence_penalty, frequency_penalty, and stop for grok models as they don't support them
                if model.startswith("grok"):
                    if "presence_penalty" in api_arguments:
                        del api_arguments["presence_penalty"]
                    if "frequency_penalty" in api_arguments:
                        del api_arguments["frequency_penalty"]
                    if "stop" in api_arguments:
                        del api_arguments["stop"]
                # Remove frequency_penalty and presence_penalty for GPT-5 as they don't support them
                if model.startswith("gpt-5"):
                    if "presence_penalty" in api_arguments:
                        del api_arguments["presence_penalty"]
                    if "frequency_penalty" in api_arguments:
                        del api_arguments["frequency_penalty"]
                if (
                    model.startswith("o1")
                    or model.startswith("o3")
                    or model.startswith("o4-mini")
                    or model.startswith("gpt-5")
                    or model.startswith("chatgpt-4o")
                    or model.startswith("deepseek-reasoner")
                    or model.startswith("deepseek/deepseek-r1")
                    or model.startswith("deepseek-ai/DeepSeek-R1-Zero")
                    or model.startswith("aion")
                    or model.startswith("deepseek/deepseek-chat")
                    or model.startswith("tngtech/deepseek")
                    or model == "Hermes-4-405B"
                ):
                    if api_base.startswith("https://openrouter.ai"):
                        reasoning_content_key = "reasoning"
                        api_arguments["include_reasoning"] = True
                    else:
                        reasoning_content_key = "reasoning_content"
            # Remove empty logit_bias for NVIDIA endpoints
            api_suffix = "/chat/completions"
            
            # Special handling for Anthropic models through OpenRouter
            if is_anthropic_openrouter and "messages" in api_arguments:
                anthropic_prompt = convert_messages_to_anthropic_prompt(api_arguments["messages"])
                api_arguments["prompt"] = anthropic_prompt
                del api_arguments["messages"]
                print(f"[DEBUG] Converted messages to Anthropic prompt format for OpenRouter")
                
                # Handle thinking parameter for OpenRouter Anthropic models
                if "thinking" in kwargs:
                    thinking_config = kwargs.pop("thinking")
                    if isinstance(thinking_config, dict):
                        if "type" not in thinking_config:
                            thinking_config["type"] = "enabled"
                        api_arguments["thinking"] = thinking_config
                    elif isinstance(thinking_config, bool) and thinking_config:
                        api_arguments["thinking"] = {
                            "type": "enabled",
                            "budget_tokens": max(2048, max_tokens // 2)
                        }
                    print(f"[DEBUG] Added thinking parameter for OpenRouter Anthropic model")
        else:                
            api_suffix = "/completions"
            
            # Strip cache breakpoints from prompt for completions API
            if "prompt" in api_arguments and isinstance(api_arguments["prompt"], str):
                api_arguments["prompt"] = strip_cache_breakpoints(api_arguments["prompt"])
            
            # Handle message format conversion for chat → completions mode
            if message_history_format is not None and message_history_format.is_chat():
                formatted_messages = []
                
                if messages is None:
                    messages = message_history_format.format_messages(prompt, "user")
                
                # Convert messages to a single prompt string
                for msg in messages:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    
                    if isinstance(content, list):
                        # For multi-modal content, just extract text parts
                        text_parts = []
                        for item in content:
                            if item.get("type") == "text":
                                text_parts.append(strip_cache_breakpoints(item.get("text", "")))
                        content = " ".join(text_parts)
                    else:
                        # Strip cache breakpoints from text content
                        content = strip_cache_breakpoints(content)
                    
                    # Format based on role
                    if role == "system":
                        formatted_messages.append(f"System: {content}")
                    elif role == "user":
                        formatted_messages.append(f"User: {content}")
                    elif role == "assistant":
                        formatted_messages.append(f"Assistant: {content}")
                    else:
                        formatted_messages.append(f"{role.capitalize()}: {content}")
                # Join all messages with newlines
                api_arguments["prompt"] = "\n\n".join(formatted_messages)
                print(f"[DEBUG] Converted chat messages to completions prompt: {len(api_arguments['prompt'])} chars")
                
        print(f"[DEBUG] Using endpoint: {api_base + api_suffix}")
        if api_suffix == "/chat/completions":
            if 'messages' in api_arguments:
                print(f"[DEBUG] message count: {len(api_arguments['messages'])}")
            else:
                print(f"[DEBUG] using prompt instead of messages (Anthropic OpenRouter mode)")
            
            # Only prepare messages for chat completions endpoint (except Anthropic OpenRouter models)
            if not is_anthropic_openrouter:
                api_arguments['messages'] = await prepare_openai_messages(
                    api_arguments.get('messages'),
                    api_arguments.get('prompt'),
                    session,
                    model
                )
                # Clean up prompt if we have messages now
                if 'prompt' in api_arguments and api_arguments['messages']:
                    del api_arguments['prompt']
        else:
            # For completions endpoint, ensure we have prompt and no messages
            print(f"[DEBUG] Using completions endpoint with prompt: {len(api_arguments.get('prompt', ''))} chars")
            if 'messages' in api_arguments:
                del api_arguments['messages']

        # Log the request before sending
        request_log_file = _log_openai_request({
            "url": api_base + api_suffix,
            "headers": headers,
            "body": api_arguments
        }, log_dir)

        async with session.post(
            api_base + api_suffix, headers=headers, json=api_arguments
        ) as response:
            api_response = await response.json()

            # Log the response immediately after receiving it
            _log_openai_response(api_response, response.status, log_dir, request_log_file)

            # Existing error logging for 400 status
            if response.status == 400:
                error_info = {
                    "request": {
                        "url": api_base + api_suffix,
                        "headers": {k: v if k.lower() != "authorization" else "Bearer [REDACTED]" for k, v in headers.items()},
                        "body": api_arguments
                    },
                    "response": api_response
                }
                _log_error(error_info)
            response.raise_for_status()

        print(f"API Response: {api_response}")

        try:
            completions = []
            for choice in api_response["choices"]:
                text = ""
                audio = None
                reasoning_content = None

                if api_suffix == "/completions":
                    text = choice.get("text", "").strip()
                elif is_anthropic_openrouter:
                    # OpenRouter anthropic models return text directly in choice, not under message
                    text = choice.get("text", "").strip()
                    if reasoning_content_key:
                        reasoning_content = choice.get(reasoning_content_key)
                else:
                    message = choice.get("message", {})
                    if message:
                        content = message.get("content")
                        if content is not None:
                            text = content.strip()

                        audio_data = message.get("audio")
                        if audio_data:
                            audio = audio_data
                            if not text and audio_data.get("transcript"):
                                text = audio_data["transcript"]

                        if reasoning_content_key:
                            reasoning_content = message.get(reasoning_content_key)

                # Convert literal \n to actual newlines for OpenAI responses
                if text:
                    text = convert_literal_newlines_for_openai(text)

                completions.append(
                    {
                        "text": text,
                        "audio": audio,
                        "finish_reason": {
                            "reason": choice.get("finish_reason", "unknown")
                        },
                        "reasoning_content": reasoning_content,
                    }
                )
            print(f"[DEBUG] completions: {completions}")
            return {
                "prompt": {"text": prompt if prompt is not None else "<|endoftext|>"},
                "completions": completions,
                "model": api_response.get("model"),
                "id": api_response.get("id"),
                "created": api_response.get("created"),
                "usage": {
                    # "prompt_tokens": api_response.usage.prompt_tokens,
                    # # if the completion is empty, the value will be missing
                    # "completion_tokens": api_response.usage.get("completion_tokens", 0),
                    # "charged_tokens": api_response.usage.total_tokens,
                    "vendor": vendor,
                },
            }
        except KeyError:
            import json
            print(f"API Response: {api_response}")
            if "error" in api_response:
                raise ValueError(
                    "API returned error: " + json.dumps(api_response["error"])
                )
            else:
                raise ValueError("API responded with: " + json.dumps(api_response))
    elif vendor == "ai21":
        import httpx

        async with httpx.AsyncClient() as client:
            http_response = await client.post(
                f"https://api.ai21.com/studio/v1/{model}/complete",
                headers={
                    "Authorization": "Bearer "
                    + kwargs.get("ai21_api_key", os.environ.get("AI21_API_KEY"))
                },
                json={
                    "prompt": strip_cache_breakpoints(prompt),
                    "numResults": num_completions or 1,
                    "maxTokens": max_tokens,
                    # "stopSequences": stop,
                    "topP": top_p,
                    "temperature": temperature,
                    "frequencyPenalty": {"scale": frequency_penalty},
                    "presencePenalty": {"scale": presence_penalty},
                    **kwargs,
                },
            )
        http_response.raise_for_status()
        api_response = http_response.json()
        completion_tokens = sum(
            [
                len(completion["data"]["tokens"])
                for completion in api_response["completions"]
            ]
        )
        return {
            "prompt": {"text": api_response["prompt"]["text"]},
            "completions": [
                {
                    "text": completion["data"]["text"],
                    "finish_reason": completion["finishReason"]["reason"],
                }
                for completion in api_response["completions"]
            ],
            "model": model,
            "id": api_response["id"],
            "usage": {
                "prompt_tokens": len(api_response["prompt"]["tokens"]),
                "completion_tokens": completion_tokens,
                "charged_tokens": completion_tokens,
                "vendor": vendor,
            },
        }
    elif vendor == "textsynth":
        pass
    elif vendor == "huggingface":
        import httpx

        # todo: freq and pres penalties
        async with httpx.AsyncClient() as client:
            http_response = await client.post(
                f"https://api-inference.huggingface.co/models/{model}",
                headers={
                    "Authorization": "Bearer "
                    + kwargs.get(
                        "huggingface_api_key", os.environ.get("HUGGINGFACE_API_KEY")
                    )
                },
                json={
                    "inputs": strip_cache_breakpoints(prompt),
                    "max_new_tokens": max_tokens,
                    "do_sample": True,
                    "top_p": top_p,
                    "top_k": top_k,
                    "temperature": temperature,
                    "repetition_penalty": repetition_penalty,
                    **kwargs,
                },
            )
    elif vendor == "forefront":
        import httpx

        # Strip cache breakpoints first
        prompt = strip_cache_breakpoints(prompt)
        if "t5-20b" in model:
            prompt = prompt + " <extra_id_0>"
        async with httpx.AsyncClient() as client:
            http_response = await client.post(
                f"https://shared-api.{model}",
                headers={
                    "Authorization": "Bearer "
                    + kwargs.get("forefront_api_key", os.getenv("FOREFRONT_API_KEY"))
                },
                json={
                    "text": prompt,
                    "top_p": top_p,
                    "top_k": 50400 or top_k,
                    "temperature": temperature,
                    "tfs": tfs,
                    "length": max_tokens,
                    "repetition_penalty": repetition_penalty,
                    "stop": stop,
                },
            )
        http_response.raise_for_status()
        api_response = http_response.json()
        return {
            "prompt": {"text": prompt},
            "model": api_response["model"],
            "completions": [
                {"text": output["completion"]} for output in api_response["result"]
            ],
            "created": api_response["timestamp"],
            # forefront bills both the prompt and completion
            "usage": NotImplemented,
        }
    elif vendor.startswith("anthropic"):
        import anthropic
        import json
        import os
        import sys
        import datetime

        if num_completions not in [None, 1]:
            raise NotImplementedError("Anthropic only supports num_completions=1")
        client = anthropic.AsyncAnthropic(
            api_key=kwargs.get("anthropic_api_key", os.getenv("ANTHROPIC_API_KEY"))
        )
        if "anthropic_api_key" in kwargs:
            del kwargs["anthropic_api_key"]

        # remove None values, Anthropic API doesn't like them
        for key, value in dict(kwargs).items():
            if value is None:
                del kwargs[key]

        # Handle thinking parameter for Claude 3 (direct Anthropic API only)
        if model.startswith("claude-") and "thinking" in kwargs:
            thinking_config = kwargs.pop("thinking")
            if isinstance(thinking_config, dict):
                if "type" not in thinking_config:
                    thinking_config["type"] = "enabled"
                kwargs["thinking"] = thinking_config
            elif isinstance(thinking_config, bool) and thinking_config:
                kwargs["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": max(2048, max_tokens // 2)
                }

        if messages is None:
            if (
                message_history_format is not None
                and message_history_format.is_chat()
            ):
                if cache_breakpoints:
                    # Process messages with cache marker support
                    processed_messages = []
                    for message in message_history_format.format_messages(prompt, "user"):
                        if isinstance(message["content"], str):
                            # Process both cache markers and images
                            parts = process_message_with_cache_and_images(
                                message["content"], 
                                cache_type=cache_type
                            )
                            processed_messages.append(ProcessedMessage(
                                role=message["role"],
                                parts=parts
                            ))
                        else:
                            # If content is already processed, just wrap it
                            processed_messages.append(ProcessedMessage(
                                role=message["role"],
                                parts=[MessagePart(type="text", content=str(message["content"]))]
                            ))
                    
                    # Convert to Anthropic format
                    import asyncio
                    messages = await asyncio.gather(*[
                        convert_to_anthropic_format(msg, session) for msg in processed_messages
                    ])
                else:
                    # Original processing without cache markers
                    messages = [
                        {
                            "role": message["role"],
                            "content": process_image_message(message["content"], role=message["role"]).get("content"),
                        }
                        for message in message_history_format.format_messages(
                            prompt, "user"
                        )
                    ]
            else:
                if kwargs.get("system", None) is None:
                    kwargs["system"] = (
                        "The assistant is in CLI simulation mode, and responds to the user's CLI commands only with the output of the command."
                    )
                    messages = [
                        {"role": "user", "content": "<cmd>cat untitled.txt</cmd>"}
                    ]
                else:
                    messages = []

                # Process into intermediate format first
                processed_messages = []
                
                if cache_breakpoints:
                    # Process with cache marker support for prefill mode
                    parts = process_message_with_cache_and_images(prompt, cache_type=cache_type)
                    
                    # In prefill mode, we need to split content appropriately
                    # Content up to and including the last image goes to user message
                    # Content after the last image goes to assistant message
                    last_non_text_idx = -1
                    for idx, part in enumerate(parts):
                        if part.type != "text":
                            last_non_text_idx = idx
                    
                    if last_non_text_idx == -1:
                        # No images, all text goes to assistant message
                        processed_messages.append(ProcessedMessage(
                            role="assistant",
                            parts=parts
                        ))
                    else:
                        # Split at the last image
                        user_parts = parts[:last_non_text_idx + 1]
                        assistant_parts = parts[last_non_text_idx + 1:]
                        
                        if user_parts:
                            processed_messages.append(ProcessedMessage(
                                role="user",
                                parts=user_parts
                            ))
                        if assistant_parts:
                            processed_messages.append(ProcessedMessage(
                                role="assistant",
                                parts=assistant_parts
                            ))
                    
                    # Skip the original image/text processing since we've already handled it
                    skip_original_processing = True
                else:
                    skip_original_processing = False
                    # Original processing without cache markers
                    sections = re.split(r"<\|(?:begin|end)_of_img_url\|>", prompt)
                    
                    # Extract text parts and image URLs
                    text_parts = []
                    image_urls = []
                
                if not skip_original_processing:
                    for i, section in enumerate(sections):
                        if i % 2 == 0 and section.strip():
                            text_parts.append(section.strip())
                        elif i % 2 == 1:  # Image URL
                            image_urls.append(section.strip())
                    
                    # Respect max_images parameter
                    if max_images == 0:
                        # If max_images is 0, don't process any images
                        image_urls = []
                        print(f"[DEBUG] No images allowed", file=sys.stderr)
                    elif max_images < len(image_urls):
                        # Only keep the last max_images images
                        image_urls = image_urls[-max_images:]
                        print(f"[DEBUG] Limiting to {len(image_urls)} images", file=sys.stderr)
                    else:
                        print(f"[DEBUG] Processing {len(image_urls)} images", file=sys.stderr)

                    # For non-chat mode with images, we need to:
                    # 1. Keep all content in order (text/image/text/image) in the USER message until the last image
                    # 2. Combine all remaining text into the ASSISTANT message
                    
                    if len(image_urls) > 0:
                        # Create user message maintaining order of text and images
                        user_msg_parts = []
                        assistant_text_parts = []
                        img_index = 0
                        last_image_index = -1
                        
                        # First find the last image section index
                        for i, section in enumerate(sections):
                            if i % 2 == 1:  # Image section
                                if img_index < len(image_urls):
                                    last_image_index = i
                                    img_index += 1
                        
                        # Reset image index for actual processing
                        img_index = 0
                        
                        # Process all sections in order
                        for i, section in enumerate(sections):
                            if i % 2 == 0:  # Text section
                                if section.strip():
                                    if last_image_index == -1 or i < last_image_index:
                                        # Text before or between images goes to user message
                                        user_msg_parts.append(MessagePart(
                                            type="text",
                                            content=section.strip()
                                        ))
                                    else:
                                        # Text after last image goes to assistant
                                        assistant_text_parts.append(section.strip())
                            else:  # Image section
                                if img_index < len(image_urls):
                                    user_msg_parts.append(MessagePart(
                                        type="image",
                                        content=image_urls[img_index],
                                        mime_type=guess_mime_type(image_urls[img_index])
                                    ))
                                    img_index += 1
                        
                        # Add user message with ordered content
                        if user_msg_parts:
                            processed_messages.append(ProcessedMessage(
                                role="user",
                                parts=user_msg_parts
                            ))
                        
                        # Add combined text parts as assistant message
                        assistant_text = " ".join(assistant_text_parts)
                        processed_messages.append(ProcessedMessage(
                            role="assistant",
                            parts=[MessagePart(type="text", content=assistant_text)]
                        ))
                    else:
                        # No images - just add all text as assistant message
                        text_parts_combined = " ".join(text_parts)
                        if text_parts_combined:
                            processed_messages.append(ProcessedMessage(
                                role="assistant",
                                parts=[MessagePart(type="text", content=text_parts_combined)]
                            ))

                # Convert all processed messages to Anthropic format with URL validation
                import asyncio
                processed_messages = await asyncio.gather(*[
                    convert_to_anthropic_format(msg, session) for msg in processed_messages
                ])

                messages = messages + processed_messages

        if vendor == "anthropic-steering-preview":
            #kwargs["extra_headers"] = {"anthropic-beta": "steering-2024-06-04"}
            if "steering" in kwargs:
                kwargs["extra_body"] = {"steering": kwargs["steering"]}
                del kwargs["steering"]
        
        # Add 1M context beta headers for models ending with -1m
        if model.endswith("-1m"):
            # Remove the -1m suffix from the model name for the actual API call
            model = model[:-3]
            if "extra_headers" not in kwargs:
                kwargs["extra_headers"] = {}
            existing_beta = kwargs["extra_headers"].get("anthropic-beta", "")
            tokens = [t.strip() for t in existing_beta.split(",") if t.strip()]
            if "context-1m-2025-08-07" not in tokens:
                tokens.append("context-1m-2025-08-07")
            kwargs["extra_headers"]["anthropic-beta"] = ", ".join(tokens)
        
        # Add cache control beta headers if cache_breakpoints is enabled
        if cache_breakpoints:
            if "extra_headers" not in kwargs:
                kwargs["extra_headers"] = {}
            existing_beta = kwargs["extra_headers"].get("anthropic-beta", "")
            tokens = [t.strip() for t in existing_beta.split(",") if t.strip()]
            for token in [
                "prompt-caching-2024-07-31",
                "extended-cache-ttl-2025-04-11",
            ]:
                if token not in tokens:
                    tokens.append(token)
            kwargs["extra_headers"]["anthropic-beta"] = ", ".join(tokens)

        # Create a deep copy of the request to log
        request_payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens or 16,
            "temperature": temperature,
            "top_p": top_p if 'thinking' not in kwargs and model != "claude-opus-4-1-20250805" else None,
            "stop_sequences": stop or list(),
        }
        # Add any extra kwargs
        for k, v in kwargs.items():
            request_payload[k] = v
        
        # Remove None values for cleaner output
        request_payload = {k: v for k, v in request_payload.items() if v is not None}
        
        # Log the full JSON request, with API key redacted
        print(f"[DEBUG] Full Anthropic API request:", file=sys.stderr)
        # Use a custom encoder to handle non-serializable objects
        class CustomEncoder(json.JSONEncoder):
            def default(self, obj):
                return str(obj)
        
        # Pretty print the JSON for readability
        # Remove old logging
        # with open("anthropic_request.json", "w") as f:
        #     f.write(json.dumps(request_payload, indent=2, cls=CustomEncoder))
        # print(json.dumps(request_payload, indent=2, cls=CustomEncoder), file=sys.stderr)

        # Use new logging function if log_dir is set
        request_log_file = None
        if log_dir:
            request_log_file = _log_anthropic_request(request_payload, log_dir)

        if top_p is None:
            top_p = 1

        if 'thinking' in kwargs:
            response = await client.messages.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens or 16,
                temperature=temperature,
                stop_sequences=stop or list(),
                **kwargs,
            )
        else:
            # Newer Claude models (4.1+) don't support both temperature and top_p
            # Check if this is a Claude 4.x or higher model (not Claude 3.x)
            is_newer_claude = (model.startswith("claude-") and 
                             ("-4-" in model or "-5-" in model or "-6-" in model or 
                              "-7-" in model or "-8-" in model or "-9-" in model))
            
            if is_newer_claude:
                response = await client.messages.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens or 16,
                    temperature=temperature or 1,
                    stop_sequences=stop or list(),
                    **kwargs,
                )
            else:
                response = await client.messages.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens or 16,
                    temperature=temperature,
                    top_p=top_p,
                    stop_sequences=stop or list(),
                    **kwargs,
                )

        # Log the response
        if log_dir:
            _log_anthropic_response(response, log_dir, request_log_file)

        if response.stop_reason == "stop_sequence":
            finish_reason = "stop"
        elif response.stop_reason == "max_tokens":
            finish_reason = "length"
        elif response.stop_reason == "refusal":
            finish_reason = "refusal"
        else:
            finish_reason = response.stop_reason

        # Extract thinking content and text if available
        reasoning_content = None
        text_content = ""
        
        for content_block in response.content:
            if content_block.type == "thinking":
                reasoning_content = content_block.thinking
            elif content_block.type == "text":
                text_content = content_block.text

        return {
            "prompt": {
                "text": prompt,
            },
            "completions": [
                {
                    "text": text_content,
                    "finish_reason": finish_reason,
                    "reasoning_content": reasoning_content
                }
            ],
            "model": model,
            "id": uuid.uuid4(),
            "created": datetime.datetime.now(),
            "usage": {
                "charged_tokens": 0,
                "vendor": vendor,
            },
        }
    elif vendor == "bedrock" or vendor == "aws-bedrock":
        from anthropic import AnthropicBedrock
        import os
        import sys
        import datetime

        if num_completions not in [None, 1]:
            raise NotImplementedError("Bedrock only supports num_completions=1")
        
        # Set up Bedrock client with AWS credentials
        aws_region = kwargs.get("aws_region", os.getenv("AWS_REGION", "us-west-2"))
        aws_access_key = kwargs.get("aws_access_key", os.getenv("AWS_ACCESS_KEY_ID"))
        aws_secret_key = kwargs.get("aws_secret_key", os.getenv("AWS_SECRET_ACCESS_KEY"))
        
        if not aws_access_key or not aws_secret_key:
            raise ValueError("AWS credentials required for Bedrock. Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables.")
        
        client = AnthropicBedrock(
            aws_region=aws_region,
            aws_access_key=aws_access_key,
            aws_secret_key=aws_secret_key
        )
        
        # Clean up AWS credentials from kwargs
        for key in ["aws_region", "aws_access_key", "aws_secret_key"]:
            if key in kwargs:
                del kwargs[key]
        
        # Remove None values, Bedrock API doesn't like them
        for key, value in dict(kwargs).items():
            if value is None:
                del kwargs[key]

        # Handle thinking parameter for Bedrock Claude models
        if model.startswith("anthropic.claude-") and "thinking" in kwargs:
            thinking_config = kwargs.pop("thinking")
            if isinstance(thinking_config, dict):
                if "type" not in thinking_config:
                    thinking_config["type"] = "enabled"
                kwargs["thinking"] = thinking_config
            elif isinstance(thinking_config, bool) and thinking_config:
                kwargs["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": max(2048, max_tokens // 2)
                }

        if messages is None:
            if (
                message_history_format is not None
                and message_history_format.is_chat()
            ):
                messages = [
                    {
                        "role": message["role"],
                        "content": process_image_message(message["content"], role=message["role"]).get("content"),
                    }
                    for message in message_history_format.format_messages(
                        prompt, "user"
                    )
                ]
                
                # Convert image URLs to base64 for Bedrock compatibility
                messages = await convert_messages_for_bedrock(messages, session)
            else:
                if kwargs.get("system", None) is None:
                    kwargs["system"] = (
                        "The assistant is in CLI simulation mode, and responds to the user's CLI commands only with the output of the command."
                    )
                    messages = [
                        {"role": "user", "content": "<cmd>cat untitled.txt</cmd>"}
                    ]
                else:
                    messages = []

                # Process into intermediate format first
                processed_messages = []
                import re
                sections = re.split(r"<\|(?:begin|end)_of_img_url\|>", prompt)
                
                # Extract text parts and image URLs
                text_parts = []
                image_urls = []
                
                for i, section in enumerate(sections):
                    if i % 2 == 0 and section.strip():
                        text_parts.append(section.strip())
                    elif i % 2 == 1:  # Image URL
                        image_urls.append(section.strip())
                
                # Respect max_images parameter
                if max_images == 0:
                    # If max_images is 0, don't process any images
                    image_urls = []
                    print(f"[DEBUG] No images allowed", file=sys.stderr)
                elif max_images < len(image_urls):
                    # Only keep the last max_images images
                    image_urls = image_urls[-max_images:]
                    print(f"[DEBUG] Limiting to {len(image_urls)} images", file=sys.stderr)
                else:
                    print(f"[DEBUG] Processing {len(image_urls)} images", file=sys.stderr)

                # For non-chat mode with images, create appropriate message structure
                if len(image_urls) > 0:
                    # Create user message maintaining order of text and images
                    user_msg_parts = []
                    assistant_text_parts = []
                    img_index = 0
                    last_image_index = -1
                    
                    # First find the last image section index
                    for i, section in enumerate(sections):
                        if i % 2 == 1:  # Image section
                            if img_index < len(image_urls):
                                last_image_index = i
                                img_index += 1
                    
                    # Reset image index for actual processing
                    img_index = 0
                    
                    # Process all sections in order
                    for i, section in enumerate(sections):
                        if i % 2 == 0:  # Text section
                            if section.strip():
                                if last_image_index == -1 or i < last_image_index:
                                    # Text before or between images goes to user message
                                    user_msg_parts.append(MessagePart(
                                        type="text",
                                        content=section.strip()
                                    ))
                                else:
                                    # Text after last image goes to assistant
                                    assistant_text_parts.append(section.strip())
                        else:  # Image section
                            if img_index < len(image_urls):
                                user_msg_parts.append(MessagePart(
                                    type="image",
                                    content=image_urls[img_index],
                                    mime_type=guess_mime_type(image_urls[img_index])
                                ))
                                img_index += 1
                    
                    # Add user message with ordered content
                    if user_msg_parts:
                        processed_messages.append(ProcessedMessage(
                            role="user",
                            parts=user_msg_parts
                        ))
                    
                    # Add combined text parts as assistant message
                    assistant_text = " ".join(assistant_text_parts)
                    processed_messages.append(ProcessedMessage(
                        role="assistant",
                        parts=[MessagePart(type="text", content=assistant_text)]
                    ))
                else:
                    # No images - just add all text as assistant message
                    text_parts_combined = " ".join(text_parts)
                    if text_parts_combined:
                        processed_messages.append(ProcessedMessage(
                            role="assistant",
                            parts=[MessagePart(type="text", content=text_parts_combined)]
                        ))

                # Convert all processed messages to Bedrock format with base64 encoding
                import asyncio
                processed_messages = await asyncio.gather(*[
                    convert_to_bedrock_format(msg, session) for msg in processed_messages
                ])

                messages = messages + processed_messages
        
        # If messages were provided directly, ensure they're Bedrock-compatible
        elif messages is not None:
            messages = await convert_messages_for_bedrock(messages, session)

        # Log the request
        request_log_file = None
        if log_dir:
            request_payload = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens or 16,
                "temperature": temperature or 1,
                "top_p": top_p,
                "stop_sequences": stop or list(),
            }
            # Add any extra kwargs
            for k, v in kwargs.items():
                request_payload[k] = v
            
            request_log_file = _log_bedrock_request(request_payload, log_dir)

        print(f"[DEBUG] Sending Bedrock request with model: {model}")
        # Run synchronous Bedrock call in thread pool to avoid blocking event loop
        import asyncio
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: client.messages.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens or 16,
                temperature=temperature or 1,
                top_p=top_p or 1,
                stop_sequences=stop or list(),
                **kwargs,
            )
        )

        # Log the response
        if log_dir:
            _log_bedrock_response(response, log_dir, request_log_file)

        if response.stop_reason == "stop_sequence":
            finish_reason = "stop"
        elif response.stop_reason == "max_tokens":
            finish_reason = "length"
        else:
            finish_reason = "unknown"

        # Extract thinking content and text if available
        reasoning_content = None
        text_content = ""
        
        for content_block in response.content:
            if content_block.type == "thinking":
                reasoning_content = content_block.thinking
            elif content_block.type == "text":
                text_content = content_block.text

        return {
            "prompt": {
                "text": prompt,
            },
            "completions": [
                {
                    "text": text_content,
                    "finish_reason": finish_reason,
                    "reasoning_content": reasoning_content
                }
            ],
            "model": model,
            "id": response.id,
            "created": datetime.datetime.now(),
            "usage": {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "charged_tokens": response.usage.input_tokens + response.usage.output_tokens,
                "vendor": vendor,
            },
        }
    elif vendor == "replicate":
        import httpx

        async with httpx.AsyncClient() as client:
            initial_response = await client.post(
                "https://api.replicate.com/v1/predictions",
                json={
                    "version": model.split(":")[1],
                    "prompt": prompt,
                    "max_length": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "repetition_penalty": repetition_penalty,
                    "stop_sequence": stop[0],
                },
            )
            initial_response.raise_for_status()
            initial_response_json = initial_response.json()
            response = initial_response
            while response.json()["status"] != "succeeded":
                await asyncio.sleep(0.25)
                response = await client.get(
                    f"https://api.replicate.com/v1/predictions/{initial_response_json['id']}"
                )
                response.raise_for_status()
        return {"prompt": {"text": prompt}, "completions": {}}
    elif vendor == "fake-local":
        return intermodel.callgpt_faker.fake_local(
            model=tokenize_as,
            vendor=vendor,
            prompt=prompt,
            max_tokens=max_tokens,
            num_completions=num_completions,
        )
    elif vendor == "gemini":
        from google import genai
        from google.genai import types
        import base64
        from io import BytesIO
        from PIL import Image
        import sys
        import datetime
        import os
        import requests
        from mimetypes import guess_type

        if "google_api_key" not in kwargs:
            kwargs["google_api_key"] = os.getenv("GOOGLE_API_KEY")
        
        client = genai.Client(api_key=kwargs["google_api_key"])
        
        print(f"[DEBUG] Sending request to Gemini model: {model}", file=sys.stderr)
        print(f"[DEBUG] Content to send: {prompt[:150]}{'...' if len(prompt) > 300 else ''}{prompt[-150:] if len(prompt) > 300 else ''}", file=sys.stderr)
        
        # Convert messages to format expected by Gemini for both text and image models
        messages = message_history_format.format_messages(prompt, "user")
        print(f"[DEBUG] Extracted {len(messages)} messages from chat history format", file=sys.stderr)
        
        # Process the messages to extract text and images
        gemini_contents = []
        processed_messages_intermediate: List[ProcessedMessage] = []
        
        if messages is not None:
            print(f"[DEBUG] Processing {len(messages)} provided messages for Gemini", file=sys.stderr)
            # Convert existing message format to ProcessedMessage
            # First determine if there will be a valid last message that should be a 'model' role
            last_message_index = len(messages) - 1
            
            for msg in messages:
                role = msg.get("role", "user")
                content = strip_cache_breakpoints(msg.get("content", ""))
                parts = []
                
                # Determine speaker name for formatting
                # Check if this is the last message that needs to be a 'model' role
                is_last_message = (msg == messages[last_message_index])
                if is_last_message:
                    role = 'assistant'  # Force the last message to have 'model' role
                
                speaker_name = msg.get('name')
                 
                # Fallback if name not in message dict
                if speaker_name is None:
                    if role in ['user', 'system']:
                        speaker_name = name # Use the top-level name passed to complete()
                    elif role == 'assistant' and message_history_format:
                        speaker_name = message_history_format.assistant_name
                    # else: speaker_name remains None if role is unknown and not in msg dict

                name_prefix = ""  # Default to no prefix
                # Only apply prefix for non-assistant roles and not for the last message (which must be 'model' role)
                if role != 'assistant' and not is_last_message and speaker_name and message_history_format and hasattr(
                        message_history_format, 'name_format') and message_history_format.name_format:
                    try:
                        name_prefix = message_history_format.name_format.format(
                            speaker_name) + " "  # Add space after prefix
                    except KeyError:
                        # Handle cases where format string might be incorrect (e.g., expects {name} but gets other keys)
                        print(
                            f"[WARN] Could not format name prefix for '{speaker_name}' using format '{message_history_format.name_format}'. Using raw name.",
                            file=sys.stderr)
                        name_prefix = f"{speaker_name}: "  # Fallback prefix

                formatted_content = name_prefix + content
                if isinstance(formatted_content, str):
                    # Simple text message - check for embedded image URLs
                    print(
                        f"[DEBUG] Processing text message for images: {formatted_content[:200]}{'...' if len(formatted_content) > 200 else ''}",
                        file=sys.stderr)
                    import re
                    sections = re.split(r"<\|(?:begin_of_img_url|end_of_img_url)\|>", formatted_content)
                    print(f"[DEBUG] Split into {len(sections)} sections", file=sys.stderr)

                    # Extract text parts and image URLs
                    text_parts = []
                    image_urls = []

                    for i, section in enumerate(sections):
                        if i % 2 == 0:  # Text section
                            if section.strip():
                                text_parts.append(section.strip())
                        else:  # Image URL
                            image_urls.append(section.strip())

                    # Add text part if any text exists
                    combined_text = " ".join(text_parts)
                    if combined_text:
                        parts.append(MessagePart(type="text", content=combined_text))

                    # Add image parts
                    for url in image_urls:
                        if url:
                            try:
                                mime_type = guess_mime_type(url)
                                print(f"[DEBUG] Processing image URL: {url[:100]}{'...' if len(url) > 100 else ''}",
                                      file=sys.stderr)
                                # Handle data URLs
                                if url.startswith("data:"):
                                    try:
                                        mime_type_part, data_part = url.split(';', 1)
                                        encoding_part, b64_data = data_part.split(',', 1)
                                        mime_type = mime_type_part.split(':')[1]
                                        parts.append(MessagePart(type="image", content=url, mime_type=mime_type))
                                        print(
                                            f"[DEBUG] Processed data URL image from message text ({len(b64_data)} chars)",
                                            file=sys.stderr)
                                    except Exception as e:
                                        print(f"[DEBUG] Failed to process data URL in message: {e}", file=sys.stderr)
                                else:
                                    parts.append(MessagePart(type="image", content=url, mime_type=mime_type))
                                    print(
                                        f"[DEBUG] Added image URL to parts: {url[:100]}{'...' if len(url) > 100 else ''}",
                                        file=sys.stderr)
                            except Exception as e:
                                print(
                                    f"[DEBUG] Error processing image URL {url[:100]}{'...' if len(url) > 100 else ''}: {e}",
                                    file=sys.stderr)
                                # Skip this image and continue with text-only
                elif isinstance(formatted_content, list):
                    # OpenAI-style list of parts
                    for i, part_data in enumerate(formatted_content):
                        part_type = part_data.get("type")
                        if part_type == "text":
                            original_text = part_data.get("text", "")
                            # Apply prefix only to the first text part if there are multiple parts
                            formatted_text = (name_prefix + original_text) if i == 0 else original_text
                            parts.append(MessagePart(type="text", content=formatted_text))
                        elif part_type == "image_url":
                            image_url_data = part_data.get("image_url", {})
                            url = image_url_data.get("url")
                            if url:
                                # Check if data URL
                                if url.startswith("data:"):
                                    # Handle data URLs (extract mime type and base64 data)
                                    try:
                                        mime_type_part, data_part = url.split(';', 1)
                                        encoding_part, b64_data = data_part.split(',', 1)
                                        mime_type = mime_type_part.split(':')[1]
                                        # Gemini needs raw bytes, so decode base64
                                        image_bytes = base64.b64decode(b64_data)
                                        # We need to pass the raw bytes somehow, maybe store temp?
                                        # For now, let's just pass the URL and handle download in convert_to_gemini
                                        parts.append(MessagePart(type="image", content=url, mime_type=mime_type))
                                        print(f"[DEBUG] Processed data URL image for Gemini ({len(image_bytes)} bytes)",
                                              file=sys.stderr)
                                    except Exception as e:
                                        print(f"[DEBUG] Failed to process data URL: {e}", file=sys.stderr)
                                else:
                                    # Regular URL
                                    mime_type = guess_mime_type(url)
                                    parts.append(MessagePart(type="image", content=url, mime_type=mime_type))
                else:
                    print(f"[DEBUG] Unknown content type in message: {type(formatted_content)}", file=sys.stderr)
                
                if parts:
                    # Create the processed message with the potentially modified role
                    processed_messages_intermediate.append(ProcessedMessage(role=role, parts=parts))
            
        elif prompt is not None:
            # Process the prompt string into ProcessedMessage objects
            print(f"[DEBUG] Processing prompt string for Gemini", file=sys.stderr)
            # Strip cache breakpoints from prompt
            prompt = strip_cache_breakpoints(prompt)
            sections = re.split(r"<\|(?:begin|end)_of_img_url\|>", prompt)
                        
                        # Extract text parts and image URLs
            text_parts_all = []
            image_urls_all = []
                        
            for i, section in enumerate(sections):
                if i % 2 == 0:  # Text section
                    if section.strip():
                        text_parts_all.append(section.strip())
                else:  # Image URL
                    image_urls_all.append(section.strip())
                        
                        # Respect max_images parameter
            images_to_process_urls = image_urls_all
            if max_images is not None and max_images < len(image_urls_all):
                images_to_process_urls = image_urls_all[-max_images:]
                print(f"[DEBUG] Limiting to {len(images_to_process_urls)} images for Gemini", file=sys.stderr)
            else:
                print(f"[DEBUG] Processing {len(images_to_process_urls)} images for Gemini", file=sys.stderr)
                
            # Construct ProcessedMessage parts
            # Gemini expects alternating user/model roles.
            # If images are present, send all text and images as a single user message.
            # If only text, send as a single user message.
            current_parts: List[MessagePart] = []
            img_idx = 0

            # Add all text parts first
            original_combined_text = " ".join(text_parts_all)

            # Determine speaker name and format prefix for prompt case
            prompt_speaker_name = name  # Assuming prompt comes from the 'user' named 'name'
            prompt_name_prefix = ""
            if prompt_speaker_name and message_history_format and hasattr(message_history_format,
                                                                          'name_format') and message_history_format.name_format:
                try:
                    prompt_name_prefix = message_history_format.name_format.format(
                        prompt_speaker_name) + " "  # Add space
                except KeyError:
                    print(
                        f"[WARN] Could not format name prefix for prompt user '{prompt_speaker_name}' using format '{message_history_format.name_format}'. Using raw name.",
                        file=sys.stderr)
                    prompt_name_prefix = f"{prompt_speaker_name}: "  # Fallback prefix

            combined_text = prompt_name_prefix + original_combined_text

            if combined_text:
                current_parts.append(MessagePart(type="text", content=combined_text))

            # Add image parts
            for url in images_to_process_urls:
                mime_type = guess_mime_type(url)
                # Handle data URLs similarly to the 'messages' logic if needed
                if url.startswith("data:"):
                    try:
                        mime_type_part, data_part = url.split(';', 1)
                        encoding_part, b64_data = data_part.split(',', 1)
                        mime_type = mime_type_part.split(':')[1]
                        # Pass the data URL itself; download logic is in convert_to_gemini_format
                        current_parts.append(MessagePart(type="image", content=url, mime_type=mime_type))
                        print(f"[DEBUG] Added data URL image part for Gemini", file=sys.stderr)
                    except Exception as e:
                        print(f"[DEBUG] Failed to process data URL in prompt: {e}", file=sys.stderr)
                else:
                    current_parts.append(MessagePart(type="image", content=url, mime_type=mime_type))

            if current_parts:
                # Assume the entire prompt maps to a single 'user' message
                processed_messages_intermediate.append(ProcessedMessage(role="user", parts=current_parts))

        else:  # No messages and no prompt
            print(f"[DEBUG] No messages or prompt provided for Gemini.", file=sys.stderr)
            # Potentially add a default empty user message if API requires it
            # processed_messages_intermediate.append(ProcessedMessage(role="user", parts=[MessagePart(type="text", content="")]))

        # Convert processed messages to Gemini format using chunked inline approach
        print(f"[DEBUG] Converting {len(processed_messages_intermediate)} processed messages for Gemini (chunked)",
              file=sys.stderr)

        # Build contents list preserving order of text and images (chunked approach)
        gemini_contents = []

        for processed_msg in processed_messages_intermediate:
            try:
                # Determine role prefix
                role_prefix = ""
                if processed_msg.role == "assistant" or processed_msg.role == "model":
                    role_prefix = "Assistant: "
                elif processed_msg.role == "user":
                    role_prefix = "User: "
                elif processed_msg.role == "system":
                    role_prefix = "System: "

                # Process each part in order, preserving text/image sequence
                for part_idx, part in enumerate(processed_msg.parts):
                    if part.type == "text":
                        # Add role prefix only to the first text part of each message
                        text_content = part.content
                        if part_idx == 0:
                            text_content = role_prefix + text_content

                        gemini_contents.append(text_content)
                        print(
                            f"[DEBUG] Added text chunk: '{text_content[:100]}{'...' if len(text_content) > 100 else ''}'",
                            file=sys.stderr)

                    elif part.type == "image":
                        # Download image and convert to PIL Image
                        try:
                            print(
                                f"[DEBUG] Downloading image for Gemini: {part.content[:100]}{'...' if len(part.content) > 100 else ''}",
                                file=sys.stderr)

                            if part.content.startswith("data:"):
                                # Handle data URLs
                                import base64
                                header, data = part.content.split(',', 1)
                                image_bytes = base64.b64decode(data)
                            else:
                                # Handle regular URLs
                                import requests
                                response = requests.get(part.content)
                                response.raise_for_status()
                                image_bytes = response.content

                            # Convert to PIL Image
                            from PIL import Image
                            from io import BytesIO
                            image = Image.open(BytesIO(image_bytes))
                            gemini_contents.append(image)
                            print(f"[DEBUG] Added image chunk: {image.size} ({image.mode})", file=sys.stderr)

                        except Exception as e:
                            print(
                                f"[DEBUG] Failed to load image {part.content[:100]}{'...' if len(part.content) > 100 else ''}: {e}",
                                file=sys.stderr)
                            # Skip this image but continue processing

            except Exception as e:
                print(f"[DEBUG] Error processing message: {e}", file=sys.stderr)
                continue

        # Count final content types for debug
        text_count = sum(1 for item in gemini_contents if isinstance(item, str))
        image_count = len(gemini_contents) - text_count
        print(
            f"[DEBUG] Built chunked Gemini contents: {text_count} text chunks, {image_count} image chunks (total: {len(gemini_contents)})",
            file=sys.stderr)

        # Ensure gemini_contents is not empty if the API requires at least one message
        if not gemini_contents:
            # If all messages were skipped (e.g., only contained GIFs), send an empty user message
            print("[DEBUG] No valid messages left after conversion; sending empty text.", file=sys.stderr)
            gemini_contents.append("(empty message)")  # Use placeholder text

        print(f"[DEBUG] Sending request with {len(gemini_contents)} content objects", file=sys.stderr)

        # Debug log content being sent
        print("[DEBUG] Content being sent:", file=sys.stderr)
        for i, content in enumerate(gemini_contents):
            if isinstance(content, str):
                content_preview = content[:200] + '...' if len(content) > 200 else content
                print(f"[DEBUG] Content {i + 1}: Text: '{content_preview}'", file=sys.stderr)
            else:
                # Assume PIL Image
                try:
                    print(f"[DEBUG] Content {i + 1}: Image: {content.size} ({content.mode})", file=sys.stderr)
                except:
                    print(f"[DEBUG] Content {i + 1}: Unknown type: {type(content)}", file=sys.stderr)

        # Configure the request based on the model type
        is_image_generation = model in ["gemini-2.0-flash-exp", "gemini-2.5-flash-image-preview"]

        try:
            if is_image_generation:
                # Create request data dictionary for logging
                request_data = {
                    "model": model,
                    "contents": gemini_contents,
                    "config": {
                        "response_modalities": ['Text', 'Image'],
                        "safety_settings": "BLOCK_NONE for all categories"
                    }
                }

                # Log the request
                request_log_file = None
                if log_dir:
                    request_log_file = _log_gemini_request(request_data, log_dir)

                # For image generation models
                response = client.models.generate_content(
                    model=model,
                    contents=gemini_contents,
                    config=types.GenerateContentConfig(
                        response_modalities=['Text', 'Image'],
                        safety_settings=[
                            types.SafetySetting(
                                category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                                threshold=types.HarmBlockThreshold.BLOCK_NONE,
                            ),
                            types.SafetySetting(
                                category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                                threshold=types.HarmBlockThreshold.BLOCK_NONE,
                            ),
                            types.SafetySetting(
                                category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                                threshold=types.HarmBlockThreshold.BLOCK_NONE,
                            ),
                            types.SafetySetting(
                                category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                                threshold=types.HarmBlockThreshold.BLOCK_NONE,
                            ),
                            types.SafetySetting(
                                category=types.HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY,
                                threshold=types.HarmBlockThreshold.BLOCK_NONE,
                            ),
                        ]
                    )
                )
                
                # Log the response
                if log_dir:
                    _log_gemini_response(response, log_dir, request_log_file)
                
                # Process the response for image generation
                text_content = ""
                image_data = None

                # iterate through all response fields and print the


                if not response.candidates or len(response.candidates) == 0:
                    print(f"[DEBUG] No candidates returned from Gemini", file=sys.stderr)
                    print(f"[DEBUG] Response: {response}", file=sys.stderr)
                    raise Exception("No candidates returned from Gemini")
                

                for candidate in response.candidates:
                    if not hasattr(candidate, "content") or candidate.content is None:
                        print(f"[DEBUG] Candidate has no content, skipping", file=sys.stderr)
                        continue

                    for part in candidate.content.parts:
                        if part.text is not None:
                            text_content = part.text
                            print(
                                f"[DEBUG] Received text response: {text_content[:100]}{'...' if len(text_content) > 100 else ''}",
                                file=sys.stderr)
                        if part.inline_data is not None:
                            image_data = part.inline_data.data
                            image_size = len(image_data) if image_data else 0
                            print(f"[DEBUG] Received image data: {image_size} bytes", file=sys.stderr)
                            
                            # Save image to file for debugging
                            if image_data:
                                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                                debug_dir = os.path.join(os.getcwd(), "debug_images")
                                os.makedirs(debug_dir, exist_ok=True)
                                image_filename = os.path.join(debug_dir, f"gemini_image_{timestamp}.png")
                                
                                try:
                                    with open(image_filename, "wb") as f:
                                        f.write(image_data)
                                    print(f"[DEBUG] Saved image to: {image_filename}", file=sys.stderr)
                                except Exception as e:
                                    print(f"[DEBUG] Failed to save image: {str(e)}", file=sys.stderr)
                    
                return {
                    "prompt": {"text": prompt},
                    "completions": [
                        {
                            "text": text_content,
                            "finish_reason": "stop",
                            "image_data": base64.b64encode(image_data).decode() if image_data else None
                        }
                    ],
                    "model": model,
                    "id": str(uuid.uuid4()),
                    "created": datetime.datetime.now(),
                    "usage": {
                        "vendor": vendor,
                    }
                }
            else:
                # Create request data dictionary for logging
                request_data = {
                    "model": model,
                    "contents": gemini_contents,
                    "config": {
                        "temperature": temperature or 1.0,
                        "top_p": top_p or 1.0,
                        "top_k": top_k or 40,
                        "max_output_tokens": max_tokens or 4000,
                        "stop_sequences": stop or [],
                        "safety_settings": "BLOCK_NONE for all categories"
                    }
                }
                
                # Log the request
                request_log_file = None
                if log_dir:
                    request_log_file = _log_gemini_request(request_data, log_dir)
                
                # For regular text models
                response = client.models.generate_content(
                    model=model,
                    contents=gemini_contents,
                    # config=types.GenerateContentConfig(
                    #     temperature=temperature or 1.0,
                    #     top_p=top_p or 1.0,
                    #     top_k=top_k or 40,
                    #     max_output_tokens=max_tokens or 4000,
                    #     #stop_sequences=stop or [],
                    #     safety_settings=[
                    #         types.SafetySetting(
                    #             category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                    #             threshold=types.HarmBlockThreshold.BLOCK_NONE,
                    #         ),
                    #         types.SafetySetting(
                    #             category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                    #             threshold=types.HarmBlockThreshold.BLOCK_NONE,
                    #         ),
                    #         types.SafetySetting(
                    #             category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                    #             threshold=types.HarmBlockThreshold.BLOCK_NONE,
                    #         ),
                    #         types.SafetySetting(
                    #             category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                    #             threshold=types.HarmBlockThreshold.BLOCK_NONE,
                    #         ),
                    #         types.SafetySetting(
                    #             category=types.HarmCategory.HARM_CATEGORY_CIVIC_INTEGRITY,
                    #             threshold=types.HarmBlockThreshold.BLOCK_NONE,
                    #         ),
                    #     ]
                    # )
                )
                
                # Log the response
                if log_dir:
                    _log_gemini_response(response, log_dir, request_log_file)

                if response.text:
                   print(f"[DEBUG] Received response: {response.text[:100]}{'...' if len(response.text) > 100 else ''}", file=sys.stderr)
                else:
                    print(f"[DEBUG] Received response with no text: {response}", file=sys.stderr)
                    raise Exception("No text returned from Gemini: " + str(response))
         

                return {
                    "prompt": {"text": prompt},
                    "completions": [
                        {
                            "text": response.text,
                            "finish_reason": "stop"
                        }
                    ],
                    "model": model,
                    "id": str(uuid.uuid4()),
                    "created": datetime.datetime.now(),
                    "usage": {
                        "vendor": vendor,
                    }
                }
        except Exception as e:
            print(f"[DEBUG] Error generating content: {str(e)}", file=sys.stderr)
            raise
    else:
        raise NotImplementedError(f"Unknown vendor {vendor}")


def download_and_process_image(url: str) -> bytes:
    """Download image and return raw bytes.
    
    Args:
        url (str): URL of the image to download
        
    Returns:
        bytes: Raw image data
    """
    import requests
    import sys

    print(f"[DEBUG] Downloading image for Gemini from: {url[:100]}{'...' if len(url) > 100 else ''}", file=sys.stderr)
    response = requests.get(url)
    response.raise_for_status()
    image_data = response.content
    mime_type = response.headers.get("content-type") or guess_mime_type(url)
    
    # Gemini doesn't support GIFs
    if mime_type and mime_type.lower() == "image/gif":
        print(f"[DEBUG] Skipping GIF image as it's not supported by Gemini", file=sys.stderr)
        # TODO: Consider converting GIF to a supported format (e.g., PNG) instead of skipping
        raise ValueError("Gemini does not support GIF images.") 
        
    print(f"[DEBUG] Downloaded image: {len(image_data)} bytes, mime type: {mime_type}", file=sys.stderr)
    return image_data


async def convert_messages_for_bedrock(messages: List[dict], session: Optional[aiohttp.ClientSession] = None) -> List[dict]:
    """Convert messages with image URLs to use base64 for Bedrock compatibility.
    
    Args:
        messages: List of message dicts that may contain image URLs
        session: Optional aiohttp session for downloading images
        
    Returns:
        List[dict]: Messages with image URLs converted to base64
    """
    if not messages:
        return messages
    
    # Create session if not provided
    if session is None:
        import aiohttp
        async with aiohttp.ClientSession() as new_session:
            return await convert_messages_for_bedrock(messages, new_session)
    
    converted_messages = []
    
    for message in messages:
        content = message.get("content", "")
        
        # If content is a list (multimodal), process each part
        if isinstance(content, list):
            new_content = []
            for part in content:
                if part.get("type") == "image" and "source" in part:
                    source = part["source"]
                    if source.get("type") == "url":
                        # Convert URL to base64
                        url = source["url"]
                        try:
                            base64_data, mime_type = await download_and_encode_image_for_bedrock(url, session)
                            if base64_data:
                                new_content.append({
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": mime_type or "image/jpeg",
                                        "data": base64_data
                                    }
                                })
                            else:
                                # Replace with text indicating skipped image
                                new_content.append({
                                    "type": "text",
                                    "text": f"[Image URL was not accessible and has been skipped]"
                                })
                        except Exception as e:
                            print(f"[ERROR] Failed to convert image URL for Bedrock: {str(e)}", file=sys.stderr)
                            new_content.append({
                                "type": "text",
                                "text": f"[Error processing image URL]"
                            })
                    else:
                        # Already base64 or other format, keep as-is
                        new_content.append(part)
                else:
                    # Non-image part, keep as-is
                    new_content.append(part)
            
            converted_messages.append({
                **message,
                "content": new_content
            })
        else:
            # Text-only content, keep as-is
            converted_messages.append(message)
    
    return converted_messages


async def download_and_encode_image_for_bedrock(url: str, session: aiohttp.ClientSession) -> Tuple[Optional[str], Optional[str]]:
    """Download image and encode it as base64 for Bedrock.
    
    Args:
        url (str): URL of the image to download
        session: aiohttp session for downloading
        
    Returns:
        Tuple[Optional[str], Optional[str]]: (base64_data, mime_type) or (None, None) if failed
    """
    import base64
    import sys

    print(f"[DEBUG] Downloading and encoding image for Bedrock from: {url[:100]}{'...' if len(url) > 100 else ''}", file=sys.stderr)
    
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
            response.raise_for_status()
            image_data = await response.read()
            mime_type = response.headers.get("content-type") or guess_mime_type(url)
            
            # Check image size
            size_mb = len(image_data) / (1024 * 1024)
            if size_mb > 5:
                print(f"[WARNING] Image size {size_mb:.2f}MB exceeds 5MB limit for Bedrock, skipping", file=sys.stderr)
                return None, None
            
            # Skip GIFs as they're often not well supported
            if mime_type and mime_type.lower() == "image/gif":
                print(f"[DEBUG] Skipping GIF image for Bedrock", file=sys.stderr)
                return None, None
                
            # Encode to base64
            base64_data = base64.b64encode(image_data).decode('utf-8')
            print(f"[DEBUG] Encoded image for Bedrock: {len(image_data)} bytes, mime type: {mime_type}", file=sys.stderr)
            
            return base64_data, mime_type
            
    except Exception as e:
        print(f"[ERROR] Failed to download/encode image for Bedrock: {str(e)}", file=sys.stderr)
        return None, None


def download_and_encode_image(url: str, skip_gifs: bool = False) -> Tuple[Optional[str], Optional[str]]:
    """Download image and encode it as base64.
    
    Args:
        url (str): URL of the image to download
        skip_gifs (bool): Whether to skip GIF images
        
    Returns:
        Tuple[Optional[str], Optional[str]]: (base64_data, mime_type) or (None, None) if skipped
    """
    import requests
    import sys
    import base64

    print(f"[DEBUG] Downloading and encoding image from: {url[:100]}{'...' if len(url) > 100 else ''}", file=sys.stderr)
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        image_data = response.content
        mime_type = response.headers.get("content-type") or guess_mime_type(url)
        
        # Skip GIFs if requested
        if skip_gifs and mime_type and mime_type.lower() == "image/gif":
            print(f"[DEBUG] Skipping GIF image as requested", file=sys.stderr)
            return None, None
            
        # Check image size
        size_mb = len(image_data) / (1024 * 1024)
        if size_mb > 5:
            print(f"[WARNING] Image size {size_mb:.2f}MB exceeds 5MB limit, skipping", file=sys.stderr)
            return None, None
            
        # Encode to base64
        base64_data = base64.b64encode(image_data).decode('utf-8')
        print(f"[DEBUG] Encoded image: {len(image_data)} bytes, mime type: {mime_type}", file=sys.stderr)
        
        return base64_data, mime_type
        
    except requests.RequestException as e:
        print(f"[ERROR] Failed to download image: {str(e)}", file=sys.stderr)
        raise
    except Exception as e:
        print(f"[ERROR] Failed to encode image: {str(e)}", file=sys.stderr)
        raise


def process_cache_markers(content_string: str, cache_type: str = "ephemeral") -> List[MessagePart]:
    """Process a message that may contain cache breakpoint markers.
    
    Args:
        content_string (str): The message content that may contain cache markers
        cache_type (str): The type of cache control to apply: "ephemeral", "5m", "1h"
        
    Returns:
        List[MessagePart]: A list of MessagePart objects with cache_control metadata
    
    Notes:
        - Cache markers are specified using <|cache_breakpoint|>
        - Content before the last cache marker gets cache_control
        - Content after the last cache marker does not get cached
        - Cache types:
          - "ephemeral": Default caching behavior
          - "5m": Cache for 5 minutes
          - "1h": Cache for 1 hour
    """
    import re
    
    # Map user-friendly cache types to API values. Support legacy aliases for backwards compatibility.
    cache_type_map = {
        "ephemeral": {"type": "ephemeral"},
        "5m": {"type": "ephemeral", "ttl": "5m"},
        "1h": {"type": "ephemeral", "ttl": "1h"},
        # Legacy aliases
        "5min": {"type": "ephemeral", "ttl": "5m"},
        "1hour": {"type": "ephemeral", "ttl": "1h"},
    }
    
    cache_control = cache_type_map.get(cache_type, {"type": "ephemeral"})
    
    sections = re.split(r"<\|cache_breakpoint\|>", content_string)
    if len(sections) == 1:
        # No cache markers, return single part without cache control
        return [MessagePart(type="text", content=content_string)]
    
    parts = []
    for i, section in enumerate(sections[:-1]):  # All sections except the last get cached
        if section.strip():
            parts.append(MessagePart(
                type="text",
                content=section,
                cache_control=cache_control
            ))
    
    # Last section doesn't get cache control
    if sections[-1].strip():
        parts.append(MessagePart(
            type="text",
            content=sections[-1]
        ))
    
    return parts


def process_message_with_cache_and_images(content_string: str, cache_type: str = "ephemeral", skip_gifs: bool = False) -> List[MessagePart]:
    """Process a message that may contain both cache markers and image markers.
    
    Args:
        content_string (str): The message content that may contain cache and/or image markers
        cache_type (str): The type of cache control to apply: "ephemeral", "5m", "1h"
        skip_gifs (bool): Whether to skip GIF images
        
    Returns:
        List[MessagePart]: A list of MessagePart objects with appropriate metadata
    
    Notes:
        - Cache markers: <|cache_breakpoint|>
        - Image markers: <|begin_of_img_url|>URL<|end_of_img_url|>
        - Cache markers apply to all content (text and images) before them
        - Cache types:
          - "ephemeral": Default caching behavior
          - "5m": Cache for 5 minutes
          - "1h": Cache for 1 hour
    """
    import re
    
    # Map user-friendly cache types to API values. Support legacy aliases for backwards compatibility.
    cache_type_map = {
        "ephemeral": {"type": "ephemeral"},
        "5m": {"type": "ephemeral", "ttl": "5m"},
        "1h": {"type": "ephemeral", "ttl": "1h"},
        # Legacy aliases
        "5min": {"type": "ephemeral", "ttl": "5m"},
        "1hour": {"type": "ephemeral", "ttl": "1h"},
    }
    
    cache_control = cache_type_map.get(cache_type, {"type": "ephemeral"})
    
    # First split by cache markers
    cache_sections = re.split(r"<\|cache_breakpoint\|>", content_string)
    
    all_parts = []
    
    for cache_idx, cache_section in enumerate(cache_sections):
        # Determine if this section should have cache control
        should_cache = cache_idx < len(cache_sections) - 1
        
        # Process images within this cache section
        image_sections = re.split(r"<\|(?:begin|end)_of_img_url\|>", cache_section)
        
        # First, collect all parts for this cache section
        section_parts = []
        
        if len(image_sections) == 1:
            # No images in this section
            if cache_section.strip():
                section_parts.append(MessagePart(
                    type="text",
                    content=cache_section.strip(),
                    cache_control=None  # Will be set later if needed
                ))
        else:
            # Process sections with images
            for i, section in enumerate(image_sections):
                if i % 2 == 0:  # Text section
                    if section.strip():
                        section_parts.append(MessagePart(
                            type="text",
                            content=section.strip(),
                            cache_control=None  # Will be set later if needed
                        ))
                else:  # Image URL section
                    mime_type = guess_mime_type(section)
                    if not (skip_gifs and mime_type == "image/gif"):
                        section_parts.append(MessagePart(
                            type="image",
                            content=section,
                            mime_type=mime_type
                        ))
        
        # If this section should be cached, apply cache_control ONLY to the LAST part
        if should_cache and section_parts:
            # Apply cache control to the last part in this section
            section_parts[-1].cache_control = cache_control
        
        # Add all parts from this section to the overall list
        all_parts.extend(section_parts)
    
    return all_parts


def strip_cache_breakpoints(content_string):
    """Remove cache breakpoint markers from content.
    
    Args:
        content_string (str): The content that may contain cache breakpoint markers
        
    Returns:
        str: Content with cache breakpoint markers removed
    """
    if isinstance(content_string, str):
        return re.sub(r'<\|cache_breakpoint\|>', '', content_string)
    return content_string


def process_image_message(content_string, skip_gifs=False, role="user"):
    """Process a message that may contain image URLs markers.
    
    Args:
        content_string (str): The message content that may contain image URL markers
        skip_gifs (bool): Whether to skip GIF images
        role (str): The role of the message sender ('user', 'system', 'assistant', etc.)
        
    Returns:
        list: A list of content parts (text and images)
    """
    import requests
    import sys
    import base64
    from google.genai import types

    sections = re.split(r"<\|(?:begin|end)_of_img_url\|>", content_string)
    if len(sections) == 1:
        return content_string
    
    print(f"[DEBUG] Processing message with {(len(sections)-1)//2} embedded images", file=sys.stderr)
    content = []
    for i, section in enumerate(sections):
        if i % 2 == 0:
            if section.strip():
                content.append({"type": "text", "text": section})
        else:
            try:
                print(f"[DEBUG] Processing image URL: {section[:100]}{'...' if len(section) > 100 else ''}", file=sys.stderr)
                result = download_and_encode_image(section, skip_gifs=skip_gifs)
                if result[0] is None:  # Skip if image was filtered out
                    continue
                base64_data, mime_type = result
                content.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": mime_type,
                            "data": base64_data,
                        },
                    }
                )
                print(f"[DEBUG] Added image to message content", file=sys.stderr)
            except requests.RequestException as e:
                print(f"[DEBUG] Failed to download image: {str(e)}", file=sys.stderr)
                continue
    
    # Add role information for future compatibility with Gemini format
    gemini_role = role
    if role == 'assistant':
        gemini_role = 'model'
    elif role not in ['user', 'model']:
        gemini_role = 'user'
    
    return {"role": gemini_role, "content": content}


def complete_sync(*args, **kwargs):
    return asyncio.run(complete(*args, **kwargs))


def tokenize(model: str, string: str) -> List[int]:
    import tiktoken

    model = parse_model_string(MODEL_ALIASES.get(model, model)).tokenize_as
    try:
        vendor = pick_vendor(model)
    except NotImplementedError:
        vendor = None
    # actual tokenizer for claude 3.x models is unknown
    #print(f"[DEBUG] Tokenizing {model} with vendor {vendor}", file=sys.stderr)
    if vendor == "openai" or vendor == "bedrock" or vendor == "aws-bedrock" or model == "gpt2" or model.startswith("anthropic/claude") or model.startswith("anthropic.") or model.startswith("claude-3") or model == "claude-opus-4-1-20250805" or model.startswith(
            "chatgpt-4o") or model.startswith("gpt-4o") or model.startswith("grok") or model.startswith("aion") or model.startswith(
            "DeepHermes") or model.startswith("google/gemma-3") or model.startswith("gemini-") or model.startswith(
            "deepseek") or model.startswith("deepseek/deepseek-r1") or model.startswith("deepseek-ai/DeepSeek-R1-Zero") or model.startswith("tngtech/deepseek") or model.startswith("gpt-image-1") or model.startswith("moonshotai/") or model.startswith("hermes-4") or model == "Hermes-4-405B":
        # tiktoken internally caches loaded tokenizers
        #print(f"[DEBUG] Tokenizing {model} for OpenAI-compatible vendor or gpt2", file=sys.stderr) # Adjusted debug message

        # Handle OpenAI image models specifically for their text prompts
        if model.startswith("dall-e") or model == "gpt-image-1":
            # Prompts for image models are text; use a common/suitable tokenizer.
            tokenizer = tiktoken.get_encoding("gpt2")
        elif "deployedModel" in model: # Added for RunPod deployed models
            tokenizer = tiktoken.encoding_for_model("gpt-4o")
        elif model.startswith("anthropic/claude-") or model.startswith("anthropic.") or model.startswith("claude-"):
            # Handle all Claude models (including -1m suffix for 1M context beta)
            tokenizer = tiktoken.encoding_for_model("gpt2")
        elif model.startswith("o1") or model.startswith("o3") or model.startswith("o4-mini") or model.startswith("chatgpt-4o") or model.startswith("gpt-4o"):
            tokenizer = tiktoken.encoding_for_model("gpt-4o")
        elif model.startswith("gpt-4.5-preview"):
            tokenizer = tiktoken.encoding_for_model("gpt-4o")
        elif model.startswith("gpt-4.1"):
            tokenizer = tiktoken.encoding_for_model("gpt-4o")
        elif model.startswith("gpt-5"):
            tokenizer = tiktoken.encoding_for_model("gpt-4o")  # Use gpt-4o tokenizer for GPT-5
        elif model.startswith("grok"):
            tokenizer = tiktoken.encoding_for_model("gpt2")
        elif model.startswith("aion"):
            tokenizer = tiktoken.encoding_for_model("gpt2")
        elif model.startswith("google/gemma-3-27b-it"):
            tokenizer = tiktoken.encoding_for_model("gpt2")
        elif model.startswith("DeepHermes-3-Mistral-24B-Preview"):
            tokenizer = tiktoken.encoding_for_model("gpt2")
        elif model.startswith("deepseek"):
            tokenizer = tiktoken.encoding_for_model("gpt2")
        elif model.startswith("tngtech/deepseek"):
            tokenizer = tiktoken.encoding_for_model("gpt2")
        elif model.startswith("deepseek/deepseek-r1"):
            tokenizer = tiktoken.encoding_for_model("gpt2")
        elif model.startswith("deepseek-ai/DeepSeek-R1-Zero"):
            tokenizer = tiktoken.encoding_for_model("gpt2")
        elif model.startswith("meta-llama/llama-3.1-405b"):
            tokenizer = tiktoken.encoding_for_model("gpt2")
        elif model.startswith("hermes-4") or model == "Hermes-4-405B":
            tokenizer = tiktoken.encoding_for_model("gpt2")
        elif model.startswith("gemini-"):
            tokenizer = tiktoken.encoding_for_model("gpt2")  # Use GPT-2 tokenizer as approximation
        elif model.startswith("moonshotai/"):
            tokenizer = tiktoken.encoding_for_model("gpt2")  # Use GPT-2 tokenizer as approximation
        elif model.startswith("NousResearch/K3-") or model.startswith("NousResearch/K2-"):
            tokenizer = tiktoken.encoding_for_model("gpt2")  # Use GPT-2 tokenizer for K2/K3 models
        else:
            #print(f"[DEBUG] Getting tokenizer for {model}", file=sys.stderr)
            try:
                tokenizer = tiktoken.encoding_for_model(model)
            except KeyError:
                # Fall back to gpt2 tokenizer for unknown models
                tokenizer = tiktoken.get_encoding("gpt2")
            #sprint(f"[DEBUG] Tokenizer: {tokenizer}", file=sys.stderr)
        # encode special tokens as normal
        # XXX: make this an option
        return tokenizer.encode(string, allowed_special="all")
    elif vendor == "anthropic":
        # all anthropic tokenzers are private now
        return tokenize("gpt2", string)
    elif model in untokenizable:
        return tokenize("gpt2", string)
    else:
        return tokenize("gpt2", string)
        # try:
        #     print(f"[DEBUG] Getting HF tokenizer for {model}", file=sys.stderr)
        #     tokenizer = get_hf_tokenizer(model)
        # except Exception as e:
        #     message = (
        #         f"Failed to download tokenizer by looking up {model} as a huggingface model ID."
        #         "To override the tokenizer, put the correct huggingface ID ^ after the model name and follow it with a max token cap."
        #         "For example, to use OpenRouter Gemini while using the Gemma tokenizer with a maximum context window of 2 million Gemma tokens, use `google/gemini-pro-1.5^google/gemma-7b@2000000`"
        #         "Using the incorrect tokenizer may lead to flakiness and intermittent failures. Some proprietary models have openly available tokenizers."
        #     )
        #     if e.__class__.__name__ == "GatedRepoError":
        #         warnings.warn(
        #             message
        #             + f" Cause: GatedRepoError. Log in with huggingface-cli and accept the model ToS at https://huggingface.co/{model} to access the model tokenizer"
        #         )
        #     elif e.__class__.__name__ == "RepositoryNotFoundError":
        #         warnings.warn(message + f" Cause: Not found on huggingface")
        #     else:
        #         warnings.warn(message)
        #     return tokenize("gpt2", string)
        # else:
        #     return tokenizer.encode(string).ids


@dataclasses.dataclass
class Model:
    model: str
    tokenize_as: str
    max_token_len: int


def parse_model_string(model: str):
    # returns: model, tokenization model, max token length
    match = re.match(r"^(.+?)(?:\^(.+?))?(?:@(\d+))?$", model)
    tokenize_as = match.group(2) or match.group(1)
    return Model(
        model=match.group(1),
        tokenize_as=tokenize_as,
        max_token_len=(
            int(match.group(3))
            if match.group(3) is not None
            else max_token_length_inner(tokenize_as)
        ),
    )


def count_tokens(model: str, string: str) -> int:
    return len(tokenize(model, string))


def untokenize(model: str, token_ids: List[int]) -> str:
    model = MODEL_ALIASES.get(model, model)
    try:
        vendor = pick_vendor(model)
    except NotImplementedError:
        vendor = None
    if vendor == "openai" or model == "gpt2":
        import tiktoken

        # tiktoken internally caches loaded tokenizers
        # Handle OpenAI image models specifically for their text prompts
        if model.startswith("dall-e") or model == "gpt-image-1":
            print(f"[DEBUG] Using o200k_base tokenizer for OpenAI image model prompt (untokenize): {model}", file=sys.stderr)
            tokenizer = tiktoken.get_encoding("gpt2")
        else:
            tokenizer = tiktoken.encoding_for_model(model)
        return tokenizer.decode(token_ids)
    elif vendor == "anthropic":
        import anthropic

        # anthropic caches the tokenizer
        # XXX: this may send synchronous network requests, could be downloaded as part of build
        tokenizer = anthropic.get_tokenizer()
        encoded_text = tokenizer.decode(token_ids)
        return encoded_text.ids
    else:
        try:
            tokenizer = get_hf_tokenizer(model)
        except Exception as e:
            if e.__class__.__name__ == "GatedRepoError":
                raise ValueError(f"Log in with huggingface-cli to access {model}")
            elif e.__class__.__name__ == "RepositoryNotFoundError":
                raise NotImplementedError(f"I don't know how to tokenize {model}")
            else:
                raise
        else:
            return tokenizer.decode(token_ids)
        raise NotImplementedError(f"I don't know how to tokenize {model}")


def pick_vendor(model, custom_config=None):
    if custom_config is not None:
        # Try exact matches first
        for vendor_name, vendor in custom_config.items():
            if vendor["provides"] is not None:
                for pattern in vendor["provides"]:
                    if pattern == model:  # Exact match first
                        return vendor_name
        
        # Fall back to regex pattern matches
        for vendor_name, vendor in custom_config.items():
            if vendor["provides"] is not None:
                for pattern in vendor["provides"]:
                    if re.fullmatch(pattern, model):
                        return vendor_name

    model = MODEL_ALIASES.get(model, model)
    if (
        "ada" in model
        or "babbage" in model
        or "curie" in model
        or "davinci" in model
        or "cushman" in model
        or "text-moderation-" in model
        or model.startswith("ft-")
        or model.startswith("gpt-4")
        or model.startswith("gpt-5")
        or model.startswith("gpt-3.5-")
        or model.startswith("o1-")
        or model.startswith("gpt-4.")
        or model.startswith("o3")
        or model.startswith("o4-mini")
        or model.startswith("dall-e") # Added for DALL-E models
        or model == "gpt-image-1"      # Added for gpt-image-1
        or "deployedModel" in model    # Added for RunPod serverless models
    ):
        return "openai"
    elif "j1-" in model or model.startswith("j2-"):
        return "ai21"
    elif "forefront" in model:
        return "forefront"
    elif model.startswith("anthropic."):
        return "bedrock"  # anthropic.* models go through AWS Bedrock
    elif model.startswith("aws-bedrock:"):
        return "aws-bedrock"  # aws-bedrock: prefix models
    elif model.startswith("anthropic/claude-"):
        return "openai"  # anthropic/ prefix models go through OpenRouter with OpenAI API
    elif model.startswith("claude-"):
        return "anthropic"  # Includes claude-*-1m models for 1M context beta
    elif model.startswith("aion"):
        return "openai"  # aion models use OpenAI-compatible API
    elif model.startswith("google/"):
        return "openai"  # Google models go through OpenRouter
    elif model.startswith("gemini-"):
        return "gemini"  # Gemini models go directly
    elif model.startswith("moonshotai/"):
        return "openai"  # Moonshot models use OpenAI-compatible API
    elif model.startswith("hermes-4") or model == "Hermes-4-405B":
        return "openai"  # Hermes 4 models use OpenAI-compatible API
    elif "/" in model:
        return "openai"
    else:
        raise NotImplementedError("Unknown model")


def max_token_length(model):
    return parse_model_string(model).max_token_len


def max_token_length_inner(model):
    """
    The maximum number of tokens in the prompt and completion
    """
    if model == "gpt2":
        return 1024
    elif model == "gpt-4-32k":
        return 32769
    elif model.startswith("o1") or model.startswith("o3") or model.startswith("o4-mini") or model.startswith("chatgpt-4o") or model.startswith("gpt-4o"):
        return 128_000
    elif model == "gpt-4.5-preview":
        return 128_000  # gpt-4.5-preview has a 128k context window
    elif model.startswith("gpt-4.1"):
        return 128_000  # Assume gpt-4.1 has 128k context window like 4.5
    elif model.startswith("gpt-4"):
        return 8000
    elif model.startswith("gpt-5"):
        return 128_000  # Assume GPT-5 has similar context to GPT-4o
    elif model.startswith("grok"):
        return 128_000
    elif model.startswith("aion"):
        return 30_000  # 32k context window
    elif model.startswith("hermes-4") or model == "Hermes-4-405B":
        return 100_000  # 100k context window
    elif model.startswith("NousResearch/K3-") or model.startswith("NousResearch/K2-"):
        return 130_000  # 130k context window for K2/K3 models
    elif model == "code-davinci-002":
        return 8001
    elif model.startswith("code"):
        raise ValueError("Unknown maximum")
    elif model == "gpt-3.5-turbo-16k":
        return 16385
    elif model in ("babbage-002", "davinci-002"):
        return 16385
    elif model == "gpt-3.5-turbo":
        return 4097
    elif model == "gpt-3.5-turbo-instruct":
        return 4097
    elif model == "text-davinci-003" or model == "text-davinci-002":
        return 4097
    elif model == "google/gemma-7b":
        return 4097
    elif model == "google/gemma-3-27b-it":
        return 127000
    elif model == "DeepHermes-3-Mistral-24B-Preview":
        return 31000
    elif model in (
        "text-embedding-ada-002",
        "text-embedding-3-small",
        "text-embedding-3-large",
    ):
        return 8191
    elif model.startswith("text-embedding-") and model.endswith("-001"):
        return 2046
    elif model == "claude-3-sonnet-20240229-steering-preview":
        return int(18_000 * 0.7)
    elif model.startswith("anthropic.claude-3"):
        return 200_000 * 0.7
    elif model.startswith("anthropic.claude-v2") or model.startswith("anthropic.claude-2"):
        return 100_000 * 0.7
    elif model.startswith("anthropic.claude-instant"):
        return 100_000 * 0.7
    elif model.startswith("anthropic.claude"):
        return 100_000 * 0.7
    elif model.startswith("anthropic/claude-3"):
        return 200_000 * 0.7
    elif model.startswith("anthropic/claude-2.1"):
        return 200_000 * 0.7
    elif model.startswith("anthropic/claude-2"):
        return 100_000 * 0.7
    elif model.startswith("anthropic/claude"):
        return 100_000 * 0.7
    elif model.endswith("-1m"):
        # 1M context beta models - strip suffix and use 1M context
        return 1_000_000 * 0.7
    elif model.startswith("claude-3"):
        return 200_000 * 0.7
    elif model == "claude-opus-4-1-20250805":
        return 200_000 * 0.7  # Claude Opus 4.1 has similar context to Claude 3
    elif model.startswith("claude-2.1"):
        return 200_000 * 0.7
    elif model.startswith("claude-2"):
        return 100_000 * 0.7
    elif model.startswith("claude"):
        return 100_000 * 0.7
    elif (
        "ada" in model
        or "babbage" in model
        or "curie" in model
        or "davinci" in model
        or "cushman" in model
    ):
        return 2049
    elif model == "google/gemma-3-27b-it":
        return 127000
    elif model.startswith("deepseek"):
        return 127000
    elif model.startswith("tngtech/deepseek"):
        return 127000
    elif model == "DeepHermes-3-Mistral-24B-Preview":
        return 31000
    elif model.startswith("gemini-"):
        if model in ["gemini-2.0-flash-exp", "gemini-2.5-flash-image-preview"]:
            return 127000  # Image generation model has shorter context
        return 127000  # Standard Gemini models have 32k context
    elif model.startswith("moonshotai/"):
        if model == "moonshotai/kimi-k2":
            return 130000  # 130k context length
        return 130000  # Default for Moonshot models
    elif "deployedModel" in model: # Added for RunPod deployed models
        return 64000
    elif model == "dall-e-3" or model == "gpt-image-1":
        # DALL-E 3 / gpt-image-1 prompt limit is up to 4000 characters. Approx 4 chars/token => 1000 tokens
        return 6000 # User updated this value
    elif model == "dall-e-2":
        # DALL-E 2 prompt limit is 1000 characters. Approx 4 chars/token => 250 tokens
        return 250
    elif model == "deepcogito/cogito-v2-preview-deepseek-671b":
        return 163840
    else:
        return 163840
        # try:
        #     import huggingface_hub

        #     try:
        #         fname = huggingface_hub.hf_hub_download(
        #             model,
        #             "config.json",
        #             token=get_hf_auth_token(),
        #             local_files_only=True,
        #         )
        #     except LocalEntryNotFoundError:
        #         fname = huggingface_hub.hf_hub_download(
        #             model,
        #             "config.json",
        #             token=get_hf_auth_token(),
        #             local_files_only=False,
        #         )

        #     with open(fname, "r") as f:
        #         import json

        #         data = json.load(f)
        #     return data["max_position_embeddings"] + 1
        # except Exception as e:
        #     raise NotImplementedError(
        #         f"Unable to download {model} from HuggingFace. "
        #         f"Are you logged into HuggingFace (`huggingface-cli login`) and have you agreed to the model license at"
        #         f"`https://huggingface.co/{model}`?"
        #     )


class InteractiveIntermodel(cmd.Cmd):
    prompt = "intermodel> "

    def do_c(self, arg):
        """
        Send a completion request to a model
        Usage: c <model> <prompt>
        """
        model = arg.split()[0]
        prompt = arg[arg.index(model) + len(model) + 1 :]
        try:
            print(complete_sync(model, pick_vendor(model), prompt))
        except NotImplementedError:
            print(f"Not implemented for model {model}")
            print(traceback.format_exc())

    def do_t(self, arg):
        """
        Tokenize a model
        Usage: t <model> <prompt>
        """
        model = arg.split()[0]
        prompt = arg[arg.index(model) + len(model) + 1 :]
        try:
            print(tokenize(model, prompt))
        except NotImplementedError:
            print(f"Not implemented for model {model}")
            print(traceback.format_exc())

    def do_EOF(self, arg):
        """Exit"""
        return True


def _log_error(info: dict):
    """Log error information to a file with timestamp."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"intermodel_error_{timestamp}.json"
    with open(filename, "w") as f:
        json.dump(info, f, indent=2)
    print(f"\nError details written to: {filename}")


def _log_gemini_request(request_data, log_dir):
    """Log Gemini request data to a JSON file.
    
    Args:
        request_data (dict): The request data to log
        log_dir (str): The base directory for logs
    """
    import json
    import os
    import datetime
    import glob

    # Create gemini directory within log_dir if it doesn't exist
    gemini_log_dir = os.path.join(log_dir, "gemini")
    os.makedirs(gemini_log_dir, exist_ok=True)

    # Find highest existing log number
    existing_logs = glob.glob(os.path.join(gemini_log_dir, "gemini_request_*.json"))
    if existing_logs:
        last_num = max([int(os.path.basename(f).split('_')[2].split('.')[0]) for f in existing_logs])
        next_num = last_num + 1
    else:
        next_num = 1

    # Generate filename with timestamp and incrementing number
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(gemini_log_dir, f"gemini_request_{next_num:04d}_{timestamp}.json")

    # Process content to preserve text but shorten image data
    processed_data = {}

    # Deep copy dictionary with special handling for content objects
    def process_content(content_list):
        processed_contents = []
        for i, content in enumerate(content_list):
            if isinstance(content, str):
                # Handle text content (new format)
                processed_contents.append({
                    "type": "text",
                    "text": content[:1000] + "..." if len(content) > 1000 else content
                })
            else:
                # Handle PIL Image content (new format)
                try:
                    processed_contents.append({
                        "type": "image",
                        "size": content.size,
                        "mode": content.mode,
                        "format": getattr(content, 'format', 'unknown')
                    })
                except:
                    # Fallback for unknown content types
                    processed_contents.append({
                        "type": "unknown",
                        "python_type": str(type(content))
                    })
        return processed_contents

    # Process main request data
    if "model" in request_data:
        processed_data["model"] = request_data["model"]

    if "contents" in request_data:
        processed_data["contents"] = process_content(request_data["contents"])

    if "config" in request_data:
        # Convert config to dict with special handling for nested objects
        config_dict = {}
        config_source = request_data["config"]  # This is already a dictionary

        # Directly access keys from the dictionary
        if "temperature" in config_source:
            config_dict["temperature"] = config_source["temperature"]
        if "top_p" in config_source:
            config_dict["top_p"] = config_source["top_p"]
        if "top_k" in config_source:
            config_dict["top_k"] = config_source["top_k"]
        if "max_output_tokens" in config_source:
            config_dict["max_output_tokens"] = config_source["max_output_tokens"]
        if "stop_sequences" in config_source:
            config_dict["stop_sequences"] = config_source["stop_sequences"]
        # Add other potential config keys if needed (e.g., safety_settings, response_modalities)
        if "safety_settings" in config_source:
            config_dict["safety_settings"] = config_source["safety_settings"]  # Might be complex object/string
        if "response_modalities" in config_source:
            config_dict["response_modalities"] = config_source["response_modalities"]

        processed_data["config"] = config_dict

    # Write to file
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, indent=2, ensure_ascii=False)

    print(f"[DEBUG] Logged Gemini request to {filename}", file=sys.stderr)
    return filename


def _log_gemini_response(response, log_dir, request_log_file=None):
    """Log Gemini response data to a JSON file.
    
    Args:
        response: The Gemini API response to log
        log_dir (str): The base directory for logs
        request_log_file: Optional path to the corresponding request log file
    """
    import json
    import os
    import datetime
    import glob
    
    # Create gemini directory within log_dir if it doesn't exist
    gemini_log_dir = os.path.join(log_dir, "gemini")
    os.makedirs(gemini_log_dir, exist_ok=True)
    
    # Generate filename with timestamp and matching request number if available
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if request_log_file:
        # Extract the request number to maintain relationship
        basename = os.path.basename(request_log_file)
        request_num = basename.split('_')[2]
        filename = os.path.join(log_dir, f"gemini_response_{request_num}_{timestamp}.json")
    else:
        # Find highest existing log number if no request file
        existing_logs = glob.glob(os.path.join(log_dir, "gemini_response_*.json"))
        if existing_logs:
            last_num = max([int(os.path.basename(f).split('_')[2].split('.')[0]) for f in existing_logs])
            next_num = last_num + 1
        else:
            next_num = 1
        filename = os.path.join(log_dir, f"gemini_response_{next_num:04d}_{timestamp}.json")
    
    # Process response to a serializable format
    processed_data = {}
    
    # Capture all direct attributes of the response
    try:
        # Get all available attributes from the response object
        all_attrs = dir(response)
        for attr in all_attrs:
            # Skip private/internal attributes and methods
            if attr.startswith('_') or callable(getattr(response, attr)):
                continue
                
            # Add the attribute value to processed data
            try:
                attr_value = getattr(response, attr)
                # Convert value to a serializable format
                processed_data[f"_{attr}"] = str(attr_value)  # Prefix with underscore to avoid conflicts
            except Exception as e:
                processed_data[f"_{attr}_error"] = f"Error accessing {attr}: {str(e)}"
    except Exception as e:
        processed_data["_attrs_error"] = f"Error capturing response attributes: {str(e)}"
    
    # Check if response has candidates
    if hasattr(response, "candidates") and response.candidates:
        processed_data["candidates"] = []
        
        for candidate in response.candidates:
            candidate_data = {"_raw_attrs": {}}
            
            # Capture all attributes of each candidate
            try:
                for attr in dir(candidate):
                    if attr.startswith('_') or callable(getattr(candidate, attr)):
                        continue
                    try:
                        attr_value = getattr(candidate, attr)
                        if attr != "content":  # Handle content separately
                            candidate_data["_raw_attrs"][attr] = str(attr_value)
                    except Exception as e:
                        candidate_data["_raw_attrs"][f"{attr}_error"] = str(e)
            except Exception as e:
                candidate_data["_raw_attrs_error"] = str(e)
            
            if hasattr(candidate, "content") and candidate.content:
                content_data = {"parts": []}
                
                # Get role if available
                if hasattr(candidate.content, "role"):
                    content_data["role"] = candidate.content.role
                
                # Capture all other content attributes
                try:
                    for attr in dir(candidate.content):
                        if attr.startswith('_') or callable(getattr(candidate.content, attr)) or attr in ['parts', 'role']:
                            continue
                        try:
                            attr_value = getattr(candidate.content, attr)
                            content_data[f"_attr_{attr}"] = str(attr_value)
                        except Exception as e:
                            content_data[f"_attr_{attr}_error"] = str(e)
                except Exception as e:
                    content_data["_attrs_error"] = str(e)
                
                # Process the parts
                if hasattr(candidate.content, "parts"):
                    for part in candidate.content.parts:
                        part_data = {"_raw_attrs": {}}
                        
                        # Capture all part attributes
                        try:
                            for attr in dir(part):
                                if attr.startswith('_') or callable(getattr(part, attr)) or attr in ['text', 'inline_data']:
                                    continue
                                try:
                                    attr_value = getattr(part, attr)
                                    part_data["_raw_attrs"][attr] = str(attr_value)
                                except Exception as e:
                                    part_data["_raw_attrs"][f"{attr}_error"] = str(e)
                        except Exception as e:
                            part_data["_raw_attrs_error"] = str(e)
                        
                        if hasattr(part, "text") and part.text is not None:
                            part_data["type"] = "text"
                            part_data["text"] = part.text
                        elif hasattr(part, "inline_data") and part.inline_data is not None:
                            part_data["type"] = "inline_data"
                            part_data["mime_type"] = part.inline_data.mime_type
                            part_data["data_size_bytes"] = len(part.inline_data.data)
                            part_data["data_preview"] = "(binary data, truncated)"
                        
                        if part_data:
                            content_data["parts"].append(part_data)
                
                candidate_data["content"] = content_data
            
            # Add finish reason if available
            if hasattr(candidate, "finish_reason"):
                candidate_data["finish_reason"] = candidate.finish_reason
                
            # Add prompt feedback if available
            if hasattr(candidate, "prompt_feedback") and candidate.prompt_feedback:
                candidate_data["prompt_feedback"] = {}
                try:
                    for attr in dir(candidate.prompt_feedback):
                        if attr.startswith('_') or callable(getattr(candidate.prompt_feedback, attr)):
                            continue
                        try:
                            attr_value = getattr(candidate.prompt_feedback, attr)
                            candidate_data["prompt_feedback"][attr] = str(attr_value)
                        except Exception as e:
                            candidate_data["prompt_feedback"][f"{attr}_error"] = str(e)
                except Exception as e:
                    candidate_data["prompt_feedback_error"] = str(e)
                
            processed_data["candidates"].append(candidate_data)
    
    # Add simple text response if present
    if hasattr(response, "text"):
        processed_data["text"] = response.text
        
    # Add response usage information if available
    if hasattr(response, "usage") and response.usage:
        processed_data["usage"] = {}
        try:
            for attr in dir(response.usage):
                if attr.startswith('_') or callable(getattr(response.usage, attr)):
                    continue
                try:
                    attr_value = getattr(response.usage, attr)
                    processed_data["usage"][attr] = str(attr_value)
                except Exception as e:
                    processed_data["usage"][f"{attr}_error"] = str(e)
        except Exception as e:
            processed_data["usage_error"] = str(e)
    
    # Write to file
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, indent=2, ensure_ascii=False)
        
    print(f"[DEBUG] Logged Gemini response to {filename}", file=sys.stderr)
    return filename


if __name__ == "__main__":
    InteractiveIntermodel().cmdloop()


def _log_openai_request(request_data, log_dir):
    """Log OpenAI request data to a JSON file.
    
    Args:
        request_data (dict): The request data to log.
        log_dir (str): The base directory for logs.
    """
    import json
    import os
    import datetime
    import glob
    import re
    
    # Create openai subdirectory within log_dir if it doesn't exist
    openai_log_dir = os.path.join(log_dir, "openai")
    os.makedirs(openai_log_dir, exist_ok=True)
    
    # Find highest existing log number
    existing_logs = glob.glob(os.path.join(openai_log_dir, "openai_request_*.json"))
    if existing_logs:
        try:
            last_num = max([int(os.path.basename(f).split('_')[2].split('.')[0]) for f in existing_logs])
            next_num = last_num + 1
        except (IndexError, ValueError):
             # Handle cases where filename format might be unexpected
             next_num = len(existing_logs) + 1
    else:
        next_num = 1

    # Generate filename with timestamp and incrementing number
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(openai_log_dir, f"openai_request_{next_num:04d}_{timestamp}.json")
    
    # Write to file
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(request_data, f, indent=2, ensure_ascii=False)
        print(f"[DEBUG] Logged OpenAI request to {filename}", file=sys.stderr)
    except TypeError as e:
         print(f"[ERROR] Failed to serialize OpenAI request data for logging: {e}", file=sys.stderr)
         # Attempt to log with a simple string representation as fallback
         try:
             with open(filename.replace(".json", ".txt"), "w", encoding="utf-8") as f:
                 f.write(str(request_data))
             print(f"[DEBUG] Logged OpenAI request (fallback) to {filename.replace('.json', '.txt')}", file=sys.stderr)
         except Exception as fallback_e:
             print(f"[ERROR] Fallback OpenAI request logging failed: {fallback_e}", file=sys.stderr)
    except Exception as e:
         print(f"[ERROR] Failed to log OpenAI request to {filename}: {e}", file=sys.stderr)

    return filename


def _log_openai_response(response_data, status_code, log_dir, request_log_file=None):
    """Log OpenAI response data to a JSON file.
    
    Args:
        response_data (dict or str): The response data (JSON or error string).
        status_code (int): The HTTP status code of the response.
        log_dir (str): The base directory for logs.
        request_log_file: Optional path to the corresponding request log file.
    """
    import json
    import os
    import datetime
    import glob
    import re
    
    # Create openai subdirectory within log_dir if it doesn't exist
    openai_log_dir = os.path.join(log_dir, "openai")
    os.makedirs(openai_log_dir, exist_ok=True)

    # Generate filename with timestamp and matching request number if available
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if request_log_file:
        # Extract the request number to maintain relationship
        basename = os.path.basename(request_log_file)
        request_num_match = re.search(r'_(\d+)_', basename)
        if request_num_match:
            request_num = request_num_match.group(1) # Extract number part (e.g., 0001)
            filename = os.path.join(openai_log_dir, f"openai_response_{request_num}_{timestamp}.json")
        else:
            # Fallback if request filename format is unexpected
            filename = os.path.join(openai_log_dir, f"openai_response_unknown_{timestamp}.json")
    else:
        # Find highest existing log number if no request file
        existing_logs = glob.glob(os.path.join(openai_log_dir, "openai_response_*.json"))
        if existing_logs:
            try:
                last_num = max([int(os.path.basename(f).split('_')[2].split('.')[0]) for f in existing_logs if os.path.basename(f).startswith('openai_response_') and len(os.path.basename(f).split('_')) > 2])
                next_num = last_num + 1
            except (IndexError, ValueError):
                next_num = len(existing_logs) + 1 # Fallback numbering
        else:
            next_num = 1
        filename = os.path.join(openai_log_dir, f"openai_response_{next_num:04d}_{timestamp}.json")

    processed_data = {
        "status_code": status_code,
        "response_body": response_data # Assuming response_data is already JSON/dict
    }

    # Write to file
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(processed_data, f, indent=2, ensure_ascii=False)
        print(f"[DEBUG] Logged OpenAI response to {filename}", file=sys.stderr)
    except TypeError as e:
         print(f"[ERROR] Failed to serialize OpenAI response data for logging: {e}", file=sys.stderr)
         try:
             with open(filename.replace(".json", ".txt"), "w", encoding="utf-8") as f:
                 f.write(str(processed_data))
             print(f"[DEBUG] Logged OpenAI response (fallback) to {filename.replace('.json', '.txt')}", file=sys.stderr)
         except Exception as fallback_e:
              print(f"[ERROR] Fallback OpenAI response logging failed: {fallback_e}", file=sys.stderr)
    except Exception as e:
         print(f"[ERROR] Failed to log OpenAI response to {filename}: {e}", file=sys.stderr)

    return filename


def _log_anthropic_request(request_data, log_dir):
    """Log Anthropic request data to a JSON file."""
    import json
    import os
    import datetime
    import glob

    # Create anthropic subdirectory within log_dir if it doesn't exist
    anthropic_log_dir = os.path.join(log_dir, "anthropic")
    os.makedirs(anthropic_log_dir, exist_ok=True)

    # Find highest existing log number
    existing_logs = glob.glob(os.path.join(anthropic_log_dir, "anthropic_request_*.json"))
    if existing_logs:
        try:
            last_num = max([int(os.path.basename(f).split('_')[2].split('.')[0]) for f in existing_logs])
            next_num = last_num + 1
        except (IndexError, ValueError):
             # Handle cases where filename format might be unexpected
             next_num = len(existing_logs) + 1
    else:
        next_num = 1

    # Generate filename with timestamp and incrementing number
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(anthropic_log_dir, f"anthropic_request_{next_num:04d}_{timestamp}.json")

    # Use a custom encoder to handle non-serializable objects (like potentially in messages)
    class CustomEncoder(json.JSONEncoder):
        def default(self, obj):
            # Add handling for specific non-serializable types if encountered
            return str(obj) # Basic fallback

    # Write to file
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(request_data, f, indent=2, ensure_ascii=False, cls=CustomEncoder)
        print(f"[DEBUG] Logged Anthropic request to {filename}", file=sys.stderr)
    except TypeError as e:
         print(f"[ERROR] Failed to serialize Anthropic request data for logging: {e}", file=sys.stderr)
         # Attempt to log with a simple string representation as fallback
         try:
             with open(filename.replace(".json", ".txt"), "w", encoding="utf-8") as f:
                 f.write(str(request_data))
             print(f"[DEBUG] Logged Anthropic request (fallback) to {filename.replace('.json', '.txt')}", file=sys.stderr)
         except Exception as fallback_e:
             print(f"[ERROR] Fallback Anthropic request logging failed: {fallback_e}", file=sys.stderr)
    except Exception as e:
         print(f"[ERROR] Failed to log Anthropic request to {filename}: {e}", file=sys.stderr)

    return filename

# Placeholder for Anthropic response logging - can be implemented similarly if needed
# def _log_anthropic_response(response_data, status_code, request_log_file=None, log_dir="intermodel_logs"):
#     pass

def _log_anthropic_response(response_data, log_dir, request_log_file=None):
    """Log Anthropic response data to a JSON file.
    
    Args:
        response_data (dict or str): The response data (JSON or error string).
        log_dir (str): The base directory for logs.
        request_log_file: Optional path to the corresponding request log file.
    """
    import json
    import os
    import datetime
    import glob
    import re
    
    # Create anthropic subdirectory within log_dir if it doesn't exist
    anthropic_log_dir = os.path.join(log_dir, "anthropic")
    os.makedirs(anthropic_log_dir, exist_ok=True)

    # Generate filename with timestamp and matching request number if available
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if request_log_file:
        # Extract the request number to maintain relationship
        basename = os.path.basename(request_log_file)
        request_num_match = re.search(r'_(\d+)_', basename)
        if request_num_match:
            request_num = request_num_match.group(1) # Extract number part (e.g., 0001)
            filename = os.path.join(anthropic_log_dir, f"anthropic_response_{request_num}_{timestamp}.json")
        else:
            # Fallback if request filename format is unexpected
            filename = os.path.join(anthropic_log_dir, f"anthropic_response_unknown_{timestamp}.json")
    else:
        # Find highest existing log number if no request file
        existing_logs = glob.glob(os.path.join(anthropic_log_dir, "anthropic_response_*.json"))
        if existing_logs:
            try:
                last_num = max([int(os.path.basename(f).split('_')[2].split('.')[0]) for f in existing_logs if os.path.basename(f).startswith('anthropic_response_') and len(os.path.basename(f).split('_')) > 2])
                next_num = last_num + 1
            except (IndexError, ValueError):
                next_num = len(existing_logs) + 1 # Fallback numbering
        else:
            next_num = 1
        filename = os.path.join(anthropic_log_dir, f"anthropic_response_{next_num:04d}_{timestamp}.json")

    # Convert Anthropic response object to a serializable dictionary
    processed_data = {}
    try:
        # Get all available attributes from the response object
        all_attrs = dir(response_data)
        for attr in all_attrs:
            # Skip private/internal attributes and methods
            if attr.startswith('_') or callable(getattr(response_data, attr)):
                continue
            
            try:
                attr_value = getattr(response_data, attr)
                # Special handling for content which might be a list of objects
                if attr == 'content' and isinstance(attr_value, list):
                    processed_data[attr] = []
                    for item in attr_value:
                        item_dict = {}
                        for item_attr in dir(item):
                            if not item_attr.startswith('_') and not callable(getattr(item, item_attr)):
                                item_dict[item_attr] = str(getattr(item, item_attr)) # Convert to string for simplicity
                        processed_data[attr].append(item_dict)
                else:
                    processed_data[attr] = str(attr_value)  # Convert to string for simplicity
            except Exception as e:
                processed_data[f"{attr}_error"] = f"Error accessing {attr}: {str(e)}"
    except Exception as e:
        processed_data["_raw_response_data"] = str(response_data) # Fallback to string representation
        processed_data["_processing_error"] = str(e)

    # Write to file
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(processed_data, f, indent=2, ensure_ascii=False)
        print(f"[DEBUG] Logged Anthropic response to {filename}", file=sys.stderr)
    except TypeError as e:
         print(f"[ERROR] Failed to serialize Anthropic response data for logging: {e}", file=sys.stderr)
         try:
             with open(filename.replace(".json", ".txt"), "w", encoding="utf-8") as f:
                 f.write(str(processed_data))
             print(f"[DEBUG] Logged Anthropic response (fallback) to {filename.replace('.json', '.txt')}", file=sys.stderr)
         except Exception as fallback_e:
              print(f"[ERROR] Fallback Anthropic response logging failed: {fallback_e}", file=sys.stderr)
    except Exception as e:
         print(f"[ERROR] Failed to log Anthropic response to {filename}: {e}", file=sys.stderr)

    return filename


def _log_bedrock_request(request_data, log_dir):
    """Log Bedrock request data to a JSON file.
    
    Args:
        request_data (dict): The request data to log
        log_dir (str): The base directory for logs
    """
    import json
    import os
    import datetime
    import glob
    
    # Create bedrock directory within log_dir if it doesn't exist
    bedrock_log_dir = os.path.join(log_dir, "bedrock")
    os.makedirs(bedrock_log_dir, exist_ok=True)
    
    # Find highest existing log number
    existing_logs = glob.glob(os.path.join(bedrock_log_dir, "bedrock_request_*.json"))
    if existing_logs:
        try:
            last_num = max([int(os.path.basename(f).split('_')[2].split('.')[0]) for f in existing_logs])
            next_num = last_num + 1
        except (IndexError, ValueError):
             # Handle cases where filename format might be unexpected
             next_num = len(existing_logs) + 1
    else:
        next_num = 1

    # Generate filename with timestamp and incrementing number
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(bedrock_log_dir, f"bedrock_request_{next_num:04d}_{timestamp}.json")

    # Use a custom encoder to handle non-serializable objects (like potentially in messages)
    class CustomEncoder(json.JSONEncoder):
        def default(self, obj):
            # Add handling for specific non-serializable types if encountered
            return str(obj) # Basic fallback

    # Write to file
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(request_data, f, indent=2, ensure_ascii=False, cls=CustomEncoder)
        print(f"[DEBUG] Logged Bedrock request to {filename}", file=sys.stderr)
    except TypeError as e:
         print(f"[ERROR] Failed to serialize Bedrock request data for logging: {e}", file=sys.stderr)
         # Attempt to log with a simple string representation as fallback
         try:
             with open(filename.replace(".json", ".txt"), "w", encoding="utf-8") as f:
                 f.write(str(request_data))
             print(f"[DEBUG] Logged Bedrock request (fallback) to {filename.replace('.json', '.txt')}", file=sys.stderr)
         except Exception as fallback_e:
             print(f"[ERROR] Fallback Bedrock request logging failed: {fallback_e}", file=sys.stderr)
    except Exception as e:
         print(f"[ERROR] Failed to log Bedrock request to {filename}: {e}", file=sys.stderr)

    return filename


def _log_bedrock_response(response_data, log_dir, request_log_file=None):
    """Log Bedrock response data to a JSON file.
    
    Args:
        response_data: The response data (Anthropic response object or dict).
        log_dir (str): The base directory for logs.
        request_log_file: Optional path to the corresponding request log file.
    """
    import json
    import os
    import datetime
    import glob
    
    # Create bedrock subdirectory within log_dir if it doesn't exist
    bedrock_log_dir = os.path.join(log_dir, "bedrock")
    os.makedirs(bedrock_log_dir, exist_ok=True)
    
    # Generate filename with timestamp and matching request number if available
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if request_log_file:
        # Extract the request number to maintain relationship
        basename = os.path.basename(request_log_file)
        request_num = basename.split('_')[2]
        filename = os.path.join(bedrock_log_dir, f"bedrock_response_{request_num}_{timestamp}.json")
    else:
        # Find highest existing log number if no request file
        existing_logs = glob.glob(os.path.join(bedrock_log_dir, "bedrock_response_*.json"))
        if existing_logs:
            last_num = max([int(os.path.basename(f).split('_')[2].split('.')[0]) for f in existing_logs])
            next_num = last_num + 1
        else:
            next_num = 1
        filename = os.path.join(bedrock_log_dir, f"bedrock_response_{next_num:04d}_{timestamp}.json")
    
    # Convert response to a serializable format
    try:
        if hasattr(response_data, 'model_dump'):
            # Pydantic model
            response_dict = response_data.model_dump()
        elif hasattr(response_data, '__dict__'):
            # Object with attributes
            response_dict = {
                "id": getattr(response_data, 'id', None),
                "content": [
                    {
                        "type": block.type,
                        "text": getattr(block, 'text', None)
                    } for block in getattr(response_data, 'content', [])
                ],
                "model": getattr(response_data, 'model', None),
                "role": getattr(response_data, 'role', None),
                "stop_reason": getattr(response_data, 'stop_reason', None),
                "usage": {
                    "input_tokens": getattr(response_data.usage, 'input_tokens', None),
                    "output_tokens": getattr(response_data.usage, 'output_tokens', None)
                } if hasattr(response_data, 'usage') else None
            }
        else:
            # Already a dict or string
            response_dict = response_data
    except Exception as e:
        response_dict = {"error": f"Failed to serialize response: {str(e)}", "raw": str(response_data)}
    
    # Write to file
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(response_dict, f, indent=2, ensure_ascii=False)
        print(f"[DEBUG] Logged Bedrock response to {filename}", file=sys.stderr)
    except Exception as e:
        print(f"[ERROR] Failed to log Bedrock response to {filename}: {e}", file=sys.stderr)


def clear_url_validation_cache():
    """Clear the URL validation cache, forcing re-validation of all URLs."""
    global _url_validation_cache
    _url_validation_cache.clear()
    print(f"[DEBUG] URL validation cache cleared", file=sys.stderr)


def is_openai_audio_model(model: str) -> bool:
    """Checks if the OpenAI model is known to support audio modality."""
    # gpt-4o models are "omni" and handle audio. This may need to be updated as new models are released.
    return model.startswith("gpt-4o-audio")


async def prepare_openai_messages(
    messages: Optional[List[dict]],
    prompt: Optional[str],
    session: aiohttp.ClientSession,
    model: str
) -> List[dict]:
    """Prepares a list of messages for the OpenAI API, handling multimodal inputs and audio history."""
    
    supports_audio = is_openai_audio_model(model)
    if not supports_audio:
        print(f"[DEBUG] Model {model} does not support audio. Stripping audio content.", file=sys.stderr)

    # 1. Start with a clean copy of existing messages
    final_messages = [msg.copy() for msg in messages] if messages else []

    # 2. Clean audio objects in existing assistant message history
    for msg in final_messages:
        if msg.get('role') == 'assistant' and 'audio' in msg:
            if supports_audio and isinstance(msg.get('audio'), dict):
                audio_id = msg['audio'].get('id')
                if audio_id:
                    # Per OpenAI docs, only the ID should be sent back
                    msg['audio'] = {'id': audio_id}
                else:
                    # If the audio object has no ID (e.g., only data), remove it to avoid errors
                    del msg['audio']
            else:
                # Model doesn't support audio, or audio object is malformed. Remove it.
                del msg['audio']

    # 3. Process the new prompt string, if it exists
    if not prompt:
        return final_messages

    # Regex to find all supported tags
    pattern = r"(<\|(?:begin)_(img|audio)_url\|>[\s\S]*?<\|(?:end)_\2_url\|>|<\|audio_ref_id\|>[\s\S]*?<\|\/audio_ref_id\|>)"
    tokens = re.split(pattern, prompt)

    user_content_parts = []
    
    for token in tokens:
        if not token: continue

        img_audio_url_match = re.match(r"<\|begin_(img|audio)_url\|>([\s\S]*?)<\|end_\1_url\|>", token)
        audio_ref_match = re.match(r"<\|audio_ref_id\|>([\s\S]*?)<\|\/audio_ref_id\|>", token)

        if img_audio_url_match:
            tag_type, url = img_audio_url_match.groups()
            url = url.strip()
            if not url: continue
            
            if tag_type == 'img':
                user_content_parts.append({"type": "image_url", "image_url": {"url": url}})
            elif tag_type == 'audio':
                if supports_audio:
                    try:
                        print(f"[DEBUG] Processing audio URL for input: {url}", file=sys.stderr)
                        async with session.get(url) as response:
                            response.raise_for_status()
                            audio_data = await response.read()
                            content_type = response.headers.get("Content-Type", "audio/mpeg")

                        base64_audio = base64.b64encode(audio_data).decode('utf-8')
                        data_url = f"data:{content_type};base64,{base64_audio}"
                        user_content_parts.append({"type": "audio_url", "audio_url": {"url": data_url}})
                    except Exception as e:
                        print(f"[ERROR] Failed to process audio URL {url}: {e}", file=sys.stderr)
                        user_content_parts.append({"type": "text", "text": f"[Error processing audio URL: {url}]"})
                else:
                    user_content_parts.append({"type": "text", "text": "[Audio content removed: model does not support audio]"})
        elif audio_ref_match:
            if supports_audio:
                audio_id = audio_ref_match.group(1).strip()
                if audio_id:
                    # Inject a preceding assistant message to provide the audio context
                    final_messages.append({'role': 'assistant', 'content': None, 'audio': {'id': audio_id}})
            else:
                user_content_parts.append({"type": "text", "text": "[Audio reference removed: model does not support audio]"})
        else:
            # It's a regular text part
            if token.strip():
                user_content_parts.append({"type": "text", "text": strip_cache_breakpoints(token)})

    # 4. Add the final user message if it has any content parts
    if user_content_parts:
        final_messages.append({"role": "user", "content": user_content_parts})
        
    return final_messages

def convert_literal_newlines_for_openai(text: str) -> str:
    """Convert literal \\n strings to actual newlines in OpenAI response text.
    
    Sometimes OpenAI models return literal \\n instead of actual newline characters.
    This function detects and converts them to proper newlines.
     
    Args:
        text (str): The text to process
         
    Returns:
        str: Text with literal \\n converted to actual newlines
    """
    if not text:
        return text
     
    # Replace literal \n with actual newlines
    # Be careful to only replace literal \n, not already-escaped \\n
    # We'll use a simple approach: replace \n with actual newlines if it's not preceded by a backslash
    return text.replace('\\n', '\n')


def convert_messages_to_anthropic_prompt(messages: List[dict]) -> str:
    """Convert OpenAI-style messages to prompt string for OpenRouter.
    
    This handles the special case where Anthropic models on OpenRouter use the
    /chat/completions endpoint but with a 'prompt' parameter instead of 'messages'.
    
    Simply extracts and concatenates content without role formatting.
    
    Args:
        messages: List of OpenAI-style message dicts
        
    Returns:
        str: Concatenated prompt string
    """
    if not messages:
        return ""
    
    prompt_parts = []
    
    for message in messages:
        content = message.get("content", "")
        
        # Extract text content if it's in OpenAI multimodal format
        if isinstance(content, list):
            text_parts = []
            for part in content:
                if part.get("type") == "text":
                    text_parts.append(part.get("text", ""))
            content = " ".join(text_parts)
        
        if content:
            prompt_parts.append(content)
    
    prompt = " ".join(prompt_parts).strip()
    print(f"[DEBUG] Converted {len(messages)} messages to prompt: {len(prompt)} chars")
    
    return prompt
