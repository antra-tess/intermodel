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
import sys
import datetime
import requests
from mimetypes import guess_type

import aiohttp
import tenacity
from huggingface_hub.utils import LocalEntryNotFoundError

import intermodel.callgpt_faker

from dotenv import load_dotenv

from intermodel.hf_token import get_hf_tokenizer, get_hf_auth_token

load_dotenv()

MODEL_ALIASES = {}
untokenizable = set()

session: Optional[aiohttp.ClientSession] = None


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
    force_api_mode=None,
    **kwargs,
):
    message_history_format = kwargs.get("message_history_format", None)
    messages = kwargs.get("messages", None)
    name = kwargs.get("name", None)
    max_images = kwargs.get("max_images", 10)  # Default to 10 images
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

        if "openai_api_key" not in kwargs:
            kwargs["openai_api_key"] = os.getenv("OPENAI_API_KEY")
        rest = dict(kwargs)
        headers = {
            "Content-Type": "application/json",
        }
        if (api_key := rest.pop("openai_api_key", None)) is not None:
            headers["Authorization"] = f"Bearer {api_key}"
        api_base = rest.pop("api_base", "https://api.openai.com/v1")
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
        if not model.startswith("o1"):
            api_arguments["max_tokens"] = max_tokens
        # remove None values, OpenAI API doesn't like them
        for key, value in dict(api_arguments).items():
            if value is None:
                del api_arguments[key]
        # Helper function to check if force_api_mode is effectively not set
        # Helper functions for specific API modes
        def is_force_api_mode_chat(mode):
            return mode.lower() == "chat"
            
        def is_force_api_mode_completions(mode):
            return mode.lower() == "completions"
                
        print(f"[DEBUG] force_api_mode: {force_api_mode}")
        print(f"[DEBUG] is_force_api_mode_chat(force_api_mode): {is_force_api_mode_chat(force_api_mode)}")
        print(f"[DEBUG] is_force_api_mode_completions(force_api_mode): {is_force_api_mode_completions(force_api_mode)}")
        if (
            is_force_api_mode_chat(force_api_mode) or
            (not is_force_api_mode_completions(force_api_mode)) and (
                (message_history_format is not None and message_history_format.is_chat()) or
                model.startswith("gpt-3.5") or
                model.startswith("gpt-4") or
                model.startswith("o1") or
                model.startswith("openpipe:") or
                model.startswith("gpt4") or
                model.startswith("chatgpt-4o") or
                model.startswith("grok") or
                model.startswith("deepseek-reasoner") or
                model.startswith("deepseek/deepseek-r1") or
                model.startswith("deepseek-ai/DeepSeek-R1-Zero") or
                model.startswith("aion") or
                model.startswith("google/gemma-3-27b-it") or
                model.startswith("DeepHermes-3-Mistral-24B-Preview") or
                api_base.startswith("https://integrate.api.nvidia.com")
            )
        ) and not model.endswith("-base"):
        
            if messages is None:
                if (
                    message_history_format is not None
                    and message_history_format.is_chat() 
                ):
                    api_arguments["messages"] = message_history_format.format_messages(
                        api_arguments["prompt"], "user"
                    )
                    print(f"[DEBUG] used format_messages, message count: {len(api_arguments['messages'])}")
                else:
                    api_arguments["messages"] = [
                        {
                            "role": "system",
                            "content": f"Respond to the chat, where your username is shown as {name}. Only respond with the content of your message, without including your username.",
                        },
                        {"role": "user", "content": api_arguments["prompt"]},
                    ]
                    print(f"[DEBUG] chat history sent as a single user message")
            else:
                api_arguments["messages"] = messages
                print(f"[DEBUG] messages sent as is, message count: {len(api_arguments['messages'])}")
            if "prompt" in api_arguments:
                del api_arguments["prompt"]
            if "logprobs" in api_arguments:
                del api_arguments["logprobs"]
            if model.startswith("o1") or model.startswith("deepseek") or api_base.startswith("https://integrate.api.nvidia.com") or model.startswith("aion"):
                if "logit_bias" in api_arguments:
                    del api_arguments["logit_bias"]
                if (
                    model.startswith("o1")
                    or model.startswith("chatgpt-4o")
                    or model.startswith("deepseek-reasoner")
                    or model.startswith("deepseek/deepseek-r1")
                    or model.startswith("deepseek-ai/DeepSeek-R1-Zero")
                    or model.startswith("grok")
                    or model.startswith("aion")
                ):
                    if api_base.startswith("https://openrouter.ai"):
                        reasoning_content_key = "reasoning"
                        api_arguments["include_reasoning"] = True
                    else:
                        reasoning_content_key = "reasoning_content"
            # Remove empty logit_bias for NVIDIA endpoints
            api_suffix = "/chat/completions"
        else:                
            api_suffix = "/completions"
            
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
                                text_parts.append(item.get("text", ""))
                        content = " ".join(text_parts)
                    
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
            print(f"[DEBUG] message count: {len(api_arguments['messages'])}")

        async with session.post(
            api_base + api_suffix, headers=headers, json=api_arguments
        ) as response:
            api_response = await response.json()
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
            return {
                "prompt": {"text": prompt if prompt is not None else "<|endoftext|>"},
                "completions": [
                    {
                        "text": (
                            completion["text"].strip()
                            if api_suffix == "/completions"
                            else completion["message"]["content"].strip()
                        ),
                        "finish_reason": {
                            "reason": completion.get("finish_reason", "unknown")
                        },
                        "reasoning_content": (
                            completion["message"].get(reasoning_content_key, None)
                            if reasoning_content_key is not None
                            else None
                        ),
                    }
                    for completion in api_response["choices"]
                ],
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
                    "prompt": prompt,
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
                    "inputs": prompt,
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

        # Handle thinking parameter for Claude 3
        if model.startswith("claude-3") and "thinking" in kwargs:
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
                messages = messages + process_image_messages(prompt, max_images=max_images)

        if vendor == "anthropic-steering-preview":
            kwargs["extra_headers"] = {"anthropic-beta": "steering-2024-06-04"}
            if "steering" in kwargs:
                kwargs["extra_body"] = {"steering": kwargs["steering"]}
                del kwargs["steering"]

        # Create a deep copy of the request to log
        request_payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens or 16,
            "temperature": temperature or 1,
            "top_p": top_p if 'thinking' not in kwargs else None,
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
        with open("anthropic_request.json", "w") as f:
            f.write(json.dumps(request_payload, indent=2, cls=CustomEncoder))
        #print(json.dumps(request_payload, indent=2, cls=CustomEncoder), file=sys.stderr)

        if 'thinking' in kwargs:
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
                temperature=temperature or 1,
                top_p=top_p or 1,
                stop_sequences=stop or list(),
                **kwargs,
            )
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
            "id": uuid.uuid4(),
            "created": datetime.datetime.now(),
            "usage": {
                "charged_tokens": 0,
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
        
        if messages is not None:
            print(f"[DEBUG] Converting {len(messages)} messages to Gemini format", file=sys.stderr)
            
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                
                # Create a content object with parts for each message
                content_parts = []
                
                # Handle string content
                if isinstance(content, str):
                    # Check if this message contains image URLs
                    if "<|begin_of_img_url|>" in content and "<|end_of_img_url|>" in content:
                        print(f"[DEBUG] Found image URLs in message, processing", file=sys.stderr)
                        
                        # Parse out the text and image URLs
                        sections = re.split(r"<\|(?:begin|end)_of_img_url\|>", content)
                        
                        # Extract text parts and image URLs
                        text_parts = []
                        image_urls = []
                        
                        for i, section in enumerate(sections):
                            if i % 2 == 0:  # Text section
                                if section.strip():
                                    text_parts.append(section.strip())
                            else:  # Image URL
                                image_urls.append(section.strip())
                        
                        # Respect max_images parameter
                        total_images = len(image_urls)
                        images_to_process = min(total_images, max_images)
                        
                        print(f"[DEBUG] Found {total_images} images, will process {images_to_process}", file=sys.stderr)
                        
                        if images_to_process < total_images:
                            # Only keep the last images_to_process images
                            image_urls = image_urls[-images_to_process:]
                        
                        # Combine text parts into a single prompt for this message
                        text_prompt = " ".join(text_parts)
                        
                        # Add text part first if it exists
                        if text_prompt:
                            content_parts.append(types.Part(text=text_prompt))
                        
                        # Then add the image parts
                        for url in image_urls:
                            try:
                                print(f"[DEBUG] Processing image URL: {url[:100]}{'...' if len(url) > 100 else ''}", file=sys.stderr)
                                
                                # Download the image
                                response = requests.get(url)
                                response.raise_for_status()
                                image_data = response.content
                                
                                mime_type = (
                                    response.headers.get("content-type")
                                    or guess_type(url)[0]
                                    or "image/jpeg"
                                )
                                
                                # Skip GIFs for Gemini models
                                if mime_type.lower() == "image/gif":
                                    print(f"[DEBUG] Skipping GIF image as it's not supported by Gemini", file=sys.stderr)
                                    continue
                                
                                # Create a Part object with the image for Gemini
                                image_part = types.Part(
                                    inline_data=types.Blob(
                                        mime_type=mime_type,
                                        data=image_data
                                    )
                                )
                                content_parts.append(image_part)
                                print(f"[DEBUG] Added image to message ({len(image_data)} bytes)", file=sys.stderr)
                            except Exception as e:
                                print(f"[DEBUG] Failed to process image: {str(e)}", file=sys.stderr)
                                continue
                    else:
                        # No images, just text
                        content_parts.append(types.Part(text=content))
                
                # Create a Content object with the parts and role
                if content_parts:
                    # Map roles to Gemini's expected format (only 'user' or 'model')
                    gemini_role = role
                    if role == 'assistant':
                        gemini_role = 'model'
                    elif role not in ['user', 'model']:
                        # Treat system and any other roles as user by default
                        gemini_role = 'user'
                    
                    gemini_contents.append(types.Content(parts=content_parts, role=gemini_role))
        else:
            # If no messages format was provided, use the prompt directly
            gemini_contents = [types.Content(parts=[types.Part(text=prompt)], role="user")]
            
        print(f"[DEBUG] Sending request with {len(gemini_contents)} content objects", file=sys.stderr)
        
        # Debug log last three messages
        print("[DEBUG] Last three messages being sent:", file=sys.stderr)
        for i, content in enumerate(gemini_contents[-3:]):
            parts_info = []
            for part in content.parts:
                if part.text is not None:
                    if len(part.text) > 300:
                        parts_info.append(f"Text: '{part.text[:150]}...{part.text[-150:]}'")
                    else:
                        parts_info.append(f"Text: '{part.text}'")
                elif part.inline_data is not None:
                    parts_info.append(f"Image: {len(part.inline_data.data)} bytes, type: {part.inline_data.mime_type}")
            
            print(f"[DEBUG] Message {i+1}: Role={content.role}, Parts={parts_info}", file=sys.stderr)
        
        # Configure the request based on the model type
        is_image_generation = model == "gemini-2.0-flash-exp-image-generation"
        
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
                request_log_file = _log_gemini_request(request_data)
                
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
                _log_gemini_response(response, request_log_file)
                
                # Process the response for image generation
                text_content = ""
                image_data = None

                # iterate through all response fields and print the


                if not response.candidates or len(response.candidates) == 0:
                    print(f"[DEBUG] No candidates returned from Gemini", file=sys.stderr)
                    print(f"[DEBUG] Response: {response}", file=sys.stderr)
                    raise Exception("No candidates returned from Gemini")
                

                for candidate in response.candidates:
                    for part in candidate.content.parts:
                        if part.text is not None:
                            text_content = part.text
                            print(f"[DEBUG] Received text response: {text_content[:100]}{'...' if len(text_content) > 100 else ''}", file=sys.stderr)
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
                        "max_output_tokens": max_tokens or 2048,
                        "stop_sequences": stop or [],
                        "safety_settings": "BLOCK_NONE for all categories"
                    }
                }
                
                # Log the request
                request_log_file = _log_gemini_request(request_data)
                
                # For regular text models
                response = client.models.generate_content(
                    model=model,
                    contents=gemini_contents,
                    config=types.GenerateContentConfig(
                        temperature=temperature or 1.0,
                        top_p=top_p or 1.0,
                        top_k=top_k or 40,
                        max_output_tokens=max_tokens or 2048,
                        stop_sequences=stop or [],
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
                _log_gemini_response(response, request_log_file)

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


def download_and_encode_image(url, skip_gifs=False):
    """Download image and convert to base64.
    
    Args:
        url (str): URL of the image to download
        skip_gifs (bool): If True, skip GIF images (for Gemini models)
    """
    import base64
    import requests
    from mimetypes import guess_type
    import sys

    print(f"[DEBUG] Downloading image from: {url[:100]}{'...' if len(url) > 100 else ''}", file=sys.stderr)
    response = requests.get(url)
    response.raise_for_status()
    image_data = response.content
    mime_type = (
        response.headers.get("content-type")
        or guess_type(url)[0]
        or "application/octet-stream"
    )
    
    # Skip GIFs if requested (for Gemini models)
    if skip_gifs and mime_type.lower() == "image/gif":
        print(f"[DEBUG] Skipping GIF image as it's not supported", file=sys.stderr)
        return None, None
        
    print(f"[DEBUG] Downloaded image: {len(image_data)} bytes, mime type: {mime_type}", file=sys.stderr)
    return base64.b64encode(image_data).decode(), mime_type


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


def process_image_messages(prompt: str, prompt_role: str = "user", max_images: int = 10, skip_gifs: bool = False) -> list:
    """Convert a prompt containing image URLs into a messages array.

    Args:
        prompt (str): The input prompt text with image URL markers
        prompt_role (str): The role to use for text messages (default: "user")
        max_images (int): Maximum number of images to process (default: 10)
        skip_gifs (bool): If True, skip GIF images (for Gemini models)

    Returns:
        list: Array of message objects with text and images
    """
    import requests
    import sys
    import base64
    from google.genai import types

    print(f"[DEBUG] Processing image messages with role: {prompt_role}", file=sys.stderr)
    
    messages = []
    sections = re.split(r"<\|(?:begin|end)_of_img_url\|>", prompt)

    # Constants
    total_images = (len(sections) - 1) // 2
    images_to_process = min(total_images, max_images)
    
    print(f"[DEBUG] Found {total_images} images, will process {images_to_process} (max limit: {max_images})", file=sys.stderr)
    
    image_counter = 0
    current_msg_parts = []
    
    # Process each section
    for i, section in enumerate(sections):
        if i % 2 == 0:  # Text section
            if section.strip():
                current_msg_parts.append(types.Part(text=section.strip()))
        else:  # Image URL section
            image_counter += 1
            if total_images - image_counter < images_to_process:
                try:
                    print(f"[DEBUG] Processing image URL: {section[:100]}{'...' if len(section) > 100 else ''}", file=sys.stderr)
                    result = download_and_encode_image(section, skip_gifs=skip_gifs)
                    if result[0] is None:  # Skip if image was filtered out
                        continue
                    base64_data, mime_type = result
                    image_part = types.Part(
                        inline_data=types.Blob(
                            mime_type=mime_type,
                            data=base64.b64decode(base64_data)
                        )
                    )
                    current_msg_parts.append(image_part)
                    print(f"[DEBUG] Added image to message parts", file=sys.stderr)
                except requests.RequestException as e:
                    print(f"[DEBUG] Failed to download image: {str(e)}", file=sys.stderr)
                    continue  # Skip failed image downloads
            else:
                # For images beyond the processing limit, add a text note
                url_note = f"[Image URL (exceeding limit): {section[:30]}...]"
                current_msg_parts.append(types.Part(text=url_note))
                print(f"[DEBUG] Skipped processing image due to limit: {section[:50]}...", file=sys.stderr)
    
    # Create a Content object with all parts
    if current_msg_parts:
        # Map roles to Gemini's expected format (only 'user' or 'model')
        gemini_role = prompt_role
        if prompt_role == 'assistant':
            gemini_role = 'model'
        elif prompt_role not in ['user', 'model']:
            # Treat system and any other roles as user by default
            gemini_role = 'user'
            
        messages.append(types.Content(parts=current_msg_parts, role=gemini_role))

    print(f"[DEBUG] Finished processing image messages, created {len(messages)} messages", file=sys.stderr)
    return messages


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
    if vendor == "openai" or model == "gpt2" or model.startswith("claude-3") or model.startswith("chatgpt-4o") or model.startswith("grok") or model.startswith("aion") or model.startswith("DeepHermes") or model.startswith("google/gemma-3") or model.startswith("gemini-"):
        # tiktoken internally caches loaded tokenizers
        if model.startswith("claude-3"):
            tokenizer = tiktoken.encoding_for_model("gpt2")
        elif model.startswith("o1"):
            tokenizer = tiktoken.encoding_for_model("gpt-4o")
        elif model.startswith("chatgpt-4o"):
            tokenizer = tiktoken.encoding_for_model("gpt-4o")
        elif model.startswith("gpt-4.5-preview"):
            tokenizer = tiktoken.encoding_for_model("gpt-4o")
        elif model.startswith("grok"):
            tokenizer = tiktoken.encoding_for_model("gpt2")
        elif model.startswith("aion"):
            tokenizer = tiktoken.encoding_for_model("gpt2")
        elif model.startswith("google/gemma-3-27b-it"):
            tokenizer = tiktoken.encoding_for_model("gpt2")
        elif model.startswith("DeepHermes-3-Mistral-24B-Preview"):
            tokenizer = tiktoken.encoding_for_model("gpt2")
        elif model.startswith("gemini-"):
            tokenizer = tiktoken.encoding_for_model("gpt2")  # Use GPT-2 tokenizer as approximation
        else:
            tokenizer = tiktoken.encoding_for_model(model)
        # encode special tokens as normal
        # XXX: make this an option
        return tokenizer.encode(string, allowed_special="all")
    elif vendor == "anthropic":
        # all anthropic tokenzers are private now
        return tokenize("gpt2", string)
    elif model in untokenizable:
        return tokenize("gpt2", string)
    else:
        try:
            tokenizer = get_hf_tokenizer(model)
        except Exception as e:
            message = (
                f"Failed to download tokenizer by looking up {model} as a huggingface model ID."
                "To override the tokenizer, put the correct huggingface ID ^ after the model name and follow it with a max token cap."
                "For example, to use OpenRouter Gemini while using the Gemma tokenizer with a maximum context window of 2 million Gemma tokens, use `google/gemini-pro-1.5^google/gemma-7b@2000000`"
                "Using the incorrect tokenizer may lead to flakiness and intermittent failures. Some proprietary models have openly available tokenizers."
            )
            if e.__class__.__name__ == "GatedRepoError":
                warnings.warn(
                    message
                    + f" Cause: GatedRepoError. Log in with huggingface-cli and accept the model ToS at https://huggingface.co/{model} to access the model tokenizer"
                )
            elif e.__class__.__name__ == "RepositoryNotFoundError":
                warnings.warn(message + f" Cause: Not found on huggingface")
            else:
                warnings.warn(message)
            return tokenize("gpt2", string)
        else:
            return tokenizer.encode(string).ids


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
        or model.startswith("gpt-3.5-")
        or model.startswith("o1-")
    ):
        return "openai"
    elif "j1-" in model or model.startswith("j2-"):
        return "ai21"
    elif "forefront" in model:
        return "forefront"
    elif model.startswith("claude-"):
        return "anthropic"
    elif model.startswith("aion"):
        return "openai"  # aion models use OpenAI-compatible API
    elif model.startswith("google/"):
        return "openai"  # Google models go through OpenRouter
    elif model.startswith("gemini-"):
        return "gemini"  # Gemini models go directly
    elif "/" in model:
        return "huggingface"
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
    elif model.startswith("o1"):
        return 128_000
    elif model == "gpt-4.5-preview":
        return 128_000  # gpt-4.5-preview has a 128k context window
    elif model.startswith("gpt-4"):
        return 8193
    elif model.startswith("chatgpt-4o"):
        return 128_000
    elif model.startswith("grok"):
        return 128_000
    elif model.startswith("aion"):
        return 30_000  # 32k context window
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
    elif model.startswith("claude-3"):
        return 200_000 * 0.7
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
    elif model == "DeepHermes-3-Mistral-24B-Preview":
        return 31000
    elif model.startswith("gemini-"):
        if model == "gemini-2.0-flash-exp-image-generation":
            return 127000  # Image generation model has shorter context
        return 127000  # Standard Gemini models have 32k context
    else:
        try:
            import huggingface_hub

            try:
                fname = huggingface_hub.hf_hub_download(
                    model,
                    "config.json",
                    token=get_hf_auth_token(),
                    local_files_only=True,
                )
            except LocalEntryNotFoundError:
                fname = huggingface_hub.hf_hub_download(
                    model,
                    "config.json",
                    token=get_hf_auth_token(),
                    local_files_only=False,
                )

            with open(fname, "r") as f:
                import json

                data = json.load(f)
            return data["max_position_embeddings"] + 1
        except Exception as e:
            raise NotImplementedError(
                f"Unable to download {model} from HuggingFace. "
                f"Are you logged into HuggingFace (`huggingface-cli login`) and have you agreed to the model license at"
                f"`https://huggingface.co/{model}`?"
            )


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


def _log_gemini_request(request_data):
    """Log Gemini request data to a JSON file.
    
    Args:
        request_data (dict): The request data to log
    """
    import json
    import os
    import datetime
    import glob
    
    # Create gemini directory if it doesn't exist
    log_dir = os.path.join(os.getcwd(), "gemini")
    os.makedirs(log_dir, exist_ok=True)
    
    # Find highest existing log number
    existing_logs = glob.glob(os.path.join(log_dir, "gemini_request_*.json"))
    if existing_logs:
        last_num = max([int(os.path.basename(f).split('_')[2].split('.')[0]) for f in existing_logs])
        next_num = last_num + 1
    else:
        next_num = 1
        
    # Generate filename with timestamp and incrementing number
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(log_dir, f"gemini_request_{next_num:04d}_{timestamp}.json")
    
    # Process content to preserve text but shorten image data
    processed_data = {}
    
    # Deep copy dictionary with special handling for content objects
    def process_content(content_list):
        processed_contents = []
        for content in content_list:
            processed_content = {"role": content.role, "parts": []}
            
            for part in content.parts:
                if part.text is not None:
                    # Include full text
                    processed_content["parts"].append({"type": "text", "text": part.text})
                elif part.inline_data is not None:
                    # Shorten image data but keep metadata
                    processed_content["parts"].append({
                        "type": "inline_data",
                        "mime_type": part.inline_data.mime_type,
                        "data_size_bytes": len(part.inline_data.data),
                        "data_preview": "(binary data, truncated)" 
                    })
            
            processed_contents.append(processed_content)
        return processed_contents
    
    # Process main request data
    if "model" in request_data:
        processed_data["model"] = request_data["model"]
    
    if "contents" in request_data:
        processed_data["contents"] = process_content(request_data["contents"])
    
    if "config" in request_data:
        # Convert config to dict with special handling for nested objects
        config_dict = {}
        if hasattr(request_data["config"], "temperature"):
            config_dict["temperature"] = request_data["config"].temperature
        if hasattr(request_data["config"], "top_p"):
            config_dict["top_p"] = request_data["config"].top_p
        if hasattr(request_data["config"], "top_k"):
            config_dict["top_k"] = request_data["config"].top_k
        if hasattr(request_data["config"], "max_output_tokens"):
            config_dict["max_output_tokens"] = request_data["config"].max_output_tokens
        if hasattr(request_data["config"], "stop_sequences"):
            config_dict["stop_sequences"] = request_data["config"].stop_sequences
        
        processed_data["config"] = config_dict
    
    # Write to file
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, indent=2, ensure_ascii=False)
        
    print(f"[DEBUG] Logged Gemini request to {filename}", file=sys.stderr)
    return filename


def _log_gemini_response(response, request_log_file=None):
    """Log Gemini response data to a JSON file.
    
    Args:
        response: The Gemini API response to log
        request_log_file: Optional path to the corresponding request log file
    """
    import json
    import os
    import datetime
    import glob
    
    # Create gemini directory if it doesn't exist
    log_dir = os.path.join(os.getcwd(), "gemini")
    os.makedirs(log_dir, exist_ok=True)
    
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
    
    # Check if response has candidates
    if hasattr(response, "candidates") and response.candidates:
        processed_data["candidates"] = []
        
        for candidate in response.candidates:
            candidate_data = {}
            
            if hasattr(candidate, "content") and candidate.content:
                content_data = {"parts": []}
                
                # Get role if available
                if hasattr(candidate.content, "role"):
                    content_data["role"] = candidate.content.role
                
                # Process the parts
                if hasattr(candidate.content, "parts"):
                    for part in candidate.content.parts:
                        part_data = {}
                        
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
                
            processed_data["candidates"].append(candidate_data)
    
    # Add simple text response if present
    if hasattr(response, "text"):
        processed_data["text"] = response.text
    
    # Write to file
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, indent=2, ensure_ascii=False)
        
    print(f"[DEBUG] Logged Gemini response to {filename}", file=sys.stderr)
    return filename


if __name__ == "__main__":
    InteractiveIntermodel().cmdloop()
