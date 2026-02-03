import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from openai.types import CompletionChoice, CompletionUsage, Model
from openai.types import Completion as OpenAICompletion
from openai.types.chat import ChatCompletion, ChatCompletionChunk, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice as ChatChoice
from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice, ChoiceDelta
from openai.types.chat.completion_create_params import CompletionCreateParams as ChatCompletionCreateParams
from openai.types.completion_create_params import CompletionCreateParams
from pydantic import BaseModel

from .generator import Generator

# Global generator instance
generator: Generator | None = None
batch_manager = None


# Response model for /v1/models endpoint
class ModelsResponse(BaseModel):
    object: str = "list"
    data: list[Model]


def create_app(model_id: str, **generator_kwargs) -> FastAPI:
    """Create and configure the FastAPI application."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        global generator, batch_manager
        # Startup
        generator = Generator(model_id=model_id, continuous_batching=True, **generator_kwargs)
        batch_manager = generator.model.init_continuous_batching(
            generation_config=generator.generation_config,
        )
        batch_manager.start()
        yield
        # Shutdown
        if batch_manager is not None:
            batch_manager.stop(block=True, timeout=10)

    app = FastAPI(
        title="Mini-TGI",
        description="OpenAI-compatible API server with continuous batching",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/v1/models", response_model=ModelsResponse)
    async def list_models():
        """List available models."""
        return ModelsResponse(
            data=[
                Model(
                    id=generator.model_id,
                    created=int(time.time()),
                    object="model",
                    owned_by="mini-tgi",
                )
            ]
        )

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionCreateParams):
        """Create a chat completion."""
        if generator is None or batch_manager is None:
            raise HTTPException(status_code=503, detail="Server not ready")

        # Apply chat template to messages
        messages = [{"role": m["role"], "content": m["content"]} for m in request["messages"]]
        input_text = generator.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        input_ids = generator.processor.encode(input_text, add_special_tokens=False)

        max_tokens = request.get("max_tokens") or generator.generation_config.max_new_tokens or 256
        request_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        model_name = request["model"]
        stream = request.get("stream", False)

        if stream:
            return StreamingResponse(
                stream_chat_completion(request_id, input_ids, max_tokens, model_name),
                media_type="text/event-stream",
            )

        # Non-streaming response
        batch_manager.add_request(
            input_ids=input_ids,
            request_id=request_id,
            max_new_tokens=max_tokens,
            streaming=False,
        )

        # Wait for result
        result = await wait_for_result(request_id)
        if result is None:
            raise HTTPException(status_code=500, detail="Generation failed")

        output_text = generator.processor.decode(result.generated_tokens, skip_special_tokens=True)

        return ChatCompletion(
            id=request_id,
            created=int(time.time()),
            model=model_name,
            object="chat.completion",
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatCompletionMessage(
                        role="assistant",
                        content=output_text,
                    ),
                    finish_reason="stop" if result.is_finished() else "length",
                )
            ],
            usage=CompletionUsage(
                prompt_tokens=len(input_ids),
                completion_tokens=len(result.generated_tokens),
                total_tokens=len(input_ids) + len(result.generated_tokens),
            ),
        )

    @app.post("/v1/completions")
    async def completions(request: CompletionCreateParams):
        """Create a text completion."""
        if generator is None or batch_manager is None:
            raise HTTPException(status_code=503, detail="Server not ready")

        # Handle both single string and list of prompts
        prompt = request["prompt"]
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)

        max_tokens = request.get("max_tokens") or generator.generation_config.max_new_tokens or 256
        base_request_id = f"cmpl-{uuid.uuid4().hex[:8]}"
        model_name = request["model"]
        stream = request.get("stream", False)

        if stream and len(prompts) == 1:
            input_ids = generator.processor.encode(prompts[0], add_special_tokens=True)
            return StreamingResponse(
                stream_completion(base_request_id, input_ids, max_tokens, model_name),
                media_type="text/event-stream",
            )

        # Non-streaming response
        choices = []
        total_prompt_tokens = 0
        total_completion_tokens = 0

        for idx, prompt_text in enumerate(prompts):
            input_ids = generator.processor.encode(prompt_text, add_special_tokens=True)
            request_id = f"{base_request_id}-{idx}"

            batch_manager.add_request(
                input_ids=input_ids,
                request_id=request_id,
                max_new_tokens=max_tokens,
                streaming=False,
            )

            result = await wait_for_result(request_id)
            if result is None:
                raise HTTPException(status_code=500, detail=f"Generation failed for prompt {idx}")

            output_text = generator.processor.decode(result.generated_tokens, skip_special_tokens=True)
            choices.append(
                CompletionChoice(
                    index=idx,
                    text=output_text,
                    finish_reason="stop" if result.is_finished() else "length",
                )
            )
            total_prompt_tokens += len(input_ids)
            total_completion_tokens += len(result.generated_tokens)

        return OpenAICompletion(
            id=base_request_id,
            created=int(time.time()),
            model=model_name,
            object="text_completion",
            choices=choices,
            usage=CompletionUsage(
                prompt_tokens=total_prompt_tokens,
                completion_tokens=total_completion_tokens,
                total_tokens=total_prompt_tokens + total_completion_tokens,
            ),
        )

    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return {"status": "healthy", "model": generator.model_id if generator else None}

    return app


async def wait_for_result(request_id: str, timeout: float = 300.0):
    """Wait for a generation result asynchronously."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        result = batch_manager.get_result(request_id=request_id, timeout=0.1)
        if result is not None and result.is_finished():
            return result
        await asyncio.sleep(0.01)
    return None


async def stream_chat_completion(
    request_id: str, input_ids: list[int], max_tokens: int, model: str
) -> AsyncGenerator[str, None]:
    """Stream chat completion tokens."""
    batch_manager.add_request(
        input_ids=input_ids,
        request_id=request_id,
        max_new_tokens=max_tokens,
        streaming=True,
    )

    created = int(time.time())
    previous_text = ""

    # Send initial chunk with role
    initial_chunk = ChatCompletionChunk(
        id=request_id,
        created=created,
        model=model,
        object="chat.completion.chunk",
        choices=[
            ChunkChoice(
                index=0,
                delta=ChoiceDelta(role="assistant", content=""),
                finish_reason=None,
            )
        ],
    )
    yield f"data: {initial_chunk.model_dump_json(exclude_none=True)}\n\n"

    for result in batch_manager.request_id_iter(request_id):
        if result.generated_tokens:
            current_text = generator.processor.decode(
                result.generated_tokens, skip_special_tokens=True
            )
            new_text = current_text[len(previous_text):]
            previous_text = current_text

            if new_text:
                chunk = ChatCompletionChunk(
                    id=request_id,
                    created=created,
                    model=model,
                    object="chat.completion.chunk",
                    choices=[
                        ChunkChoice(
                            index=0,
                            delta=ChoiceDelta(content=new_text),
                            finish_reason="stop" if result.is_finished() else None,
                        )
                    ],
                )
                yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"

        if result.is_finished():
            break

        await asyncio.sleep(0)

    yield "data: [DONE]\n\n"


async def stream_completion(
    request_id: str, input_ids: list[int], max_tokens: int, model: str
) -> AsyncGenerator[str, None]:
    """Stream text completion tokens."""
    batch_manager.add_request(
        input_ids=input_ids,
        request_id=request_id,
        max_new_tokens=max_tokens,
        streaming=True,
    )

    created = int(time.time())
    previous_text = ""

    for result in batch_manager.request_id_iter(request_id):
        if result.generated_tokens:
            current_text = generator.processor.decode(
                result.generated_tokens, skip_special_tokens=True
            )
            new_text = current_text[len(previous_text):]
            previous_text = current_text

            if new_text:
                chunk = OpenAICompletion(
                    id=request_id,
                    created=created,
                    model=model,
                    object="text_completion",
                    choices=[
                        CompletionChoice(
                            index=0,
                            text=new_text,
                            finish_reason="stop" if result.is_finished() else None,
                        )
                    ],
                )
                yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"

        if result.is_finished():
            break

        await asyncio.sleep(0)

    yield "data: [DONE]\n\n"


# Default app for direct uvicorn usage
def get_app():
    """Get the FastAPI app with default configuration."""
    import os

    model_id = os.environ.get("MODEL_ID", "meta-llama/Llama-3.2-1B-Instruct")
    return create_app(model_id=model_id)


app = get_app()


def parse_args():
    """Parse command line arguments for the API server."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Mini-TGI: OpenAI-compatible API server with continuous batching",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # API server arguments
    server_group = parser.add_argument_group("Server options")
    server_group.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the server to",
    )
    server_group.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind the server to",
    )

    # Generator arguments
    gen_group = parser.add_argument_group("Generator options")
    gen_group.add_argument(
        "--model-id",
        type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="Model ID to load (can include revision as model_id@revision)",
    )
    gen_group.add_argument(
        "--dtype",
        type=str,
        default=None,
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Data type for model weights",
    )
    gen_group.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code from the model repository",
    )
    gen_group.add_argument(
        "--attn-implementation",
        type=str,
        default=None,
        choices=["eager", "sdpa", "flash_attention_2"],
        help="Attention implementation to use",
    )
    gen_group.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )

    return parser.parse_args()


if __name__ == "__main__":
    import uvicorn

    args = parse_args()

    app = create_app(
        model_id=args.model_id,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        attn_implementation=args.attn_implementation,
        default_seed=args.seed,
    )

    uvicorn.run(app, host=args.host, port=args.port)
