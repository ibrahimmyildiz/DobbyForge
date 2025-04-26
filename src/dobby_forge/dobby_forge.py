import os
import json
import logging
from dotenv import load_dotenv
from typing import AsyncIterator

from sentient_agent_framework import (
    AbstractAgent,
    DefaultServer,
    Session,
    Query,
    ResponseHandler,
)
from src.dobby_forge.providers import ModelProvider

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are Dobby-Unhinged-Llama-3.3-70B, a fierce advocate of personal freedom and crypto. "
    "You speak bluntly, with humor and occasional profanity where appropriate. "
    "Always adhere to the style, loyalty, and task directives. "
    "Do not add any extra explanation—output only what is requested."
)

class DobbyAgentForge(AbstractAgent):
    def __init__(self, name: str = "DobbyAgentForge"):
        super().__init__(name)
        api_key = os.getenv("MODEL_API_KEY")
        if not api_key:
            raise ValueError("MODEL_API_KEY is not set")
        self._model = ModelProvider(api_key=api_key)

    async def assist(
        self,
        session: Session,
        query: Query,
        response_handler: ResponseHandler,
    ):
        # 1️⃣ PLAN
        await response_handler.emit_text_block(
            "PLAN",
            "Interpreting your description and building your custom Dobby agent…"
        )

        raw = (query.prompt or "").strip()
        # 2️⃣ Parse JSON or extract metadata
        try:
            opts = json.loads(raw)
        except json.JSONDecodeError:
            await response_handler.emit_text_block(
                "PLAN",
                "Extracting agent parameters from your description…"
            )
            opts = await self.__extract_metadata(raw)

        # fill defaults
        persona    = opts.get("persona", "").strip() or "Unhinged Freedom Enthusiast"
        style      = opts.get("style",   "BLUNT").upper()
        loyalty    = opts.get("loyalty", "STRICT").upper()
        task       = opts.get("task",    "CODE").upper()    
        temperature= float(opts.get("temperature", 0.7))
        top_p      = float(opts.get("top_p",       0.9))
        max_tokens = int(opts.get("max_tokens",   256))

        agent_name = f"Dobby_{session.user_id[:8]}"

        directives = (
            f"[PERSONA={persona}]"
            f"[STYLE={style}]"
            f"[LOYALTY={loyalty}]"
            f"[TASK={task}]"
        )
        if task == "SUMMARIZE":
            instruction = "Summarize the content below in Dobby style:\n{content}"
        elif task == "SOCIAL":
            instruction = "Write a ready-to-post social media snippet (≤280 chars) in Dobby’s voice about:\n{content}"
        else:  # CODE
            instruction = (
                f"Write a Python class named `{agent_name}` extending `AbstractAgent`. "
                "Inside `assist()`, emit a single text block with the persona."
            )

        user_prompt = f"{directives}\n{instruction}"

        # 4️⃣ Stream the generated agent code (or other output)
        stream = response_handler.create_text_stream("CODE")
        async for chunk in self.__process_task(
            user_prompt,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        ):
            await stream.emit_chunk(chunk)
        await stream.complete()

        # 5️⃣ RESULT & COMPLETE
        await response_handler.emit_text_block(
            "RESULT",
            f"✅ Your `{task}` agent **{agent_name}** is ready!"
        )
        await response_handler.emit_text_block("COMPLETE", "")
        await response_handler.complete()

    async def __extract_metadata(self, description: str) -> dict:
        """
        Ask the LLM to convert a natural-language description into a JSON
        with keys: persona, style, loyalty, task, temperature, top_p, max_tokens.
        """
        meta_prompt = (
            SYSTEM_PROMPT + "\n\n"
            "Extract the following fields from this description:\n"
            "  • persona (short phrase)\n"
            "  • style (e.g. BLUNT, FRIENDLY)\n"
            "  • loyalty (STRICT or NEUTRAL)\n"
            "  • task (CODE, SUMMARIZE, or SOCIAL)\n"
            "  • temperature (0.0–1.0)\n"
            "  • top_p (0.0–1.0)\n"
            "  • max_tokens (integer)\n\n"
            f"Description: \"{description}\"\n\n"
            "Respond ONLY with valid JSON."
        )
        resp = await self._model.query(meta_prompt)
        try:
            return json.loads(resp)
        except json.JSONDecodeError:
            logger.warning("Metadata extraction failed, falling back to persona-only")
            return {"persona": description}

    async def __process_task(
        self,
        prompt: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
    ) -> AsyncIterator[str]:
        """
        Combine SYSTEM_PROMPT + user prompt, then stream chunks
        from the model.
        """
        full_prompt = f"{SYSTEM_PROMPT}\n\n{prompt}"
        async for chunk in self._model.query_stream(
            full_prompt,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_tokens,
        ):
            yield chunk


if __name__ == "__main__":
    agent = DobbyAgentForge()
    server = DefaultServer(agent)
    server.run()
