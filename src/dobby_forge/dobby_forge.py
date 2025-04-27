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

from src.dobby_forge.providers.model_provider import ModelProvider

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
        response_handler: ResponseHandler
    ):
        """Main method to process the user query and generate agent output."""
        final_response_stream = response_handler.create_text_stream("FINAL_RESPONSE")

        raw_prompt = (query.prompt or "").strip()

        # Step 1️⃣: Parse the input
        try:
            opts = json.loads(raw_prompt)
            logger.info("Parsed structured JSON input successfully.")
        except json.JSONDecodeError:
            logger.info("Natural language input detected. Extracting metadata...")
            opts = await self.__extract_metadata(raw_prompt)

        # Step 2️⃣: Fill defaults if missing
        persona     = opts.get("persona", "Unhinged Freedom Enthusiast").strip()
        style       = opts.get("style", "BLUNT").upper()
        loyalty     = opts.get("loyalty", "STRICT").upper()
        task        = opts.get("task", "CODE").upper()
        temperature = float(opts.get("temperature", 0.7))
        top_p       = float(opts.get("top_p", 0.9))
        max_tokens  = int(opts.get("max_tokens", 256))

        directives = (
            f"[PERSONA={persona}]"
            f"[STYLE={style}]"
            f"[LOYALTY={loyalty}]"
            f"[TASK={task}]"
        )

        # Step 3️⃣: Build dynamic instruction
        if task == "CODE":
            instruction = (
                "Write a Python class named `DobbyAgents` extending `AbstractAgent`. "
                "Inside `assist()`, emit a single text block that reflects the assigned persona's voice."
            )
        elif task == "SUMMARIZE":
            instruction = "Summarize the content below in Dobby's style:\n{content}"
        elif task == "SOCIAL":
            instruction = (
                "Write a ready-to-post social media snippet (≤280 chars) in Dobby’s voice about:\n{content}"
            )
        else:
            logger.warning(f"Unknown task '{task}' provided. Defaulting to CODE generation.")
            instruction = (
                "Write a Python class named `DobbyAgents` extending `AbstractAgent`. "
                "Inside `assist()`, emit a single text block that reflects the assigned persona's voice."
            )

        # Step 4️⃣: Combine system prompt, directives, and instruction
        full_prompt = f"{SYSTEM_PROMPT}\n\n{directives}\n\n{instruction}"

        logger.info("Prompt construction complete. Starting streaming response...")

        try:
            async for chunk in self._model.query_stream(
                full_prompt,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            ):
                await final_response_stream.emit_chunk(chunk)
        except Exception as e:
            logger.error(f"Error generating agent output: {str(e)}")
            await final_response_stream.emit_chunk("Error generating response.")

        await final_response_stream.complete()
        await response_handler.complete()

    async def __extract_metadata(self, description: str) -> dict:
        """
        Ask the LLM to infer metadata from a natural language description.
        """
        meta_prompt = (
            SYSTEM_PROMPT + "\n\n"
            "Extract the following fields from this description:\n"
            "  • persona (short phrase)\n"
            "  • style (e.g., BLUNT, FRIENDLY)\n"
            "  • loyalty (STRICT or NEUTRAL)\n"
            "  • task (CODE, SUMMARIZE, or SOCIAL)\n"
            "  • temperature (0.0–1.0)\n"
            "  • top_p (0.0–1.0)\n"
            "  • max_tokens (integer)\n\n"
            f"Description: \"{description}\"\n\n"
            "Respond ONLY with valid JSON."
        )
        try:
            resp = await self._model.query(meta_prompt)
            metadata = json.loads(resp)
            logger.info(f"Metadata extracted: {metadata}")
            return metadata
        except Exception as e:
            logger.warning(f"Metadata extraction failed, fallback to persona-only. Error: {str(e)}")
            return {"persona": description}


if __name__ == "__main__":
    agent = DobbyAgentForge()
    server = DefaultServer(agent)
    server.run()
