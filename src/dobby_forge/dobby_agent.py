import logging
import os
from dotenv import load_dotenv
from sentient_agent_framework import (
    AbstractAgent,
    DefaultServer,
    Session,
    Query,
    ResponseHandler
)
from src.dobby_forge.providers.model_provider import ModelProvider
from typing import AsyncIterator

load_dotenv()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DobbyAgent(AbstractAgent):
    def __init__(self, name: str):
        super().__init__(name)

        model_api_key = os.getenv("MODEL_API_KEY")
        if not model_api_key:
            raise ValueError("MODEL_API_KEY is not set")
        self._model_provider = ModelProvider(api_key=model_api_key)

        self.persona = "rebellious startup founder"
        self.style = "sarcastic and witty"
        self.loyalty = "startups and innovation"
        self.temperature = 0.7
        self.top_p = 0.9
        self.max_tokens = 256

    async def assist(
        self,
        session: Session,
        query: Query,
        response_handler: ResponseHandler
    ):
        """Generate a sarcastic, witty social media post promoting startups and innovation."""
        await response_handler.emit_text_block(
            "GENERATE",
            "Cooking up a brutally honest startup post with extra wit..."
        )

        prompt = (query.prompt or "").strip()
        full_prompt = (
            f"Persona: {self.persona}\n"
            f"Style: {self.style}\n"
            f"Loyalty: {self.loyalty}\n\n"
            f"Task: Write a witty and sarcastic social media post (≤280 characters) "
            f"that supports startups and innovation, based on the idea: \"{prompt}\"."
        )

        stream = response_handler.create_text_stream("FINAL_RESPONSE")
        try:
            # Process the model response and stream it
            async for chunk in self._model_provider.query_stream(full_prompt):
                await stream.emit_chunk(chunk)
        except Exception as e:
            logger.error(f"Error processing request: {str(e)}")
            await stream.emit_chunk("Error generating response")
        
        await stream.complete()
        await response_handler.complete()

if __name__ == "__main__":
    agent = DobbyAgent(name="Dobby Agent")
    server = DefaultServer(agent)
    server.run()
