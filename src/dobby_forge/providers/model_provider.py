from datetime import datetime
from langchain_core.prompts import PromptTemplate
from openai import AsyncOpenAI
from typing import AsyncIterator, Optional
import logging

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class ModelProvider:
    def __init__(
        self,
        api_key: str,
        model: Optional[str] = "accounts/sentientfoundation/models/dobby-unhinged-llama-3-3-70b-new",
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = "default",
        base_url: Optional[str] = "https://api.fireworks.ai/inference/v1"
    ):
        """ Initializes model, sets up OpenAI client, configures system prompt. """
        
        # Assign API Key and initialize attributes
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self.date_context = datetime.now().strftime("%Y-%m-%d")

        # Set up OpenAI client
        self.client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )

        # Configure system prompt
        self._configure_system_prompt()

    def _configure_system_prompt(self):
        """ Configures the system prompt, allowing for dynamic or custom templates. """
        if self.system_prompt == "default":
            system_prompt_template = PromptTemplate(
                input_variables=["date_today"],
                template=(
                    "You are a helpful assistant that can answer questions and provide information. "
                    "Today’s date is: {date_today}. Keep responses clear, concise, and helpful."
                ),
            )
            self.system_prompt = system_prompt_template.format(date_today=self.date_context)
        else:
            # If the user has provided a custom system prompt, use it
            self.system_prompt = self.system_prompt

    async def query_stream(
        self,
        query: str,
        temperature: Optional[float] = None,  # Accept temperature as an optional argument
        max_tokens: Optional[int] = None,  # Accept max_tokens as an optional argument
        top_p: Optional[float] = None  # Accept top_p as an optional argument
    ) -> AsyncIterator[str]:
        """Sends query to model and yields the response in chunks."""
        
        # Use the provided temperature, max_tokens, and top_p, or default to the instance values
        temperature = temperature or self.temperature
        max_tokens = max_tokens or self.max_tokens
        top_p = top_p or 1.0  # Default value for top_p if not provided
        
        # Define message format based on the model type
        messages = self._prepare_messages(query)
        
        try:
            # Send query to OpenAI and stream the response
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
                temperature=temperature,  # Pass the temperature here
                max_tokens=max_tokens,    # Pass the max_tokens here
                top_p=top_p,              # Pass the top_p here
            )

            async for chunk in stream:
                # Yield the response chunk by chunk
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            # Log errors and re-raise
            logger.error(f"Error while streaming query: {e}")
            raise e


    async def query(
        self,
        query: str
    ) -> str:
        """Sends query to model and returns the complete response as a string."""
        
        try:
            # Collect all chunks into a full response
            chunks = []
            async for chunk in self.query_stream(query=query):
                chunks.append(chunk)
            response = "".join(chunks)
            return response
        except Exception as e:
            logger.error(f"Error while processing query: {e}")
            return "An error occurred while processing your request."

    def _prepare_messages(self, query: str) -> list:
        """Prepares the appropriate message format for the model based on the model type."""
        
        # If using smaller models, use specific message formats
        if self.model in ["o1-preview", "o1-mini"]:
            return [
                {"role": "user", "content": f"System Instruction: {self.system_prompt}\n Instruction: {query}"}
            ]
        else:
            return [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": query}
            ]
