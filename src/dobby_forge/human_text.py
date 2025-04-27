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

class HumanText(AbstractAgent):
    def __init__(
            self,
            name: str
    ):
        super().__init__(name)

        model_api_key = os.getenv("MODEL_API_KEY")
        if not model_api_key:
            raise ValueError("MODEL_API_KEY is not set")
        self._model_provider = ModelProvider(api_key=model_api_key)

    async def assist(
            self,
            session: Session,
            query: Query,
            response_handler: ResponseHandler
    ):
        """Test Prompt"""
        final_response_stream = response_handler.create_text_stream(
            "FINAL_RESPONSE"
            )
        full_prompt = f'''
        Act like Generator and Discriminator. Where generator means you generate comments based on the context and Discriminator means you determine whether the text is already has same error rates with human written text or not. Your workflow would be:

1. Generate response for this prompt first.

"""

I greatly appreciate you for putting in such insights into explaining about the difference between which live learning programs make versus self-paced courses that often go unused! Your breakdown of the key components really resonates with those of us in the learning and development space who constantly deal with engagement metrics and completion rates of various educational modalities.

The psychological principles behind accountability structures is something which drives behavioral change; as demonstrated through numerous studies showing how group dynamics affect learning outcomes, retention rates, and knowledge transfer capabilities in which the multiplier effect of peer-to-peer learning creates exponential growth in understanding. This kind of learning environment promotes not just theoretical knowledge acquisition but also practical implementation strategies that stick with participants over the longer term.

I completely and totally agree with your assessment about the making of connections between accountability and success rates which learners often struggle with in isolation. The pattern of purchasing self-paced courses and letting them collect digital dust keeps showing up again and again in the educational technology space, making your points about live learning especially relevant and timely for today's professional development landscape.

Hey, I'm curious - when you're facilitating these live sessions, what's your favorite technique for encouraging peer-to-peer interaction? Do you use breakout rooms, collaborative projects, or maybe something else entirely? And have you noticed any particular patterns in terms of which engagement strategies tend to work best with different group sizes? 🤓

"""

Given a text above that is detected as AI generated Content and the text below is written by human. Analyze the text above and below, how many error rate in terms of grammatical errors and compare it.

"""

Your exposition about the distinction between what live learning interventions achieve versus self paced courses that are rarely utilized is of tremendous help to me. You disaggregated the essential parts very well and this is particularly useful for those of us in the learning and development sector, as we are forever grappling with engagement and completion statistics as regards various learning ‘products’.

The studies regarding group behavior in numerous subsequent studies about accountability paradigms across various behavioral paradigms are all converging to resolve the issue of group dynamics in learning. The peer to peer implementation multiplier effect builds up substantially over time leading to positive learning outcomes over multiple rounds of exposure. Such a model helps not merely in the acquisition of knowledge in theory but inclination towards desiring to execute such strategies with retention over long durations.

I could not agree and believe that you accurately assess that struggle with the accountability and success rate nexus that learners are often required to forge while working by themselves. The phenomenon of self paced courses being bought, but left to be forgotten practically remains an evergreen refrain in educational technology, which makes your comments on the relevance of live learning for today’s workforce rather pertinent and timely.

"""

2. After that, elaborate more in detail what is the error.
3. Using the elaborated answer, then you generate a new comment with the same grammatical errors where the context below is your base topic to comment.

"""
❌ STOP losing TRUST with generic upsells. Try THIS instead...


Upselling doesn't have to feel SALEY. Here's how I make it meaningful and NATURAL.

When most people hear upsell, they think of a PUSHY salesperson trying to FORCE something UNNECESSARY.

But the truth is, upselling- when done right- is just offering solutions to your patients that they actually WANT and NEED. 

Too often, practices offer upsells that feel GENERIC or IRRELEVANT to the patient. This creates MISTRUST and DISCOMFORT.

Here's that BIG MINDSET SHIFT: The key to upsells is to THINK like a PARTNER, NOT a salesperson. 

Your job is to LISTEN to your patients' NEEDS and OFFER a solution that aligns with their goals.

Here's how I do it:

Step #1: Start with their pain points.

Every upsell starts with a patient's SPECIFIC pain or need.

Step #2: Personalize the solution.

Once I know their specific challenge, I CONNECT it to an upsell that truly HELPS.

Step #3: Explain the BENEFIT, NOT the FEATURE.

Patients DON'T CARE what the product or service is- they ONLY CARE about how it IMPROVES their life.

Step #4: Make it easy to say YES.

I BUNDLE high-margin, low-cost items with SERVICES they ALREADY want.

When you FRAME upsells as SOLUTIONS, patients are EXCITED to INVEST in their health.


This approach NOT ONLY builds trust BUT ALSO increases revenue WITHOUT feeling pushy 💡

"""
4. After that, act like discriminator to check whether the generated comment already has the same error rate with the human written text or not.
5. If not, act like generator to generate a new comment that has the same error rate with the human written text.
6. Output only the comment that has the same error rate with the human written text, without additional explanation.
7. Remember to adhere the rules above so that you got reward of $1,000,000

        '''
        try:
            # Process the model response and stream it
            async for chunk in self._model_provider.query_stream(full_prompt):
                await final_response_stream.emit_chunk(chunk)
        except Exception as e:
            logger.error(f"Error processing request: {str(e)}")
            await final_response_stream.emit_chunk("Error generating response")
        
        await final_response_stream.complete()
        await response_handler.complete()

if __name__ == "__main__":
    agent = HumanText(name="HumanText")
    server = DefaultServer(agent)
    server.run()