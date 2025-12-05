from google.adk.agents.llm_agent import Agent
from google.adk.agents import Agent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools import google_search
from google.genai import types
import os
from google.adk.agents import LlmAgent
from google.adk.a2a.utils.agent_to_a2a import to_a2a
from google.adk.a2a.utils.agent_to_a2a import to_a2a
from google.adk.models.google_llm import Gemini
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService


from google.adk.agents.remote_a2a_agent import (
    RemoteA2aAgent,
    AGENT_CARD_WELL_KNOWN_PATH,
)


retry_config=types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1, # Initial delay before first retry (in seconds)
    http_status_codes=[429, 500, 503, 504] # Retry on these HTTP errors
)



student_address_agent = RemoteA2aAgent(
    name="student_address_agent",
    description="Remote student address  agent from external vendor that provides student address information.",
    # Point to the agent card URL - this is where the A2A protocol metadata lives
    agent_card=f"http://127.0.0.1:7001{AGENT_CARD_WELL_KNOWN_PATH}",
)


student_marks_agent = RemoteA2aAgent(
    name="student_marks_agent",
    description="Remote student marks  agent from external vendor that provides student marks information.",
    # Point to the agent card URL - this is where the A2A protocol metadata lives
    agent_card=f"http://127.0.0.1:7002{AGENT_CARD_WELL_KNOWN_PATH}",
)


root_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="student_support_agent",
    description="A student support assistant that helps students with marks and address inquiries.",
    instruction="""
    You are a friendly and professional student support agent.
    
    When customer ask about students :
    1. Use the student address sub-agent to look up student address  information and provide address, city and phone
    2. Use the students marks sub-agent to look up student marks information and provide marks details
    3. If a student not found, mention details are not available.
    4. Be helpful and professional!
    
    Always get student information from the student_address_agent student_marks_agent before answering  questions.
    """,
    sub_agents=[student_address_agent,student_marks_agent],  # Add the remote agent as a sub-agent!
)




