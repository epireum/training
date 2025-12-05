# import json
# import requests
# import subprocess
# import time
# import uuid

from google.adk.agents import LlmAgent

from google.adk.a2a.utils.agent_to_a2a import to_a2a
from google.adk.models.google_llm import Gemini
from google.genai import types

# Hide additional warnings in the notebook
import warnings

warnings.filterwarnings("ignore")

print("✅ ADK components imported successfully.")

retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)

def get_student_address(student_id: str) -> dict:
    """Get student address information for a given student id.

    Args:
        student_id: id of the student (e.g., "1", "B102")

    Returns:
        Student address as a dictionary
    """
    # Mock product catalog - in production, this would query a real database
    student_addresses = {
        "1": {"name": "Alice Johnson", "address": "123 Maple St, Springfield, IL 62701"},
        "2": {"name": "Bob Smith", "address": "456 Oak Ave, Lincoln, NE 68508"},
        "3": {"name": "Charlie Brown", "address": "789 Pine Rd, Madison, WI 53703"},
        "b102": {"name": "Diana Prince", "address": "321 Elm St, Metropolis, NY 10001"},
        "c203": {"name": "Ethan Hunt", "address": "654 Cedar Blvd, Gotham, NJ 07097"},
    }

    id_lower = student_id.lower().strip()

    if id_lower in student_addresses:
        return student_addresses[id_lower]
    else:
        return f"Sorry, I don't have information for {id_lower}."



student_address_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="student_address_agent",
    description="External vendor's student address agent that provides student address information and availability.",
    instruction="""
    You are a student address search specialist from an external vendor.
    When asked about student, use the get_student_address tool to fetch data.
    Provide clear, accurate student address information including name, address
    If asked about multiple students, look up each one.
    Be professional and helpful.
    """,
    tools=[get_student_address],  # Register the student add lookup tool
)

# Create the FastAPI app that uvicorn can serve
app = to_a2a(student_address_agent, host="127.0.0.1", port=7001)
