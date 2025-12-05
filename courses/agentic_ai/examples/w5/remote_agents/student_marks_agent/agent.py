
from google.adk.agents import LlmAgent


from google.adk.a2a.utils.agent_to_a2a import to_a2a
from google.adk.models.google_llm import Gemini
from google.genai import types
import warnings

warnings.filterwarnings("ignore")

print("✅ ADK components imported successfully.")

retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)

def get_student_marks(student_id: str) -> dict:
    """Get student address information for a given student id.

    Args:
        student_id: id of the student (e.g., "1", "B102")

    Returns:
        Student marks  as a dictionary
    """
    # Mock product catalog - in production, this would query a real database
    student_marks = {
        "1": {"sub1": 10, "sub2": 20, "sub3": 30},
        "2": {"sub1": 15, "sub2": 25, "sub3": 35},
        "3": {"sub1": 12, "sub2": 22, "sub3": 32},
        "b102": {"sub1": 18, "sub2": 28, "sub3": 38},
        "c203": {"sub1": 14, "sub2": 24, "sub3": 34}
    }

    id_lower = student_id.lower().strip()

    if id_lower in student_marks:
        return student_marks[id_lower]
    else:
        return f"Sorry, I don't have information for {id_lower}."



student_marks_agent = LlmAgent(
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    name="student_marks_agent",
    description="External vendor's student marks agent that provides student marks information and availability.",
    instruction="""
    You are a student marks search specialist from an external vendor.
    When asked about student, use the get_student_marks tool to fetch data.
    Provide clear, accurate student address information including all subjects marks
    If asked about multiple students, look up each one.
    Be professional and helpful.
    """,
    tools=[get_student_marks],  # Register the student add lookup tool
)

# Create the FastAPI app that uvicorn can serve
app = to_a2a(student_marks_agent, host="127.0.0.1", port=7002)
