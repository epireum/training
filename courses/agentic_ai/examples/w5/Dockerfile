FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY school_agent/ ./school_agent/
COPY student_info_agent/ ./student_info_agent/
COPY student_marks_agent/ ./student_marks_agent/

EXPOSE 8000 8001 7001

