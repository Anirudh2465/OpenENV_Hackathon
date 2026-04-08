FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# HuggingFace Spaces port
ENV PORT=7860
EXPOSE 7860

# Default: launch Gradio UI with rule-based agent (no API key needed)
CMD ["python", "-m", "ui.app", "--port", "7860"]