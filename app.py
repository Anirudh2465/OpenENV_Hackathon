import os
import sys

# Retrieve tokens from the environment variables securely
# In Hugging Face Spaces, set these in the Space's Settings -> Secrets
hf_token = os.environ.get("HF_TOKEN")
google_key = os.environ.get("GOOGLE_API_KEY")

if google_key:
    os.environ["GEMINI_API_KEY"] = google_key

# Import the UI builder and CSS from our ui/app module
from ui.app import build_ui, CSS

# Build the Gradio interface
demo = build_ui()

# Launch the app. Hugging Face Spaces typically use port 7860 on 0.0.0.0
# The server keeps running and keeping the code live.
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, css=CSS)
