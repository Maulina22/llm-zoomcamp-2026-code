"""Shared LLM client for the course.

Uses Gemini through its OpenAI-compatible endpoint, so all the course code
that expects the `openai` library works unchanged.
"""

import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL = "gemini-3.6-flash"

client = OpenAI(
    api_key=os.environ["GEMINI_API_KEY"],
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)


def ask(prompt: str, model: str = MODEL) -> str:
    """Send a single prompt and return the text response."""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    print(ask("Reply with exactly: setup works"))
