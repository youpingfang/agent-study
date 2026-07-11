import asyncio
import os
from google import genai


client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

interaction = client.interactions.create(
    model="gemini-3.5-flash",
    input="你可以用来做什么"
)
print(interaction.output_text)
