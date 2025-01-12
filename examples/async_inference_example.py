import sys
import os
import asyncio
from openai import OpenAI
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.config import ConfigManager


async def main():
    # Get configuration
    config = ConfigManager()
    api_base = f"http://{config.get('server.host', 'localhost')}:{config.get('server.port', 8000)}/v1"
    model_id = config.get('model.checkpoint', 'mistralai/Mixtral-8x7B-Instruct-v0.1')

    # Initialize OpenAI client
    client = OpenAI(
        api_key="EMPTY",  # Not needed for local server
        base_url=api_base
    )

    # Test prompts
    prompts = [
        "What is your favourite condiment?",
        "What are the benefits of a healthy diet?",
        # ...more prompts...
    ]

    # Process each prompt
    for prompt in prompts:
        try:
            chat_completion = client.chat.completions.create(
                messages=[{
                    "role": "user",
                    "content": prompt
                }],
                model=model_id,
                max_tokens=50,
                temperature=0.7
            )
            
            result = chat_completion.choices[0].message.content
            print(f"Prompt: {prompt}\nGenerated Text: {result}\n")
            
        except Exception as e:
            print(f"Error processing prompt '{prompt}': {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
