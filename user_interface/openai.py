import chainlit as cl
import openai
import os
import dotenv
dotenv.load_dotenv()

openai.api_key = os.getenv("OPENAI_API")


@cl.on_message
async def main(message: str):
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[{"role": "assistant", 'content': "You judge wether or not a song would be played in the club, by answering yes or no"},
                  {"role": "user", "content": message}],
        temperature=1)
    await cl.Message(
        content=f"{response['choices'][0]['message']['content']}").send()
