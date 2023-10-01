import os
import chainlit as cl
from chainlit.prompt import Prompt
from langchain.llms import HuggingFaceHub
from langchain import LLMChain, PromptTemplate, OpenAI
from chainlit.playground.providers.langchain import LangchainGenericProvider
from chainlit.playground.config import add_llm_provider
from dotenv import load_dotenv
import os
import openai

load_dotenv()
# huggingface_api_token = os.getenv("HUGGINGFACE_API_TOKEN")

# llm = HuggingFaceHub(
#     model_kwargs={"max_length": 500},
#     repo_id="nsivaku/ccp_predictor",
#     huggingfacehub_api_token=huggingface_api_token,
# )
# add_llm_provider(
#     LangchainGenericProvider(
#         # It is important that the id of the provider matches the _llm_type
#         id=llm._llm_type,
#         # The name is not important. It will be displayed in the UI.
#         name="HuggingFaceHub",
#         # This should always be a Langchain llm instance (correctly configured)
#         llm=llm,
#         # If the LLM works with messages, set this to True
#         is_chat=True
#     )
# )

# get openAI key
openai.api_key = os.getenv("OPENAI_API")

template = """Would {song} be played in the club?
"""
@cl.on_message
async def main(message: str):
    prompt = PromptTemplate(template=template, input_variables = ["song"])
    llm_chain = LLMChain(prompt = prompt,llm=OpenAI(temperature=0,streaming=True),verbose=True)
    cl.user_session.set("llm_chain",llm_chain)

@cl.on_message
async def main(message : str):
    llm_chain = cl.user_session.get("llm_chain")

    res = await llm_chain.acall(message, callbacks=[cl.AsyncLangchainCallbackHandler()])

    await cl.Message(content=res["text"]).send()