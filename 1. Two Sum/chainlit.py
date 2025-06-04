from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain_community.tools import DuckDuckGoSearchRun

import chainlit as cl


google_key = ''.replace('l','I')



search = DuckDuckGoSearchRun()



@cl.on_chat_start
async def on_chat_start():
    model = ChatGoogleGenerativeAI(api_key=google_key, model="gemini-2.0-flash-lite", temperature=1)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", '''You're a helpful assistant,provides the relevant answers to the user queries
             add the relevant icons and emojis.'''),
            ("human", "{question}"),
        ]
    )
    cl.user_session.set("model", model)
    cl.user_session.set("prompt", prompt)


@cl.on_message
async def on_message(message: cl.Message):
    model = cl.user_session.get("model")
    prompt = cl.user_session.get("prompt")

    search_keywords =  ['current','latest','news','trending','today','recent','updates','information','events']
    needs_search = any(keyword in message.content.lower() for keyword in search_keywords)
    
    chain = prompt | model | StrOutputParser()
    response = await chain.ainvoke({"question": message.content})
        
    
    
    msg = cl.Message(content=response)
    await msg.send()
