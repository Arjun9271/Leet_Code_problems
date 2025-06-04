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





    0: client error (Connect)
    1: TLS handshake failed: cert verification failed - self signed certificate in certificate chain [CERTIFICATE_VERIFY_FAILED]
    2: [CERTIFICATE_VERIFY_FAILED]
    3: [CERTIFICATE_VERIFY_FAILED]
2025-06-04 16:51:32 - Error to search using lite backend: https://lite.duckduckgo.com/lite/ RuntimeError: error sending request for url (https://lite.duckduckgo.com/lite/): client error (Connect)

Caused by:
    0: client error (Connect)
    1: TLS handshake failed: cert verification failed - self signed certificate in certificate chain [CERTIFICATE_VERIFY_FAILED]
    2: [CERTIFICATE_VERIFY_FAILED]
    3: [CERTIFICATE_VERIFY_FAILED]
2025-06-04 16:51:32 - https://lite.duckduckgo.com/lite/ RuntimeError: error sending request for url (https://lite.duckduckgo.com/lite/): client error (Connect)

Caused by:
    0: client error (Connect)
    1: TLS handshake failed: cert verification failed - self signed certificate in certificate chain [CERTIFICATE_VERIFY_FAILED]
    2: [CERTIFICATE_VERIFY_FAILED]
    3: [CERTIFICATE_VERIFY_FAILED]
Traceback (most recent call last):
  File "/home/thatha.arjun/Desktop/chainlit_101/myenv/lib/python3.10/site-packages/chainlit/utils.py", line 47, in wrapper
    return await user_function(**params_values)
  File "/home/thatha.arjun/Desktop/chainlit_101/myenv/lib/python3.10/site-packages/chainlit/callbacks.py", line 169, in with_parent_id
    await func(message)
  File "/home/thatha.arjun/Desktop/chainlit_101/main.py", line 42, in on_message
    web_result = search.run(message.content)
  File "/home/thatha.arjun/Desktop/chainlit_101/myenv/lib/python3.10/site-packages/langchain_core/tools/base.py", line 774, in run
    raise error_to_raise
  File "/home/thatha.arjun/Desktop/chainlit_101/myenv/lib/python3.10/site-packages/langchain_core/tools/base.py", line 743, in run
    response = context.run(self._run, *tool_args, **tool_kwargs)
  File "/home/thatha.arjun/Desktop/chainlit_101/myenv/lib/python3.10/site-packages/langchain_community/tools/ddg_search/tool.py", line 74, in _run
    return self.api_wrapper.run(query)
  File "/home/thatha.arjun/Desktop/chainlit_101/myenv/lib/python3.10/site-packages/langchain_community/utilities/duckduckgo_search.py", line 114, in run
    results = self._ddgs_text(query)
  File "/home/thatha.arjun/Desktop/chainlit_101/myenv/lib/python3.10/site-packages/langchain_community/utilities/duckduckgo_search.py", line 64, in _ddgs_text
    ddgs_gen = ddgs.text(
  File "/home/thatha.arjun/Desktop/chainlit_101/myenv/lib/python3.10/site-packages/duckduckgo_search/duckduckgo_search.py", line 185, in text
    raise DuckDuckGoSearchException(err)
duckduckgo_search.exceptions.DuckDuckGoSearchException: https://lite.duckduckgo.com/lite/ RuntimeError: error sending request for url (https://lite.duckduckgo.com/lite/): client error (Connect)

Caused by:
    0: client error (Connect)
    1: TLS handshake failed: cert verification failed - self signed certificate in certificate chain [CERTIFICATE_VERIFY_FAILED]
    2: [CERTIFICATE_VERIFY_FAILED]
