<role>: act as ai engineer  whose having very good knowledge on evaluation and selection of the llms

<context>:we need to evaluate these models in terms of the better response on a larger context 
model 1: llama 3.1 8b
model2: llama 3.2 3b

use the hugging inference i will provide example code:

develop on top of that:-)

import warnings
warnings.filterwarnings("ignore")

from huggingface_hub import InferenceClient

client = InferenceClient(
    provider="novita",
    api_key= api_key,
)

stream = client.chat.completions.create(
    model="meta-llama/Llama-3.2-3B-Instruct",
   messages=[
        {
            "role": "user",
            "content": [

                {
                    "type": "text",
                    "text": """
                                1.what is E-signed date mentioned in doc ?
                                2. who was the director ?
                                3. the variation agreement was dated on ?
                                4. where the british telecom plc has been registered ?
                                5.whats the year3 qualifying period ?
                                6. whats the email of yan arnauld mentioned in doc ?

                    """
                },
                  {
                    "type":"text",
                    "text": content_1
                }

            ]
        }
   ],
    stream=True,
)

for chunk in stream:
  if chunk.choices and chunk.choices[0].delta.content is not None:
    print(chunk.choices[0].delta.content, end="")

model:

from huggingface_hub import InferenceClient  client = InferenceClient(     provider="fireworks-ai",     api_key=api_key, )  # data_string = json.dumps(data)  stream = client.chat.completions.create(     model="meta-llama/Llama-3.1-8B-Instruct",     messages=[         {             "role": "user",             "content": [                  {                     "type": "text",                     "text": """                                  1.what is E-signed date mentioned in doc ?                                 2. who was the director ?                                 3. the variation agreement was dated on ?                                 4. where the british telecom plc has been registered ?                                 5.whats the year3 qualifying period ?                      """                 },                   {                     "type":"text",                     "text": content_1                 }              ]         }     ],     stream=True, )  for chunk in stream:    if chunk.choices and chunk.choices[0].delta.content is not None:     print(chunk.choices[0].delta.content, end="")

note : dont consider the examples that i given,you frame on your own

