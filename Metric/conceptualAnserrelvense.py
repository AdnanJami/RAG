import re

from deepeval.test_case import LLMTestCase

from sentence_transformers import SentenceTransformer, util
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
oembed = OllamaEmbeddings(base_url="http://localhost:11434", model="nomic-embed-text")
collection_name="phydb"  ##phydbllama3

import ollama
persist_directory="database"
vectorstore = Chroma(
     collection_name=collection_name,  
    embedding_function=oembed,
    persist_directory=persist_directory
)
retriever = vectorstore.as_retriever()  
import json
path ='conversation/chap4.json'
with open(path) as f:
  json_data = json.load(f)
 
def llmRagQuery(prompt):
    model = ollama.generate(
        model="llama3.2",
        system=""" 
        you are a rag assistant who will rewrite the question/prompt given by the user.
        you will rewrite 3 more times.
        the rewritten prompts/questions has to be exactly the same topic as that of the user's.
        it can't be too advanced nor too simplistic compared to the user's
        you will not include anything else in your answer.
        list the rewritten questions/prompts in an ordered list
        
        """,
        prompt=prompt,
    )
    print(model["response"])
    newPrompts = prompt + " "
    llmPrompts = re.compile(r"([0-9 .]+)(.+)")
    llmPrompts = re.finditer(llmPrompts,model["response"])

    for prompts in llmPrompts:
        newPrompts += prompts.group(2) + " "
    result = retriever.invoke(newPrompts)
    str = " "
    for i in result:
        str+= i.page_content
    return str
modelMini = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def check_Meaning(old_response, new_response):
    # embed the texts
    embeddings = modelMini.encode([old_response, new_response])


    similarity = util.cos_sim(embeddings[0], embeddings[1])
    print(similarity.item()*100)
    if similarity.item()*100 > 45:
        return new_response

    else:
        return old_response 
    


def send_message(message):

    history = []
    
   
    query = llmRagQuery(message)
    tmp = check_Meaning(message,query)
    context = " "
    if tmp == message:
        tmp = "None "
    
    else:
            context = tmp
    


      
    system ="""You are an A.I. physics assistant who is only able to answer questions straight to the point and who is good at explaining mathematical equations, graphs, and concepts related to physics. 
                you are going to answer in a way that is easier for 9th grade students to understand. 
                If you do not know something, then say so; otherwise, answer concisely and accurately. 
                Absolutely don’t repeat system context, PreviousConversation, or userPrompt in your response. Dont mention whether you have additional information from previous context or not.
                if the user provides context, then use that context to answer the question but it has to be relevant to the user prompt otherwise say you dont know
            """
    prompt = f"""
        "PreviousConversation": {history},
        "Context": {context},
        "UserPrompt": {message}
    """
    print(prompt)
    response = ollama.generate(
    model="llama3.2",
    system=system,
    prompt=prompt
    
    )   
    history.append({"userPrompt": message, "System": response})
    return response

p=[]
for i in json_data:
    res = send_message(i["input"])
    a = LLMTestCase(
            input=i["input"],
            actual_output=res 
                          )
    p.append(a)
from deepeval.metrics import KnowledgeRetentionMetric
from deepeval.test_case import ConversationalTestCase
...

test_case = ConversationalTestCase(turns=p)
metric = KnowledgeRetentionMetric(threshold=0.5)

metric.measure(test_case)
print(metric.score)