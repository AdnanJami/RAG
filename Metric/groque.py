from groq import Groq
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from collections import deque
oembed = OllamaEmbeddings(base_url="http://localhost:11434", model="nomic-embed-text")
collection_name="phydb"  ##phydbllama3
    
persist_directory="database"
vectorstore = Chroma(
     collection_name=collection_name,  
    embedding_function=oembed,
    persist_directory=persist_directory
)
API_KEY="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"
client = Groq(api_key=API_KEY)
def generate_response(chunks, query, context_history=None, model="llama-3.2-3b-preview"):
    context = " ".join([chunk["text"] for chunk in chunks])
    if context_history:
        context = " ".join(context_history) + " " + context
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": f"Context: {context} Query: {query}"}],
        model=model,
        stream=False,
    )
    return response.choices[0].message.content
def maintain_conversational_context(response, context_history, max_context_length=10):
    if len(context_history) >= max_context_length:
        context_history.popleft()
    context_history.append(response)
    return context_history
def rag_pipeline( question, context_history):
   
    result = vectorstore.similarity_search(question)
    str = " "
    for i in result:
        str+= i.page_content

    # Step 4: Generate response
    response = generate_response(str, question, context_history)
    return response
def interactive_cli(data_folder="data/"):
    context_history = deque()
    print("Welcome to the RAG-powered conversational assistant! Type /bye to exit.")

    while True:
        user_input = input(">> user: ")
        if user_input.lower() == "/bye":
            print("Goodbye!")
            break

        response = rag_pipeline(data_folder, user_input, context_history)
        context_history = maintain_conversational_context(response, context_history)

        print(f">> groq: {response}")
if __name__ == "__main__":
    interactive_cli()