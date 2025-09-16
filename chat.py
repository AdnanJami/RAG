
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from langchain_core.messages import AIMessage, HumanMessage
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from flask import Flask, request, jsonify, render_template
import os
from flask_mysqldb import MySQL
from flask import redirect, url_for
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
oembed = OllamaEmbeddings(base_url="http://localhost:11434", model="nomic-embed-text")
collection_name="phydb"  ##phydbllama3
llm = Ollama(model="llama3")
 
persist_directory="database"
vectorstore = Chroma(
     collection_name=collection_name,  
    embedding_function=oembed,
    persist_directory=persist_directory
)
retriever = vectorstore.as_retriever()   



app = Flask(__name__, template_folder="template")

# MySQL Configuration
app.config['MYSQL_HOST'] = 'localhost'  # or your MySQL server
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'chatbot'


def getAll(query, params=None):
    cursor = mysql.connection.cursor()
    if params:
        cursor.execute(query, params)  # Execute with parameters
    else:
        cursor.execute(query)  # Execute without parameters
    results = cursor.fetchall()
    cursor.close()
    return results

def insert(query, values):
    cursor = mysql.connection.cursor()
    cursor.execute(query, values)  # Pass the query and the values separately
    mysql.connection.commit()
    cursor.close()
    return "success"

def delete(query, params):
    cursor = mysql.connection.cursor()
    try:
        cursor.execute(query, params)  # Execute query with params
        mysql.connection.commit()      # Commit the changes
    except Exception as e:
        print(f"Database error: {e}")
        mysql.connection.rollback()    # Rollback in case of error
        raise e
    finally:
        cursor.close()

def rename(query, val):
        try:
            cursor = mysql.connection.cursor()
            
            cursor.execute(query,val )  # Safely execute with parameters
            mysql.connection.commit()  # Commit changes to the database
        except Exception as e:
            print(f"Error renaming session: {e}")
            mysql.connection.rollback()  # Rollback if there's an error
            return jsonify({"error": "Error renaming session"}), 500
        finally:
            cursor.close()


mysql = MySQL(app)

@app.route("/home", methods=["GET"])
def index():
    
    sessions = getAll("SELECT id, name FROM sessions")
    return render_template("index.html", sessions=sessions)




@app.route('/new_chat', methods=['POST'])
def new_chat():
    chatname = request.json.get("chatname")

    if not chatname:
        return jsonify({"error": "Chat name is required"}), 400

    query = "INSERT INTO sessions (`name`) VALUES (%s)"
    
    try:
        insert(query, (chatname,))  # Pass query and values separately
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"success": True}), 200




@app.route('/delete_chat', methods=['POST'])
def delete_chat():
    session_id = request.json.get('session_id')
    if session_id is None:
        return jsonify({"error": "Session ID not provided"}), 400  # Handle missing session ID

    try:
        delete_query = "DELETE FROM sessions WHERE id = %s"
        delete(delete_query, (session_id,))  # Pass session_id as a tuple to avoid SQL injection
        return jsonify({"success": True}), 200
    except Exception as e:
        print(f"Error deleting session: {e}")
    
        return jsonify({"error": "Error deleting session"}), 500




@app.route('/rename_chat', methods=['POST'])
def rename_chat():
    session_id = request.json.get('session_id')
    new_name = request.json.get('new_name')
    query = "UPDATE sessions SET name = %s WHERE id = %s"
    val = (new_name, session_id)
    if not session_id or not new_name:
        return jsonify({"error": "Session ID or new name not provided"}), 400  # Handle missing data

    else:
        rename(query, val)

    return jsonify({"success": True}), 200  # Return success response
######chain######


# 2. Incorporate the retriever into a question-answering chain.
def creatprompt():
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)




    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )


    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )


    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain






#######endChain######
@app.route('/send_message', methods=['POST'])
def send_message():
    history = []
    session_id = request.json.get('session_id')  # Get the session ID
    message = request.json.get('message')  # Get the user message
    


    results = getAll("SELECT * from chat WHERE sessionId = %s", (session_id,))
    for result in results:
        history.extend(
        [
            HumanMessage(content=result[2]),
            AIMessage(content=result[3]),
        ]
        )
    # generating llm response
        
    rag_chain = creatprompt()

    

    
    answer = rag_chain.invoke({"input": message, "chat_history": history})
    
        
    
    # Insert user message into the chat table
    
    insert("INSERT INTO chat (sessionid, userText, modelText) VALUES (%s, %s, %s)",(session_id, message, answer["answer"]))
    
    
    return jsonify({"success": True, "session_id": session_id, "results": results, "modelText": answer["answer"]}), 200  # Return the updated conversation





@app.route('/conversation', methods=['POST'])
def conversation():
    session_id = request.json.get("session_id")
    
    # Use a parameterized query instead of f-string
    results = getAll("SELECT * from chat WHERE sessionId = %s", (session_id,))
    
    return jsonify({"success": True, "session_id": session_id, "results": results}), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)
