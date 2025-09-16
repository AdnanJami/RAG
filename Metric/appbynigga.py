
import ollama
from flask import Flask, request, jsonify, render_template
import re
import chromadb
from datetime import datetime
from flask_mysqldb import MySQL
from flask import redirect, url_for
from sentence_transformers import SentenceTransformer, util


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

modelMini = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def check_Meaning(old_response, new_response):
    # embed the texts
    embeddings = modelMini.encode([old_response, new_response])
    
    # find similarity useing cosine similarity
    similarity = util.cos_sim(embeddings[0], embeddings[1])
    print(similarity.item()*100)
    if similarity.item()*100 > 60:
        return new_response

    else:
        return old_response


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
    # Now you have the full response as well if you want to use it later
    client = chromadb.PersistentClient(path="database")
    collection_phy_cosine = client.get_or_create_collection(name="physics-cosine",  metadata={"hnsw:space": "cosine"})
    result = collection_phy_cosine.query(query_texts=prompt, n_results=3, )
    return result["documents"]

def ragQuery(prompt):
    client = chromadb.PersistentClient(path="database")
    collection_phy_cosine = client.get_or_create_collection(name="physics-cosine",  metadata={"hnsw:space": "cosine"})
    result = collection_phy_cosine.query(query_texts=prompt, n_results=3, )
    return result["documents"]

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

@app.route('/send_message', methods=['POST'])
def send_message():
    then = datetime.now()
    history = []
    
    session_id = request.json.get('session_id')  # Get the session ID
    message = request.json.get('message')  # Get the user message
   
    query = llmRagQuery(message)
    tmp = check_Meaning(message,query)
    context = ""
    if tmp == message:
        tmp = "None "
    
    else:
        for t in tmp[0]:
            context += t
    
    
    print(context)


    results = getAll("SELECT * from chat WHERE sessionId = %s", (session_id,))
    for result in results:
        history.append({"userPrompt": result[2], "System": result[3]})
    system ="""You are an A.I. assistant who is only able to answer questions straight to the point and who is good at explaining mathematical equations, graphs, and concepts related to physics. 
                you are going to answer in a way that is easier for 9th grade students to understand and try to make it fun and engaging for them. 
                If you do not know something, then say so; otherwise, answer concisely and accurately. 
                Absolutely don’t repeat system context, PreviousConversation, or userPrompt in your response.
                If context is empty then say you dont know.
            """
    prompt = f"""
        "PreviousConversation": {history},
        "Context": {context},
        "UserPrompt": {message}
    """
    response = ollama.generate(
    model="llama3.2",
    system=system,
    prompt=prompt
    
    )   

    now = datetime.now()
    time = now - then
    time = int(time.total_seconds())
    print("Time taken in seconds: ", time)

  
    insert("INSERT INTO chat (sessionid, userText, modelText,context,time) VALUES (%s, %s, %s, %s, %s)",(session_id, message, response["response"],context, time))
  
    
    return jsonify({"success": True, "session_id": session_id, "results": results, "modelText": response["response"]}), 200  # Return the updated conversation



@app.route('/conversation', methods=['POST'])
def conversation():
    session_id = request.json.get("session_id")
    
    # Use a parameterized query instead of f-string
    results = getAll("SELECT * from chat WHERE sessionId = %s", (session_id,))
    
    return jsonify({"success": True, "session_id": session_id, "results": results}), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)



