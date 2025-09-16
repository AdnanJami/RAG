
import ollama
from flask import Flask, request, jsonify, render_template
import re
import chromadb
from datetime import datetime
from flask_mysqldb import MySQL
from flask import redirect, url_for
from sentence_transformers import SentenceTransformer, util
import json
import ast
from pprint import pprint
# model = GPT4All("Meta-Llama-3-8B-Instruct.Q4_0.gguf", model_path="C:/Users/Lenovo/AppData/Local/nomic.ai/GPT4All/", allow_download=False) # downloads / loads a 4.66GB LLM

app = Flask(__name__, template_folder="template")

# MySQL Configuration
app.config['MYSQL_HOST'] = 'localhost'  # or your MySQL server
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'chatbot'

client = chromadb.PersistentClient(path="database")
collection_phy_cosine = client.get_or_create_collection(name="physics-cosine-by-page",  metadata={"hnsw:space": "cosine"})

# def format_model_response(response):
#     # Convert numbered lists (e.g., "1." or "2.") into <ol> and <li>
#     response = re.sub(r'(\d+)\.\s', r'<li>', response)
#     response = response.replace("\n", "</li>\n")  # Close list items after each newline

#     # Detect bold patterns (e.g., headings before a colon)
#     response = re.sub(r'(\b[A-Za-z\s]+\b):', r'<b>\1</b>:', response)

#     # Replace **some text** with <b>some text</b> for bold
#     response = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', response)

#     # Wrap the entire list in <ol> if necessary
#     if '<li>' in response:
#         response = '<ol>\n' + response + '\n</ol>'

#     # Replace double newlines with paragraph breaks
#     response = response.replace("\n\n", "<br><br>")

#     return response

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
   # print(similarity.item()*100)
    return similarity.item()*100


def llmRagQuery(prompt):
    model = ollama.generate(
        model="llama3.2",
        system=""" 
        you are a rag assistant who will identify the keywords and the main Keyword topic of a prompt given to you.
        if the prompt mentions the page number then your response will contain the page number
        similary if the the prompt contains a specific chapter or chapters then your response will contain the chapter numbers
        you do not need to explain, describe or answer any of the questions, or queries of the user.
        all you need to do is answer by following the given structure and make sure the mainKeyword is either one or two words and not more.
        The main keyword cannot be the following words:
        1. formula
        2. equation
        3. calculate
        4. explain/describe/how/why/what/who/when/elaborate etc
        5. measure
        6. quantify
        7. any type of verbs, adverbs or adjectives
        your response structure must look like this without any exceptions!:

        {

            "keywords": [Value, Value, ..., Value] || [],
            "mainKeyword": Value || ""
        } 


        """,
        prompt=prompt,
    )
    # print(model["response"])
    pattern = r'\{.*?\}'

    # Find the JSON object in the string
    match = re.search(pattern, model["response"], re.DOTALL)
    
    data_dict = [{"keywords": [], "mainKeyword": ""}]
    if match:
        json_string = match.group(0)  # Get the matched JSON string
        data_dict = json.loads(json_string)
        # print("Extracted JSON string:", json_string)

    # Convert the extracted string to a dictionary
    
    
    # print("page:", data_dict["page"])
    # print("chapter:", data_dict["chapter"])
    # print("keywords:", data_dict["keywords"])
    # print("mainKeyword:", data_dict["mainKeyword"])
    if data_dict["mainKeyword"] is not None:
        mainKeyword = data_dict["mainKeyword"].upper()
    if data_dict["keywords"] is not None:
        LLMkeywordsArr = [s.upper() for s in data_dict["keywords"]]


    
    # newPrompts = prompt + " "
    # llmPrompts = re.compile(r"([0-9 .]+)(.+)")
    # llmPrompts = re.finditer(llmPrompts,model["response"])

    # for prompts in llmPrompts:
    #     newPrompts += prompts.group(2) + " "
    # Now you have the full response as well if you want to use it later
    
    arr = []
    sorted_data = []
    if not mainKeyword:
         mainKeyword = "0"
    if mainKeyword != "0" :
    
        result = collection_phy_cosine.query(query_texts=prompt, n_results=5)
        # pprint(result)
        if  len(result["metadatas"]) != 0 and  len(result["documents"]) != 0:    
            for resM, resDoc, resDis in zip(result["metadatas"][0], result["documents"][0], result["distances"][0]):
                    # print(resM, "\n\n")
                    print(resM["Chapter"],resM["ChapterTitle"],resM["Keywords"], "\n\n")
                    keyswordsArr = ast.literal_eval((resM["Keywords"]))
                    commonArr1 = []
                    for keys in keyswordsArr:
                        for llmKeys in LLMkeywordsArr:
                            if check_Meaning(llmKeys,keys) > 85:
                                commonArr1.append(keys)
                        # print(keys)
                    # print(resDoc)
                    # commonArr = list(set(LLMkeywordsArr) & set(keyswordsArr))
                    count = len(commonArr1)
                    # print(commonArr, "\n")
                    # print(commonArr1)
                    arr.append({"page": resM["page"], "chapter": (resM["Chapter"] + "---" + resM["ChapterTitle"]), "count": count, "distances": resDis, "documents": resDoc,  "keywords":keyswordsArr})
            sorted_data = sorted(arr, key=lambda x: (-x['count'], x['distances']))
            arr = []
            if sorted_data is not None :
                for res in sorted_data[:3]:
                    print(res["count"])
                    print(res["distances"])
                    print(res["page"])
                    print(res["keywords"])
                    for key in res["keywords"]:
                        if check_Meaning(mainKeyword,key) > 85:
                            arr.append({"doc":res["documents"], "page":res["page"],"chapter":res["chapter"]})
                            break
            print(len(arr))
            pprint(arr)

            # Printing the sorted result
            # for item in sorted_data:
            #     print(item)
                            
            # result = {"documents": [item["documents"] for item in sorted(arr, key=lambda x: x["count"], reverse=True)[:2]]}
            # result = {"keywords": [ast.literal_eval(item["keywords"]) for item in sorted(arr, key=lambda x: x["count"], reverse=True)[:2]]}
            # pprint(result["keywords"])
            # if not result["keywords"]:
            #     meaning = check_Meaning(prompt, result["keywords"])
            # else:
            #     meaning = check_Meaning(prompt, result["documents"][0])
            # pprint(result)
        
    # else:
    #     result = collection_phy_cosine_by_page.query(query_texts=prompt, n_results=2, include=["metadatas", "distances"])
    #     meaning = check_Meaning(prompt, result["documents"])
    
    return arr

def ragQuery(prompt):
    client = chromadb.PersistentClient(path="database")
    collection_phy_cosine = client.get_or_create_collection(name="physics-cosine-by-page",  metadata={"hnsw:space": "cosine"})
    result = collection_phy_cosine.query(query_texts=prompt, n_results=2, )
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
    context = " "
    if query is not None:
        for q in query:
            context += q["doc"]
    
    
    
    print(context)


    results = getAll("SELECT * from chat WHERE sessionId = %s", (session_id,))
    for result in results:
        history.append({"userPrompt": result[2], "System": result[3]})
    system ="""You are an RAG A.I. assistant who is only able to answer questions based on the provided context and your answers are straight to the point and you are good at explaining mathematical equations, graphs, and concepts related to physics. 
                answer concisely and accurately. 
                Absolutely don’t repeat system context, PreviousConversation, context, or userPrompt in your response.
                if its not a question then your response is based on previousConversation if it exists
               
            """
    prompt = f"""
        "PreviousConversation": {history},
        "Context": {query},
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

    
    # Insert user message into the chat table
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



