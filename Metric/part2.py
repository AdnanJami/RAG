import re
import google.generativeai as genai
import json
genai.configure(api_key="AIzaSyBiqm0_3984arLrc91nFXCgZ_COQjAppWU")

# Create the model
generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 64,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
  model_name="gemini-1.5-flash",
  generation_config=generation_config,
  system_instruction="""
  There can not be any duplicate question. 
  Structure your response in this format: 
                        "question": question,
                        "referenceAnswer": answer,
                        "referenceContext": chunk"""
)



chat_session = model.start_chat(
  history=[
  ]
)
document = ""

# reading physics chapter wise txt files
try:
    with open('splited_chapter/physics-180-202.txt', 'r') as file:
        document = file.read()
except FileNotFoundError:
    print("The file does not exist.")
# with open('dataset\synthetic_dataset_chaper_1.json', 'r', encoding="utf-8") as file:
#     data = json.load(file)
# print(len(data))
def chunk_text(text, chunk_size=3000):
    # Split by sentences for simplicity, or customize for your needs
    sentences = text.split('. ')
    chunks = []
    current_chunk = []
    current_size = 0
    
    for sentence in sentences:
        current_chunk.append(sentence)
        current_size += len(sentence)
        
        if current_size >= chunk_size:
            chunks.append('. '.join(current_chunk))
            current_chunk = []
            current_size = 0
    
    # Append any remaining sentences
    if current_chunk:
        chunks.append('. '.join(current_chunk))
    
    return chunks

# Split the  document into smaller chunks
chunks = chunk_text(document, chunk_size=3000)
print(len(chunks))
# for storing dataset questions
synthetic_data = []

# change the chapter name
def SaveDataset(data, path=None, fileName="syntheticData"):
  if path is None:
      filePath = f"{fileName}.json"
  else:
      filePath = f"{path}/{fileName}.json"
  with open(filePath, 'a') as f:
      json.dump(data, f, indent=4)
  print("Saved Successfully!")

# SaveDataset(synthetic_data,"dataset","synthetic_data_chapter_1")


for chunk in chunks:
    response = chat_session.send_message(f"Generate 5 or more dataset if possible based on this text: {document}")
  # Regular expression to capture key-value pairs
    regex = r'"(\w+)":\s*"([^"]*)"'

    # Find all key-value pairs using regex
    matches = re.findall(regex, response.text)

    # Convert the list of tuples into a dictionary or list of dictionaries for each object
    current_dict = {}
    temp_result = []
    for key, value in matches:
        current_dict[key] = value
        if key == "referenceContext":
            # Append to results once the context is complete and start a new dictionary
            synthetic_data.append(current_dict)
            temp_result.append(current_dict)
            current_dict = {}
    SaveDataset(temp_result,"dataset","synthetic_dataset_chaper_11_conc")
  #     synthetic_data.append(result)








