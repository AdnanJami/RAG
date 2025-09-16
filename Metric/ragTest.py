from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pprint import pprint
from sentence_transformers import SentenceTransformer, util
import json
import ollama
import numpy as np
import mysql.connector
connection = mysql.connector.connect(
    host="localhost",  # Replace with your MySQL server host
    user="root",  # Replace with your MySQL username
    password="",  # Replace with your MySQL password
    database="chatbot"  # Replace with the database you want to use
)
def GetAll(query):
    cursor = connection.cursor()
    cursor.execute(query)  # Pass the query and the values separately
    
    results = cursor.fetchall()
    cursor.close()
    return results

result = GetAll("SELECT time FROM `chat` ")
values = [x[0] for x in result]
values.sort()

n = len(values)

# Step 3: Find n/4 and 3n/4 thresholds
n_4 = n // 4
n_3_4 = (3 * n) // 4

# Step 4: Extract lower n/4 and upper 3n/4 elements
lower_quartile = values[:n_4]
upper_quartile = values[n_3_4:]

# Step 5: Calculate the mean of each subset
mean_lower = np.mean(lower_quartile)
mean_upper = np.mean(upper_quartile)

# Output the results
print(f"Lower quartile mean: {mean_lower}")
print(f"Upper quartile mean: {mean_upper}")
connection.close()
# for  d in result:
    # time.append({"question":d[2],"answer":d[3],"context":d[4],"time":d[5]}) 



# SimilarityArr = []
# # Function to find similarity in texts (repetition tests)
# def check_repetition(old_response, new_response):
#     # Create the TF-IDF matrix
#     vectorizer = TfidfVectorizer().fit_transform([old_response, new_response])
#     vectors = vectorizer.toarray()

#     # Calculate cosine similarity
#     similarity = cosine_similarity(vectors)[0][1] * 100
    
#     # Return dict
#     return {
#         "OldResponse": old_response,
#         "newResponse": new_response,
#         "similarity" : similarity,
#         "repetition" : similarity >= 80
#     }  


# comprehensionArr = []
# # Load the pre-trained model
# model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
# def check_Meaning(old_response, new_response):
#     # embed the texts
#     embeddings = model.encode([old_response, new_response])
#     # find similarity useing cosine similarity
#     similarity = util.cos_sim(embeddings[0], embeddings[1])
#     comprehension = "Poor"
#     if similarity.item()*100 >= 85:
#         comprehension = 1.00
#     elif similarity.item()*100 >= 70 and similarity.item()*100 <85:
#         comprehension = 0.7
#     elif similarity.item()*100 >= 50 and similarity.item()*100 <70:
#         comprehension = 0.5
#     else:
#         comprehension = similarity.item()

#     # return dict
#     return {
#         "OldResponse": old_response,
#         "newResponse": new_response,
#         "similarity" : similarity.item()*100,
#         "comprehension": comprehension
#     }

# def Repetition(old_prompt, new_prompt, old_response, new_response):
#         isRepeated = check_repetition(old_prompt, new_prompt)
#         if isRepeated["repetition"]:
#             result = check_repetition(old_response,new_response)
#             return result
#         else: 
#             return None

# def llmRagQuery(prompt, system):
#         model = ollama.generate(
#             model="llama3.2",
#             system=system,
#             prompt=prompt,
#         )
#         return model["response"]


# Load dataset
# with open("dataset_1.json", "r", encoding="utf-8") as json_file:
#         data = json.load(json_file)

# def RepetitionMetric(data):
#     arr = []
#     count = 0
#     for d in data:
#         count += 1
#         for i,da in enumerate(data):
#             if d["question"] != da["question"]:
#                 result = Repetition(d["question"], da["question"], d["answer"],da["answer"])
            
#                 if result is not None:
#                     arr.append(result)
#             # else:
#             #     result = Repetition(d["question"], da["question"], d["answer"],da["answer"])
#             #     arr.append(result)
#         # print(d)
#     rep = 0
#     print(count)
#     if arr is not None:
#          rep = len(arr)
#     return (rep/count)

# def ComprehensionMetrics(data):
#   arr = []
#   count = 0
#   for d in data:
#         count += 1
#         result = llmRagQuery(d["question"],""" 
#             you are a rag assistant who will rewrite the question/prompt given by the user.
#             you will rewrite 3 more times.
#             the rewritten prompts/questions has to be exactly the same topic as that of the user's.
#             it can't be too advanced nor too simplistic compared to the user's
#             you will not include anything else in your answer.
#             list the rewritten questions/prompts in an ordered list
            
#             """)
#         comPrompt = check_Meaning(d["question"],result)
        
#         result = llmRagQuery(d["answer"],""" 
#             you are an ai assistant who will write the question/prompt based on the provided answer by an llm.
#             Write only the question and absolutely nothing else in your response.
#             """)
#         comResponse = check_Meaning(d["question"],result)
#         arr.append({"comPrompt":comPrompt,"comResponse":comResponse})

#   print(count)
#   return arr


# print(RepetitionMetric(data))
# pprint(comprehensionArr)
# pprint(SimilarityArr)


# ###### DATASET 1 #########



# for i,res in enumerate(Questions):
#     if i != (len(Questions)-1):
#         comprehensionArr.append(check_Meaning(Questions[i],Questions[i+1]))
#         SimilarityArr.append(check_repetition(Questions[i], Questions[i+1]))
#     else:
#         comprehensionArr.append(check_Meaning(Questions[i],Questions[i]))
#         SimilarityArr.append(check_repetition(Questions[i], Questions[i]))

# pprint(comprehensionArr)
# pprint(SimilarityArr)


# from deepeval.metrics import HallucinationMetric
# from deepeval.metrics import ContextualPrecisionMetric
# from deepeval.metrics import ContextualRecallMetric
# from deepeval.metrics import ContextualRelevancyMetric
# from deepeval.metrics import KnowledgeRetentionMetric
# from deepeval.test_case import ConversationalTestCase
# from deepeval.metrics import GEval
# from deepeval.test_case import LLMTestCaseParams
# from deepeval.test_case import LLMTestCase

# def ContextRelevancyMetric(input,actual_output,retrieval_context):
#         test = LLMTestCase(
#         input=input,
#         actual_output=actual_output,
#         retrieval_context=retrieval_context
#         )
#         metric = ContextualRelevancyMetric(threshold=0.8)
#         arr = None
#         try:
#             metric.measure(test)
#             arr = {

#                 "score": metric.score,
#                 "reason": metric.reason
#             }
#             print(metric.score)
#             print(metric.reason)
#         except ValueError as ve:
#             print(f"ValueError: {ve}")
#             # print(f"Test Case: {test}")
#         except Exception as e:
#             print(f"Unexpected error: {e}")

#         return arr

# def ContextRecallMetric(input,actual_output,expected_output,retrieval_context):
#         test = LLMTestCase(
#         input=input,
#         actual_output=actual_output,
#         expected_output=expected_output,
#         retrieval_context=retrieval_context
#         )
#         metric = ContextualRecallMetric(threshold=0.7)
#         arr = None
#         try:
#             metric.measure(test)
#             arr ={

#                 "score": metric.score,
#                 "reason": metric.reason
#             }
#             print(metric.score)
#             print(metric.reason)
#         except ValueError as ve:
#             print(f"ValueError: {ve}")
#             # print(f"Test Case: {test}")
#         except Exception as e:
#             print(f"Unexpected error: {e}")

#         return arr

# def ContextPrecisionMetric(input,actual_output,expected_output,retrieval_context):
#         test = LLMTestCase(
#         input=input,
#         actual_output=actual_output,
#         expected_output=expected_output,
#         retrieval_context=retrieval_context
#         )
#         metric = ContextualPrecisionMetric(threshold=0.5)
#         arr = None
#         try:
#             metric.measure(test)
#             arr = {
#                 "score": metric.score,
#                 "reason": metric.reason
#             }
#             print(metric.score)
#             print(metric.reason)
#         except ValueError as ve:
#             print(f"ValueError: {ve}")
#             # print(f"Test Case: {test}")
#         except Exception as e:
#             print(f"Unexpected error: {e}")

#         return arr

# def HallucinateMetrics(input,actual_output,context):
#         test = LLMTestCase(
#         input=input,
#         actual_output=actual_output,
#         context=context,
#         )
#         metric = HallucinationMetric(threshold=0.5)
#         arr = None
#         try:
#             metric.measure(test)
#             test_case_results.append({
#                 "score": metric.score,
#                 "reason": metric.reason
#             })
#             arr = {

#                 "score": metric.score,
#                 "reason": metric.reason
#             }
#             print(metric.score)
#             print(metric.reason)
#         except ValueError as ve:
#             print(f"ValueError: {ve}")
#             print(f"Test Case: {test}")
#         except Exception as e:
#             print(f"Unexpected error: {e}")

#         return arr

# def KnowledgeMetrics(data):
#         turns = []
#         for d in data:
#         # Retrieve necessary fields
#             actual_output = d.get("answer")
#             input = d.get("question")

#             schema = {
#                 "question":input,
#                 "answer":actual_output,
#                 "metrics": {
#                     "KnowledgeRetention":[],

#                 }
                    
                
#             }

#             test = LLMTestCase(
#             input=input,
#             actual_output=actual_output,
            
#             )
#             turns.append(test)
#         test_case = ConversationalTestCase(turns=turns)
#         metric = KnowledgeRetentionMetric(threshold=0.7)

#         arr = None
#         try:
#             metric.measure(test_case)
#             arr = {

#                 "score": metric.score,
#                 "reason": metric.reason
#             }
#             print(metric.score)
#             print(metric.reason)
#         except ValueError as ve:
#             print(f"ValueError: {ve}")
#             print(f"Test Case: {test}")
#         except Exception as e:
#             print(f"Unexpected error: {e}")
#         schema["metrics"]["KnowledgeRetention"].append(arr)
#         return  schema

# with open("dataset_2.json", "r", encoding="utf-8") as json_file:
#         data = json.load(json_file)

# test_case_results = []
# test_case_results.append(ComprehensionMetrics(data))

# # Load dataset
# with open("dataset_1.json", "r", encoding="utf-8") as json_file:
#     data = json.load(json_file)


# def runAllTests(data):
#     for d in data:
#         # Retrieve necessary fields
#         actual_output = d.get("answer")
#         input = d.get("question")
#         expected_output = d["referenceAnswer"]
#         context = []
#         context.append(d.get("referenceContext"))
#         retrieval_context = d["retrievedContext"][0] if "retrievedContext" in d and d["retrievedContext"] else None
#         schema = {
#             "question":input,
#             "answer":actual_output,
#             "referenceAnswer":expected_output,
#             "retrievedContext":retrieval_context,
#             "referenceContext": context,
#             "metrics": {
#                 "Hallucination":[],
#                 "ContextPrecision": [],
#                 "ContextRecall": [],
#                 "ContextRelevancy": []
#             }
                
            
#         }
#         schema["metrics"]["ContextPrecision"].append(ContextPrecisionMetric(input,actual_output,expected_output,retrieval_context))
#         schema["metrics"]["ContextRecall"].append(ContextRecallMetric(input,actual_output,expected_output,retrieval_context))
#         schema["metrics"]["ContextRelevancy"].append(ContextRelevancyMetric(input,actual_output,retrieval_context))
#         schema["metrics"]["Hallucination"].append(HallucinateMetrics(input,actual_output,context))
#         test_case_results.append(schema)
#         print(schema)
#         # Print debug information
#         # print(f"Input: {input}, Actual Output: {actual_output}, Context: {context}")

#         # Create the test case
#     return test_case_results

    # Save evaluation data
# with open("dataset_eval.json", "w", encoding="utf-8") as json_file:
#     json.dump(test_case_results, json_file, indent=3)

### calulating metrics ###
# # Load dataset
# with open("dataset_2_eval.json", "r", encoding="utf-8") as json_file:
#     data = json.load(json_file)
# # print(data)
# def CalculateComprehension(data):
#      sum = 0
#      count = 0
#      for d in data:
#         sum += d["comPrompt"]["comprehension"] + d["comResponse"]["comprehension"]
#         count += 2
#      avg = (sum/count)*100
#      return int(avg)
# print("Comprehension:",CalculateComprehension(data))

# with open("dataset_1_eval.json", "r", encoding="utf-8") as json_file:
#     data = json.load(json_file)

# def Calculate(data):
#     Hallucinate = 0
#     Hcount = 0
#     ContextPrecision = 0
#     CPcount = 0
#     ContextRecall = 0
#     CRcount = 0
#     ContextRelevancy = 0
#     CRVcount = 0
    
#     for d in data:
#         try:
#             # Accessing nested scores directly using indexing
#             hallucination = d["metrics"]["Hallucination"][0]["score"]
#             Hallucinate += hallucination
#             Hcount += 1
#         except (KeyError, IndexError, TypeError):
#             pass  # Safely skip if any key or index is missing

#         try:
#             precision = d["metrics"]["ContextPrecision"][0]["score"]
#             ContextPrecision += precision
#             CPcount += 1
#         except (KeyError, IndexError, TypeError):
#             pass

#         try:
#             recall = d["metrics"]["ContextRecall"][0]["score"]
#             ContextRecall += recall
#             CRcount += 1
#         except (KeyError, IndexError, TypeError):
#             pass

#         try:
#             relevancy = d["metrics"]["ContextRelevancy"][0]["score"]
#             ContextRelevancy += relevancy
#             CRVcount += 1
#         except (KeyError, IndexError, TypeError):
#             pass
#     Havg = int((Hallucinate / Hcount) * 100) if Hcount > 0 else 0
#     CPavg = (ContextPrecision / CPcount) * 100 if CPcount > 0 else 0
#     CRavg = (ContextRecall / CRcount) * 100 if CRcount > 0 else 0
#     RetrievalPrecision = int((CRavg + CPavg) / 2) if CPcount > 0 and CRcount > 0 else 0
#     CRVavg = int((ContextRelevancy / CRVcount) * 100) if CRVcount > 0 else 0
#     return { 
#          "Hallucination":Havg,
#          "RetrievalPrecision":RetrievalPrecision,
#          "Relevance":CRVavg
#     }

# print(Calculate(data))