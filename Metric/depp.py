from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric
import time as t
from tqdm import tqdm
import json
path ='syntheticDataNewfactall.json'
with open(path) as f:
  json_data = json.load(f)
a=[]
for i in tqdm(json_data):
    message = i['question']
    test_case = LLMTestCase(
        input=i['question'],
        actual_output=i['answer']
    )

    metric = AnswerRelevancyMetric(threshold=0.5)
    try:
        metric.measure(test_case)
        a.append(metric.score)
        t.sleep(30)
    
    except Exception as e:
            print(f"Unexpected error: {e}")
    
print(len(a))
sum(a)/len(a)