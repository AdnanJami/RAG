from deepeval.metrics import HallucinationMetric
from deepeval.metrics import ContextualPrecisionMetric
from deepeval.metrics import ContextualRecallMetric
from deepeval.metrics import ContextualRelevancyMetric
from deepeval.metrics import KnowledgeRetentionMetric
from deepeval.test_case import ConversationalTestCase
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric
import time as t

def ContextRelevancyMetric(input,actual_output,retrieval_context):
        test = LLMTestCase(
        input=input,
        actual_output=actual_output,
        retrieval_context=retrieval_context
        )
        metric = ContextualRelevancyMetric(threshold=0.8)
        arr = None
        try:
            metric.measure(test)
            arr = {

                "score": metric.score,
                "reason": metric.reason
            }
            print(f"ContextRelevancyMetric: {metric.score}")
            # print(metric.reason)
        except ValueError as ve:
            print(f"ValueError: {ve}")
            # print(f"Test Case: {test}")
        except Exception as e:
            print(f"Unexpected error: {e}")
        return arr
def AnswerRelevancyMetrics(input,actual_output):
        test = LLMTestCase(
        input=input,
        actual_output=actual_output
        )
        metric = AnswerRelevancyMetric(threshold=0.5)
        arr = None
        try:
            metric.measure(test)
            arr = {

                "score": metric.score,
                "reason": metric.reason
            }
            print(f"AnswerRelevancyMetrics: {metric.score}")
            # print(metric.reason)
        except ValueError as ve:
            print(f"ValueError: {ve}")
            # print(f"Test Case: {test}")
        except Exception as e:
            print(f"Unexpected error: {e}")
        t.sleep(5)
        return arr

def ContextRecallMetric(input,actual_output,expected_output,retrieval_context):
        test = LLMTestCase(
        input=input,
        actual_output=actual_output,
        expected_output=expected_output,
        retrieval_context=retrieval_context
        )
        metric = ContextualRecallMetric(threshold=0.7)
        arr = None
        try:
            metric.measure(test)
            arr ={

                "score": metric.score,
                "reason": metric.reason
            }
            print(f"ContextRecallMetric: {metric.score}")
        except ValueError as ve:
            print(f"ValueError: {ve}")
            # print(f"Test Case: {test}")
        except Exception as e:
            print(f"Unexpected error: {e}")
        
        return arr

def ContextPrecisionMetric(input,actual_output,expected_output,retrieval_context):
        test = LLMTestCase(
        input=input,
        actual_output=actual_output,
        expected_output=expected_output,
        retrieval_context=retrieval_context
        )
        metric = ContextualPrecisionMetric(threshold=0.5)
        arr = None
        try:
            metric.measure(test)
            arr = {
                "score": metric.score,
                "reason": metric.reason
            }
            print(f"ContextPrecisionMetric: {metric.score}")
        except ValueError as ve:
            print(f"ValueError: {ve}")
            # print(f"Test Case: {test}")
        except Exception as e:
            print(f"Unexpected error: {e}")
        return arr

def HallucinateMetrics(input,actual_output,context):
        test = LLMTestCase(
        input=input,
        actual_output=actual_output,
        context=context,
        )
        metric = HallucinationMetric(threshold=0.5)
        arr = None
        try:
            metric.measure(test)

            arr = {

                "score": metric.score,
                "reason": metric.reason
            }
            print(f"HallucinateMetrics: {metric.score}")
        except ValueError as ve:
            print(f"ValueError: {ve}")
            print(f"Test Case: {test}")
        except Exception as e:
            print(f"Unexpected error: {e}")
        return arr

def KnowledgeMetrics(data):
        turns = []
        for d in data:
        # Retrieve necessary fields
            actual_output = d.get("answer")
            input = d.get("question")

            schema = {
                "question":input,
                "answer":actual_output,
                "metrics": {
                    "KnowledgeRetention":[],

                }
                    
                
            }

            test = LLMTestCase(
            input=input,
            actual_output=actual_output,
            
            )
            turns.append(test)
        test_case = ConversationalTestCase(turns=turns)
        metric = KnowledgeRetentionMetric(threshold=0.7)

        arr = None
        try:
            metric.measure(test_case)
            arr = {

                "score": metric.score,
                "reason": metric.reason
            }
            print(metric.score)
            print(metric.reason)
        except ValueError as ve:
            print(f"ValueError: {ve}")
            print(f"Test Case: {test}")
        except Exception as e:
            print(f"Unexpected error: {e}")
        schema["metrics"]["KnowledgeRetention"].append(arr)
        
        return  schema




# test_case_results.append(ComprehensionMetrics(data))
import json
def runAllTests(data,n):
    test_case_results = []
    for i, d in enumerate(data):
        # Retrieve necessary fields
        actual_output = d.get("answer")
        input = d.get("question")
        expected_output = d["referenceAnswer"]
        
        context =[ d["referenceContext"] ]
        retrieval_context = [d.get("context")]
        schema = {
            "question":input,
            "answer":actual_output,
            "referenceAnswer":expected_output,
            "referenceContext":retrieval_context,
            "context": context,
            "metrics": {
                "Hallucination":[],
                "ContextPrecision": [],
                "ContextRecall": [],
                "ContextRelevancy": [],
                "AnswerRelevancy":[],
            }
                
            
        }
        schema["metrics"]["ContextPrecision"].append(ContextPrecisionMetric(input,actual_output,expected_output,retrieval_context))
        schema["metrics"]["ContextRecall"].append(ContextRecallMetric(input,actual_output,expected_output,retrieval_context))
        schema["metrics"]["ContextRelevancy"].append(ContextRelevancyMetric(input,actual_output,retrieval_context))
        schema["metrics"]["Hallucination"].append(HallucinateMetrics(input,actual_output,context))
        schema["metrics"]["AnswerRelevancy"].append(AnswerRelevancyMetrics(input,actual_output))
        test_case_results.append(schema)
        print(f"Dataset: {i}")
        # print(schema)
        # Print debug information
        # print(f"Input: {input}, Actual Output: {actual_output}, Context: {context}")
        with open(f"dataset/new_datase_con48_evaluated_{n}.json", 'a') as f:
          json.dump(schema, f, indent=4)
        print("Saved Successfully!")
        t.sleep(15)
        # Create the test case
    return test_case_results


# Load dataset

with open(f"cleaned_syntheticData.json", "r", encoding="utf-8") as json_file:
        data = json.load(json_file)

    # Save evaluation data
test_case_results = runAllTests(data,1)