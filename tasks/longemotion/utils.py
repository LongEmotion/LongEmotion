import re
import json
import datasets
import string
from collections import Counter
from datasets import Features, Value


def load_emotion_summary_dataset(**kwargs):
    """Load Emotion_Summary dataset with explicit features to handle inconsistent schema."""
    features = Features({
        'id': Value('int32'),
        'case_description': Value('string'),  # Treat as string to avoid schema conflicts
        'causes': Value('string'),
        'characteristics_of_illness': Value('string'),
        'consultation_process': Value('string'),
        'experience_and_reflection': Value('string'),
        'symptoms': Value('string'),
        'treatment_effect': Value('string'),
        'treatment_process': Value('string'),
    })
    
    data_files = kwargs.get('data_files', './data/Emotion_Summary.jsonl')
    return datasets.load_dataset('json', data_files=data_files, features=features)


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)

        return "".join(ch if ch not in exclude else " " for ch in text)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))



def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        choices = doc["choices"]
        if isinstance(choices, str):
            choices = json.loads(choices)
        choices_str = ", ".join(choices)
        return {
            "content": doc["content"],
            "subject": doc["subject"],
            "choices": choices,
            "choices_str": choices_str,
            "label": doc["label"],
        }
    return dataset.map(_process_doc)

def doc_to_text(doc):
    prompt = f"""Scenario:\n
{doc["content"]}
Question: What emotion(s) would {doc["subject"]} ultimately feel in this situation?
Choices:\n{doc["choices_str"]}
Only return the selected label in the output, without any additional content.
Please provide your answer in a structured JSON format as follows: 
```json
{{"Emotion": ...}}
```"""
    return prompt

class ExtractEmotionFilter:
    """Extract emotion field from JSON response."""

    def apply(self, resps, docs):
        filtered_resps = []
        for resp, doc in zip(resps, docs):
            resp_text = resp[0] if isinstance(resp, list) else resp
            emotion = None
            try:
                data = json.loads(resp_text)
                emotion = data.get("Emotion", None)
            except:
                matches = list(re.finditer(r'```json\s*(\{.*?\})\s*```', resp_text, re.DOTALL))
                if matches:
                    try:
                        json_str = matches[-1].group(1)
                        data = json.loads(json_str)
                        emotion = data.get("Emotion", None)
                    except:
                        pass
                if emotion is None:
                    match = re.search(r'\{[^{}]*"Emotion"\s*:\s*"?([^"}\n]+)"?\s*\}', resp_text)
                    if match:
                        emotion = match.group(1).strip().strip('"')
            if emotion is None:
                emotion = resp_text.strip()
            filtered_resps.append(emotion)
        return filtered_resps

class ExtractIndexFilter:
    """Extract index field from JSON response."""

    def apply(self, resps, docs):
        filtered_resps = []
        for resp, doc in zip(resps, docs):
            resp_text = resp[0] if isinstance(resp, list) else resp
            index = None
            try:
                data = json.loads(resp_text)
                index = data.get("index", None)
            except:
                matches = list(re.finditer(r'```json\s*(\{.*?\})\s*```', resp_text, re.DOTALL))
                if matches:
                    try:
                        json_str = matches[-1].group(1)
                        data = json.loads(json_str)
                        index = data.get("index", None)
                    except:
                        pass
                if index is None:
                    match = re.search(r'\{[^{}]*"index"\s*:\s*"?([^"}\n]+)"?\s*\}', resp_text)
                    if match:
                        index = match.group(1).strip().strip('"')
            if index is None:
                index = resp_text.strip()
            filtered_resps.append(str(index))
        return filtered_resps


class ExtractAnswerFilter:
    """Extract answer field from JSON response or return the first line."""

    def apply(self, resps, docs):
        filtered_resps = []
        for resp, doc in zip(resps, docs):
            resp_text = resp[0] if isinstance(resp, list) else resp
            answer = None
            try:
                data = json.loads(resp_text)
                answer = data.get("answer", None)
            except:
                matches = list(re.finditer(r'```json\s*(\{.*?\})\s*```', resp_text, re.DOTALL))
                if matches:
                    try:
                        json_str = matches[-1].group(1)
                        data = json.loads(json_str)
                        answer = data.get("answer", None)
                    except:
                        pass
                if answer is None:
                    match = re.search(r'\{[^{}]*"answer"\s*:\s*"?([^"}\n]+)"?\s*\}', resp_text)
                    if match:
                        answer = match.group(1).strip().strip('"')
            if answer is None:
                # Return first line if no JSON found
                answer = resp_text.strip().split('\n')[0].strip()
            filtered_resps.append(answer)
        return filtered_resps


def qa_f1_score(prediction: str, ground_truth: str, **kwargs):
    """Calculate F1 score for QA tasks."""
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()

    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def get_qa_f1_with_score(doc: dict, results: list[str], **kwargs):
    """Process results for QA tasks and return F1 score."""
    prediction = results[0].strip()
    ground_truth = doc.get("answer", "")
    score = qa_f1_score(prediction, ground_truth)
    return {"score": score, "qa_f1_score": score}


class ExtractReportSummaryFilter:
    """Extract multiple fields from JSON response for report summary task."""

    def apply(self, resps, docs):
        filtered_resps = []
        for resp, doc in zip(resps, docs):
            resp_text = resp[0] if isinstance(resp, list) else resp
            result = {}
            
            # Try to parse as JSON
            try:
                data = json.loads(resp_text)
                result = {
                    "causes": data.get("causes", ""),
                    "symptoms": data.get("symptoms", ""),
                    "treatment_process": data.get("treatment_process", ""),
                    "characteristics_of_illness": data.get("characteristics_of_illness", ""),
                    "treatment_effect": data.get("treatment_effect", "")
                }
            except:
                # Try to extract JSON from markdown code blocks
                matches = list(re.finditer(r'```json\s*(\{.*?\})\s*```', resp_text, re.DOTALL))
                if matches:
                    try:
                        json_str = matches[-1].group(1)
                        data = json.loads(json_str)
                        result = {
                            "causes": data.get("causes", ""),
                            "symptoms": data.get("symptoms", ""),
                            "treatment_process": data.get("treatment_process", ""),
                            "characteristics_of_illness": data.get("characteristics_of_illness", ""),
                            "treatment_effect": data.get("treatment_effect", "")
                        }
                    except:
                        pass
            
            filtered_resps.append(result)
        return filtered_resps


def get_report_summary_score(doc: dict, results: list, **kwargs):
    """Process results for report summary task.
    Note: This is a placeholder for integration with lm-eval-harness.
    The actual evaluation is done by score_report_summary in base.py.
    """
    # Return a dummy score for now - actual scoring is done in base.py
    return {"score": 0.0}