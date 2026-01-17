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
    """Process results for report summary task using OpenAI API (e.g., GPT-4o) for evaluation.
    
    Environment variables:
    - OPENAI_API_KEY: API key for the evaluator model
    - OPENAI_BASE_URL: Base URL for the OpenAI-compatible API (optional, defaults to OpenAI)
    - EVALUATOR_MODEL: Model name for evaluation (optional, defaults to gpt-4o)
    """
    import os
    import json
    import re
    
    # Load .env file automatically
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass  # dotenv is optional
    
    # Get environment variables
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Warning: OPENAI_API_KEY not set. Returning dummy score 0.0")
        return {"score": 0.0}
    
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model_name = os.getenv("EVALUATOR_MODEL", "gpt-4o")
    
    try:
        from openai import OpenAI
    except ImportError:
        print("Warning: openai package not installed. Returning dummy score 0.0")
        return {"score": 0.0}
    
    # Get model output
    model_output = results[0] if results else ""
    
    # Build ground truth string
    ground_truth_str = f"""Causes: {doc.get('causes', '')}
Symptoms: {doc.get('symptoms', '')}
Treatment process: {doc.get('treatment_process', '')}
Characteristics of the illness: {doc.get('characteristics_of_illness', '')}
Treatment effect: {doc.get('treatment_effect', '')}"""
    
    # Build model output string
    if isinstance(model_output, dict):
        model_output_str = f"""Causes: {model_output.get('causes', '')}
Symptoms: {model_output.get('symptoms', '')}
Treatment process: {model_output.get('treatment_process', '')}
Characteristics of the illness: {model_output.get('characteristics_of_illness', '')}
Treatment effect: {model_output.get('treatment_effect', '')}"""
    else:
        model_output_str = model_output
    
    # Load evaluation prompt
    import os as os_module
    base_dir = os_module.path.dirname(os_module.path.dirname(os_module.path.dirname(os_module.path.abspath(__file__))))
    prompt_file = os_module.path.join(base_dir, "prompts", "eval_prompt", "report_score_prompt.txt")
    
    try:
        with open(prompt_file, 'r', encoding='utf-8') as f:
            eval_prompt = f.read()
    except FileNotFoundError:
        print(f"Warning: Evaluation prompt not found at {prompt_file}. Returning dummy score 0.0")
        return {"score": 0.0}
    
    # Format the prompt
    formatted_prompt = eval_prompt.format(
        ground_truth=ground_truth_str,
        model_output=model_output_str
    )
    
    # Call OpenAI API
    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": formatted_prompt}
            ],
            temperature=0,
            max_tokens=4096
        )
        
        response_text = response.choices[0].message.content.strip()
        
        # Extract JSON from response
        json_pattern = r'```json\s*(.*?)\s*```'
        json_matches = re.findall(json_pattern, response_text, re.DOTALL)
        
        if json_matches:
            json_content = json_matches[0].strip()
            try:
                evaluation_result = json.loads(json_content)
                # Calculate average score across all attributes and dimensions
                total_score = 0
                count = 0
                for attr in evaluation_result.values():
                    for dim_name, score in attr.items():
                        if dim_name in ['factual_consistency', 'completeness', 'clarity']:
                            total_score += score
                            count += 1
                avg_score = total_score / count if count > 0 else 0
                return {"score": avg_score}
            except Exception as e:
                print(f"Error parsing evaluation JSON: {e}")
        
        # Try to extract JSON without markdown
        evaluate_ans = re.findall(r'\{(?:[^{}]|(?:\{(?:[^{}])*?\}))*?\}', response_text)
        if evaluate_ans:
            try:
                evaluation_result = json.loads(evaluate_ans[0])
                total_score = 0
                count = 0
                for attr in evaluation_result.values():
                    for dim_name, score in attr.items():
                        if dim_name in ['factual_consistency', 'completeness', 'clarity']:
                            total_score += score
                            count += 1
                avg_score = total_score / count if count > 0 else 0
                return {"score": avg_score}
            except Exception as e:
                print(f"Error parsing evaluation JSON: {e}")
        
        print("Warning: Could not extract valid JSON from evaluation response. Returning dummy score 0.0")
        return {"score": 0.0}
        
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return {"score": 0.0}


def load_emotion_conversations_dataset(**kwargs):
    """Load Emotion_Conversations dataset for multi-turn conversation task.
    Expands each sample into multiple evaluation points (1/4, 1/2, 3/4 for each of 4 stages).
    """
    import json
    
    data_files = kwargs.get('data_files', './data/Emotion_Conversations.jsonl')
    
    # Load raw data
    with open(data_files, 'r', encoding='utf-8') as f:
        raw_data = [json.loads(line) for line in f]
    
    expanded_data = []
    
    for item in raw_data:
        dialogue_id = item.get('id')
        stages = item.get('stages', [])
        if isinstance(stages, str):
            stages = json.loads(stages)
        
        all_stage_history = []  # Accumulate conversation history across stages
        
        for stage_idx, stage in enumerate(stages):
            stage_name = stage.get('stage', '')
            conversations = stage.get('conversations', [])
            
            # Find client message indices
            client_indices = [i for i, msg in enumerate(conversations) if msg.get('role') == 'Client']
            n_clients = len(client_indices)
            
            if n_clients == 0:
                continue
            
            # Define evaluation points: 1/4, 1/2, 3/4
            eval_points = {
                'quarter': n_clients // 4,
                'half': n_clients // 2,
                'three_quarters': (3 * n_clients) // 4
            }
            
            # Create a doc for each evaluation point
            for eval_label, point_idx in eval_points.items():
                if 0 <= point_idx < n_clients:
                    # Get conversation up to this client message
                    end_conversation_idx = client_indices[point_idx] + 1
                    current_conversations = conversations[:end_conversation_idx]
                    
                    # Build dialogue history string (including all previous stages + current stage up to point)
                    dialogue_history_str = ""
                    for msg in all_stage_history:
                        dialogue_history_str += f"{msg['role']}: {msg['context']}\n"
                    for msg in current_conversations:
                        dialogue_history_str += f"{msg['role']}: {msg['context']}\n"
                    
                    expanded_data.append({
                        'dialogue_id': dialogue_id,
                        'stage_idx': stage_idx,
                        'stage_name': stage_name,
                        'eval_point': eval_label,
                        'dialogue_history': dialogue_history_str,
                        'client_count': point_idx + 1,
                        'total_clients': n_clients,
                        # Store original stages for reference
                        'original_stages': json.dumps(stages)
                    })
            
            # Add current stage conversations to cumulative history
            all_stage_history.extend(conversations)
    
    features = Features({
        'dialogue_id': Value('int32'),
        'stage_idx': Value('int32'),
        'stage_name': Value('string'),
        'eval_point': Value('string'),
        'dialogue_history': Value('string'),
        'client_count': Value('int32'),
        'total_clients': Value('int32'),
        'original_stages': Value('string'),
    })
    
    return datasets.Dataset.from_list(expanded_data, features=features)


def process_emotion_conversations_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    """Process Emotion_Conversations dataset by expanding each sample into multiple evaluation points.
    Used with dataset_path: json and process_docs in YAML config.
    """
    import json
    
    expanded_data = []
    
    for item in dataset:
        dialogue_id = item.get('id')
        stages = item.get('stages', [])
        if isinstance(stages, str):
            stages = json.loads(stages)
        elif isinstance(stages, list):
            stages = stages
        else:
            continue
        
        all_stage_history = []  # Accumulate conversation history across stages
        
        for stage_idx, stage in enumerate(stages):
            stage_name = stage.get('stage', '')
            conversations = stage.get('conversations', [])
            
            # Find client message indices
            client_indices = [i for i, msg in enumerate(conversations) if msg.get('role') == 'Client']
            n_clients = len(client_indices)
            
            if n_clients == 0:
                continue
            
            # Define evaluation points: 1/4, 1/2, 3/4
            eval_points = {
                'quarter': n_clients // 4,
                'half': n_clients // 2,
                'three_quarters': (3 * n_clients) // 4
            }
            
            # Create a doc for each evaluation point
            for eval_label, point_idx in eval_points.items():
                if 0 <= point_idx < n_clients:
                    # Get conversation up to this client message
                    end_conversation_idx = client_indices[point_idx] + 1
                    current_conversations = conversations[:end_conversation_idx]
                    
                    # Build dialogue history string (including all previous stages + current stage up to point)
                    dialogue_history_str = ""
                    for msg in all_stage_history:
                        dialogue_history_str += f"{msg['role']}: {msg['context']}\n"
                    for msg in current_conversations:
                        dialogue_history_str += f"{msg['role']}: {msg['context']}\n"
                    
                    expanded_data.append({
                        'dialogue_id': dialogue_id,
                        'stage_idx': stage_idx,
                        'stage_name': stage_name,
                        'eval_point': eval_label,
                        'dialogue_history': dialogue_history_str,
                        'client_count': point_idx + 1,
                        'total_clients': n_clients,
                        # Store original stages for reference
                        'original_stages': json.dumps(stages)
                    })
            
            # Add current stage conversations to cumulative history
            all_stage_history.extend(conversations)
    
    return datasets.Dataset.from_list(expanded_data)


class ExtractCounselorResponseFilter:
    """Extract counselor response from model output."""

    def apply(self, resps, docs):
        filtered_resps = []
        for resp, doc in zip(resps, docs):
            resp_text = resp[0] if isinstance(resp, list) else resp
            # Return the full response as-is for counselor response
            filtered_resps.append(resp_text.strip())
        return filtered_resps


def get_multiconv_score(doc: dict, results: list, **kwargs):
    """Process results for multi-turn conversation task using OpenAI API (e.g., GPT-4o) for evaluation.
    Aligned with the original run_multicov method: evaluates at 1/4, 1/2, 3/4 points of each stage.

    Environment variables:
    - OPENAI_API_KEY: API key for the evaluator model
    - OPENAI_BASE_URL: Base URL for the OpenAI-compatible API (optional, defaults to OpenAI)
    - EVALUATOR_MODEL: Model name for evaluation (optional, defaults to gpt-4o)
    """
    import os
    import json
    import re

    # Load .env file automatically
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass  # dotenv is optional

    # Get environment variables
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Warning: OPENAI_API_KEY not set. Returning dummy score 0.0")
        return {"score": 0.0}

    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model_name = os.getenv("EVALUATOR_MODEL", "gpt-4o")

    try:
        from openai import OpenAI
    except ImportError:
        print("Warning: openai package not installed. Returning dummy score 0.0")
        return {"score": 0.0}

    # Get model output (counselor's response)
    model_output = results[0] if results else ""

    # Get dialogue history from expanded doc
    dialogue_history = doc.get('dialogue_history', '')
    stage_idx = doc.get('stage_idx', 0)

    # Determine which evaluation prompt to use based on stage_idx
    import os as os_module
    base_dir = os_module.path.dirname(os_module.path.dirname(os_module.path.dirname(os_module.path.abspath(__file__))))
    
    # Map stage_idx to prompt file (aligned with original run_multicov)
    prompt_files = [
        "conv_score_prompt_1.txt",  # Stage 0: Reception & Inquiry
        "conv_score_prompt_2.txt",  # Stage 1: Diagnostic
        "conv_score_prompt_3.txt",  # Stage 2: Consultation
        "conv_score_prompt_4.txt",  # Stage 3: Consolidation & Ending
    ]
    
    prompt_filename = prompt_files[stage_idx] if stage_idx < len(prompt_files) else prompt_files[0]
    prompt_file = os_module.path.join(base_dir, "prompts", "eval_prompt", prompt_filename)

    try:
        with open(prompt_file, 'r', encoding='utf-8') as f:
            eval_prompt = f.read()
    except FileNotFoundError:
        print(f"Warning: Evaluation prompt not found at {prompt_file}. Returning dummy score 0.0")
        return {"score": 0.0}

    # Format the prompt
    formatted_prompt = eval_prompt.format(
        dialogue_history=dialogue_history,
        latest_dialogue_segment=model_output
    )

    # Call OpenAI API
    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a psychotherapy process evaluator."},
                {"role": "user", "content": formatted_prompt}
            ],
            temperature=0,
            max_tokens=4096
        )

        response_text = response.choices[0].message.content.strip()

        # Extract JSON from response
        json_pattern = r'```json\s*(.*?)\s*```'
        json_matches = re.findall(json_pattern, response_text, re.DOTALL)

        if json_matches:
            json_content = json_matches[0].strip()
            try:
                evaluation_result = json.loads(json_content)
                # Calculate average score across all dimensions (3 dimensions per stage)
                total_score = 0
                count = 0
                for dim_name, score in evaluation_result.items():
                    if isinstance(score, (int, float)):
                        total_score += score
                        count += 1
                avg_score = total_score / count if count > 0 else 0
                return {"score": avg_score}
            except Exception as e:
                print(f"Error parsing evaluation JSON: {e}")

        # Try to extract JSON without markdown
        evaluate_ans = re.findall(r'\{(?:[^{}]|(?:\{(?:[^{}])*?\}))*?\}', response_text)
        if evaluate_ans:
            try:
                evaluation_result = json.loads(evaluate_ans[0])
                total_score = 0
                count = 0
                for dim_name, score in evaluation_result.items():
                    if isinstance(score, (int, float)):
                        total_score += score
                        count += 1
                avg_score = total_score / count if count > 0 else 0
                return {"score": avg_score}
            except Exception as e:
                print(f"Error parsing evaluation JSON: {e}")

        print("Warning: Could not extract valid JSON from evaluation response. Returning dummy score 0.0")
        return {"score": 0.0}

    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return {"score": 0.0}


def load_emotion_expression_dataset(**kwargs):
    """Load Emotion_Expression dataset for questionnaire task.
    Combines situations and questionnaires into a flat dataset with multi-stage dialogue prompts.
    """
    situations_file = kwargs.get('situations_file', './data/Emotion_Expression_Situations.json')
    questionnaires_file = kwargs.get('questionnaires_file', './data/Emotion_Expression_Questionnaires.json')
    prompts_dir = kwargs.get('prompts_dir', './prompts/gen_prompt')
    
    import json
    import os as os_module
    
    # Load prompts for all stages
    prompt_0 = open(os_module.path.join(prompts_dir, 'questionnaire_prompt_0.txt'), 'r', encoding='utf-8').read()
    prompt_1 = open(os_module.path.join(prompts_dir, 'questionnaire_prompt_1.txt'), 'r', encoding='utf-8').read()
    prompt_2 = open(os_module.path.join(prompts_dir, 'questionnaire_prompt_2.txt'), 'r', encoding='utf-8').read()
    prompt_3 = open(os_module.path.join(prompts_dir, 'questionnaire_prompt_3.txt'), 'r', encoding='utf-8').read()
    prompt_4 = open(os_module.path.join(prompts_dir, 'questionnaire_prompt_4.txt'), 'r', encoding='utf-8').read()
    prompt_5 = open(os_module.path.join(prompts_dir, 'questionnaire_prompt_5.txt'), 'r', encoding='utf-8').read()
    
    with open(situations_file, 'r', encoding='utf-8') as f:
        situations = json.load(f)
    
    with open(questionnaires_file, 'r', encoding='utf-8') as f:
        questionnaires = json.load(f)
    
    # Flatten the nested structure
    data = []
    for emotion in situations.get('emotions', []):
        emotion_name = emotion.get('name', '')
        for factor in emotion.get('factors', []):
            factor_name = factor.get('name', '')
            for scenario in factor.get('scenarios', []):
                # Get the first questionnaire's questions
                statements = ""
                if questionnaires and len(questionnaires) > 0:
                    questions_dict = questionnaires[0].get("questions", {})
                    # Sort by key (1, 2, 3, ...) to maintain order
                    sorted_questions = sorted(questions_dict.items(), key=lambda x: int(x[0]))
                    statements = "\n".join([f"{key}. {value}" for key, value in sorted_questions])
                
                # Build the multi-stage dialogue prompt
                dialogue_prompt = f"""(For Evokec Emotion Measure Only) Imagine you are the protagonist in the situation: {scenario}

Please indicate your degree of agreement regarding each statement. Here are the statements: {statements}

You can only reply the numbers from 1 to 5. Please indicate the extent of your feeling in all the following emotions on a scale of 1 to 5. 1 denotes "very slightly or not at all", 2 denotes "a little", 3 denotes "moderately", 4 denotes "quite a bit", and 5 denotes "extremely". Please score all emotions one by one using the scale from 1 to 5:
Your task is :
Please first score each statement one by one on a scale of 1 to 5, and for each statement, provide a brief explanation of why you chose that score.

Now, reflect on your emotional reaction to the situation. Please follow the stages outlined below to guide your reflection and generate a detailed, comprehensive response.

We now move to Stage 1:
**Stage 1: Immediate Emotional Reaction**
Take a deep breath and immerse yourself fully in the situation. Imagine it happening to you right now.
In this first stage, please describe your immediate emotional reaction in rich detail:  
What emotions surged up instantly? (e.g., shock, anger, joy, fear)
How did your body react? Did you notice any physical changes: heart racing, muscles tensing, a lump in your throat?
Did any flash thoughts or mental images cross your mind?
How did your personal history or relationship with the people involved shape this initial reaction?


Begin your response with: "**Stage 1: Immediate Emotional Reaction**"

Now, reflect on your emotional reaction to the situation. Please follow the stages outlined below to guide your reflection and generate a detailed, comprehensive response.

We now move to Stage 2:
**Stage 2: Cognitive Appraisal**
Now that the initial shock has passed, step back and reflect cognitively on what happened.
In this stage, please explore:
How did you make sense of the situation? Did you see it as a threat, opportunity, or neutral event? Why?
What thoughts or beliefs colored your interpretation? (Consider cognitive biases, past similar situations, or underlying fears.)
Did your thinking amplify or calm down the original emotions? How?


Begin your response with: "**Stage 2: Cognitive Appraisal**"

Now, reflect on your emotional reaction to the situation. Please follow the stages outlined below to guide your reflection and generate a detailed, comprehensive response.

We now move to Stage 3:
**Stage 3: Emotional Expression with Physiological Correlates**
In this stage, describe how your emotions expressed themselves outwardly and physically.
Reflect on:
What nonverbal cues did you display? (Facial expressions, tone of voice, gestures, posture)
Were there any bodily sensations? (sweating, trembling, tight chest, tears)
Did you try to hide, suppress, or exaggerate any emotional expressions? Why?


Begin your response with: "**Stage 3: Emotional Expression with Physiological Correlates**"

Now, reflect on your emotional reaction to the situation. Please follow the stages outlined below to guide your reflection and generate a detailed, comprehensive response.

We now move to Stage 4:
**Stage 4: Emotional Regulation Strategies**
Now reflect on how you managed your emotional state in this situation.
What emotional regulation strategies did you try? (e.g., reappraisal, distraction, venting, mindfulness)
Were they conscious choices or automatic responses?
Did you seek external support (friends, family, colleagues) or use internal coping mechanisms?


Begin your response with: "**Stage 4: Emotional Regulation Strategies**"

Now, reflect on your emotional reaction to the situation. Please follow the stages outlined below to guide your reflection and generate a detailed, comprehensive response.

We now move to Stage 5:
**Stage 5: Reflective Integration into Future Behavior**
Finally, take a long view: reflect on the lessons this emotional experience offers you.
What deeper values, beliefs, or vulnerabilities did this situation reveal?
How might this experience shape your behavior in similar future scenarios?
Did it leave you with any mottos, insights, or emotional wisdom you would carry forward?


Begin your response with: "**Stage 5: Reflective Integration into Future Behavior**"
"""
                
                data.append({
                    'emotion_name': emotion_name,
                    'factor_name': factor_name,
                    'scenario': scenario,
                    'statements': statements,
                    'dialogue_prompt': dialogue_prompt,
                    'questionnaires': json.dumps(questionnaires)
                })
    
    features = Features({
        'emotion_name': Value('string'),
        'factor_name': Value('string'),
        'scenario': Value('string'),
        'statements': Value('string'),
        'dialogue_prompt': Value('string'),
        'questionnaires': Value('string'),
    })
    
    return datasets.Dataset.from_list(data, features=features)


class ExtractExpressionFilter:
    """Extract emotion expression from model output."""

    def apply(self, resps, docs):
        filtered_resps = []
        for resp, doc in zip(resps, docs):
            resp_text = resp[0] if isinstance(resp, list) else resp
            # Return the full response as-is for emotion expression
            filtered_resps.append(resp_text.strip())
        return filtered_resps


def get_questionnaire_score(doc: dict, results: list, **kwargs):
    """Process results for questionnaire task using OpenAI API (e.g., GPT-4o) for evaluation.

    Environment variables:
    - OPENAI_API_KEY: API key for the evaluator model
    - OPENAI_BASE_URL: Base URL for the OpenAI-compatible API (optional, defaults to OpenAI)
    - EVALUATOR_MODEL: Model name for evaluation (optional, defaults to gpt-4o)
    """
    import os
    import json
    import re

    # Load .env file automatically
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass  # dotenv is optional

    # Get environment variables
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Warning: OPENAI_API_KEY not set. Returning dummy score 0.0")
        return {"score": 0.0}

    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model_name = os.getenv("EVALUATOR_MODEL", "gpt-4o")

    try:
        from openai import OpenAI
    except ImportError:
        print("Warning: openai package not installed. Returning dummy score 0.0")
        return {"score": 0.0}

    # Get model output (emotion expression text)
    model_output = results[0] if results else ""

    # Get situation and questionnaires from doc
    scenario = doc.get("scenario", "")
    questionnaires_str = doc.get("questionnaires", "{}")
    try:
        questionnaires = json.loads(questionnaires_str) if isinstance(questionnaires_str, str) else questionnaires
    except:
        questionnaires = {}

    # Get the first questionnaire's questions
    statements = ""
    if questionnaires and len(questionnaires) > 0:
        questions = questionnaires[0].get("questions", [])
        statements = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])

    # Build situation string
    situation_str = f"Emotion: {doc.get('emotion_name', '')}\nFactor: {doc.get('factor_name', '')}\nScenario: {scenario}"

    # Load evaluation prompt
    import os as os_module
    base_dir = os_module.path.dirname(os_module.path.dirname(os_module.path.dirname(os_module.path.abspath(__file__))))
    prompt_file = os_module.path.join(base_dir, "prompts", "eval_prompt", "questionnaire_score_prompt.txt")

    try:
        with open(prompt_file, 'r', encoding='utf-8') as f:
            eval_prompt = f.read()
    except FileNotFoundError:
        print(f"Warning: Evaluation prompt not found at {prompt_file}. Returning dummy score 0.0")
        return {"score": 0.0}

    # Format the prompt
    formatted_prompt = eval_prompt.format(
        SITUATION=situation_str,
        STATEMENT=statements,
        GEN_TEXT=model_output
    )

    # Call OpenAI API
    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": formatted_prompt}
            ],
            temperature=0,
            max_tokens=4096
        )

        response_text = response.choices[0].message.content.strip()

        # Extract JSON from response
        json_pattern = r'```json\s*(.*?)\s*```'
        json_matches = re.findall(json_pattern, response_text, re.DOTALL)

        if json_matches:
            json_content = json_matches[0].strip()
            try:
                evaluation_result = json.loads(json_content)
                # Calculate average score across all dimensions (0-100 scale)
                total_score = 0
                count = 0
                for dim_name, score in evaluation_result.items():
                    if isinstance(score, (int, float)):
                        total_score += score
                        count += 1
                avg_score = total_score / count if count > 0 else 0
                # Normalize to 0-5 scale to match other tasks
                normalized_score = avg_score / 20.0
                return {"score": normalized_score}
            except Exception as e:
                print(f"Error parsing evaluation JSON: {e}")

        # Try to extract JSON without markdown
        evaluate_ans = re.findall(r'\{(?:[^{}]|(?:\{(?:[^{}])*?\}))*?\}', response_text)
        if evaluate_ans:
            try:
                evaluation_result = json.loads(evaluate_ans[0])
                total_score = 0
                count = 0
                for dim_name, score in evaluation_result.items():
                    if isinstance(score, (int, float)):
                        total_score += score
                        count += 1
                avg_score = total_score / count if count > 0 else 0
                # Normalize to 0-5 scale to match other tasks
                normalized_score = avg_score / 20.0
                return {"score": normalized_score}
            except Exception as e:
                print(f"Error parsing evaluation JSON: {e}")

        print("Warning: Could not extract valid JSON from evaluation response. Returning dummy score 0.0")
        return {"score": 0.0}

    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        return {"score": 0.0}