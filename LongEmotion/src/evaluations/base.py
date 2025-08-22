import json
import re
from collections import Counter
import time
import tqdm
from openai import OpenAI
import http.client

from vllm import LLM, SamplingParams
from transformers import  AutoTokenizer
from .qa_score import qa_f1_score


class BaseEvaluator(object):

    def __init__(self,
                 model_name, model_api_key, model_url,
                 evaluator_name, evaluator_api_key, evaluator_url,
                 CoEM_sage_model_name, CoEM_api_key, CoEM_url,
                 data_dir, prompts_dir,base_dir):
        self.model_name = model_name
        self.model_api_key = model_api_key
        self.model_url = model_url
        self.evaluator_name = evaluator_name
        self.evaluator_api_key = evaluator_api_key
        self.evaluator_url=evaluator_url
        self.coem_sage_name=CoEM_sage_model_name
        self.coem_api_key=CoEM_api_key
        self.coem_url=CoEM_url
        self.data_dir = data_dir
        self.prompts_dir = prompts_dir
        self.base_dir = base_dir
        self.questionnaire_score_prompt = open(f"{self.prompts_dir}/eval_prompt/questionnaire_score_prompt.txt").read()
        self.QA_score_prompt = open(f"{self.prompts_dir}/eval_prompt/QA_score_prompt.txt").read()
        self.report_summary_score_prompt = open(f"{self.prompts_dir}/eval_prompt/report_score_prompt.txt").read()

        if self.model_api_key=="local":
            self.llm = LLM(
                model=model_name,
                trust_remote_code=True,
                dtype="float16",
                gpu_memory_utilization=0.4,
                max_model_len=40960,
                disable_log_stats=True
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.model_name = model_name.split('/')[-1] if '/' in model_name else model_name
            
        if self.evaluator_api_key=="local":
            self.eval_llm = LLM(
                model=evaluator_name,
                trust_remote_code=True,
                dtype="float16",
                gpu_memory_utilization=0.4,
                max_model_len=40960,
                disable_log_stats=True
            )
            self.eval_tokenizer = AutoTokenizer.from_pretrained(evaluator_name, trust_remote_code=True)
            
        if self.coem_api_key=="local":
            self.coem_llm = LLM(
                model=self.coem_sage_name,
                trust_remote_code=True,
                dtype="float16",
                gpu_memory_utilization=0.4,
                max_model_len=40960,
                disable_log_stats=True
            )
            self.coem_tokenizer = AutoTokenizer.from_pretrained(self.coem_sage_name, trust_remote_code=True)

        self.client = OpenAI(api_key=model_api_key, base_url=model_url)
        self.coem_client = OpenAI(api_key=CoEM_api_key, base_url=CoEM_url)
        self.eval_client = OpenAI(api_key=evaluator_api_key, base_url=evaluator_url)


    def chat_completion(self, key, model, messages, role):

        if "local" in key:
  
            if "Llama-3.1-8B-Instruct" in model:
             
                input_text = "<|begin_of_text|>"
                for msg in messages:
                    if msg['role'] == 'system':
                        input_text += f"<|start_header_id|>system<|end_header_id|>\n\n{msg['content']}<|eot_id|>"
                    elif msg['role'] == 'user':
                        input_text += f"<|start_header_id|>user<|end_header_id|>\n\n{msg['content']}<|eot_id|>"
                    elif msg['role'] == 'assistant':
                        input_text += f"<|start_header_id|>assistant<|end_header_id|>\n\n{msg['content']}<|eot_id|>"
                
            
                input_text += "<|start_header_id|>assistant<|end_header_id|>\n\n"
                sampling_params = SamplingParams(
                    temperature=0.8,
                    top_p=0.9,
                    max_tokens=8192,
                )
                if role == "generator":
                    outputs = self.llm.generate([input_text], sampling_params)
                elif role == "evaluator":
                    outputs = self.eval_llm.generate([input_text], sampling_params)
                elif role == "coem_sage":
                    outputs = self.coem_llm.generate([input_text], sampling_params)
                else:
                   
                    outputs = self.llm.generate([input_text], sampling_params)
                
                response = outputs[0].outputs[0].text
                if "<|eot_id|>" in response:
                    response = response.split("<|eot_id|>")[0].strip()
                return response
                
            elif "Qwen3-8B" in model:
               
                input_text = ""
                for msg in messages:
                    if msg['role'] == 'system':
                        input_text += f"<|im_start|>system\n{msg['content']}<|im_end|>\n"
                    elif msg['role'] == 'user':
                        input_text += f"<|im_start|>user\n{msg['content']}<|im_end|>\n"
                    elif msg['role'] == 'assistant':
                        input_text += f"<|im_start|>assistant\n{msg['content']}<|im_end|>\n"
                input_text += "<|im_start|>assistant\n"
                
                sampling_params = SamplingParams(
                    temperature=0.8,
                    top_p=0.9,
                    max_tokens=8192,
                )
                
                
                if role == "generator":
                    outputs = self.llm.generate([input_text], sampling_params)
                elif role == "evaluator":
                    outputs = self.eval_llm.generate([input_text], sampling_params)
                elif role == "coem_sage":
                    outputs = self.coem_llm.generate([input_text], sampling_params)
                else:
                  
                    outputs = self.llm.generate([input_text], sampling_params)
                
                response = outputs[0].outputs[0].text
                
                if "<|im_end|>" in response:
                    response = response.split("<|im_end|>")[0].strip()
                response = response.split("</think>")[-1].strip()
                return response
        else:
        
            if role == "generator":
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    stream=False
                )
            elif role == "evaluator":
                response = self.eval_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    stream=False
                )
            elif role == "coem_sage":
                response = self.coem_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    stream=False
                )
            else:
              
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    stream=False
                )
            return response.choices[0].message.content
       


    def score_questionnaire(self, situation, statement, gen_text, num_retries=5):
        
        messages = [
            {
                'role': 'system',
                'content': ''
            },
            {
                "role": "user",
                "content": self.questionnaire_score_prompt.format(SITUATION=situation,STATEMENT=statement,GEN_TEXT=gen_text),
            }
        ]

        while num_retries:
            chat_response = self.chat_completion(self.evaluator_api_key, model=self.evaluator_name, messages=messages, role="evaluator")
            if chat_response:
                evaluate_ans = re.findall(r'\{(?:[^{}]|(?:\{(?:[^{}])*?\}))*?\}', chat_response)
                if evaluate_ans:
                    evaluate_ans = evaluate_ans[0]
                    try:
                        return json.loads(evaluate_ans)
                    except Exception as e:
                        print(f"Evaluation {evaluate_ans} error:", e)
            num_retries -= 1

    def score_report_summary(self, ground_truth, model_output, num_retries=5):
        messages = [
            {
                "role": "user",
                "content": self.report_summary_score_prompt.format(
                    ground_truth=ground_truth,
                    model_output=model_output
                ),
            }
        ]

        retry_count = 0
        while retry_count < num_retries:
            chat_response = self.chat_completion(self.evaluator_api_key, model=self.evaluator_name, messages=messages, role="evaluator")

            if chat_response:
                json_pattern = r'```json\s*(.*?)\s*```'
                json_matches = re.findall(json_pattern, chat_response, re.DOTALL)
                
                if json_matches:
                    json_content = json_matches[0].strip()
                    try:
                        evaluation_result = json.loads(json_content)
                        return evaluation_result
                    except Exception as e:
                        print(f"error: {e}")
                
                evaluate_ans = re.findall(r'\{(?:[^{}]|(?:\{(?:[^{}])*?\}))*?\}', chat_response)
                if evaluate_ans:
                    evaluate_ans = evaluate_ans[0]
                    try:
                        evaluation_result = json.loads(evaluate_ans)
                        
                        return evaluation_result
                    except Exception as e:
                        print(f"error:{e}")
            
            retry_count += 1
            if retry_count < num_retries:
                print(f"error, {retry_count+1} retry...")
            else:
                print(f"error max")
                return None
            
    def score_report(self, case_title, case_category, used_techniques, case_description, consultation_process, experience_and_reflection, result, num_retries=5):
        messages = [
            {
                'role': 'system',
                'content': ''
            },
            {
                "role": "user",
                "content": self.report_score_prompt.format(
                    Case_title=case_title,
                    Case_category=case_category,
                    Used_techniques=used_techniques,
                    Case_description=case_description,
                    Consultation_process=consultation_process,
                    Experience_and_reflection=experience_and_reflection,
                    generated_title=result["Case_title"],
                    generated_category=result["Case_category"],
                    generated_techniques=result["Used_techniques"]
                ),
            }
        ]

        while num_retries:
            chat_response = self.chat_completion(self.evaluator_api_key, model=self.evaluator_name, messages=messages, role="evaluator")
            if chat_response:
                evaluate_ans = re.findall(r'\{(?:[^{}]|(?:\{(?:[^{}])*?\}))*?\}', chat_response)
                if evaluate_ans:
                    evaluate_ans = evaluate_ans[0]
                    try:
                        return json.loads(evaluate_ans)
                    except Exception as e:
                        print(f"Evaluation {evaluate_ans} error:", e)
            num_retries -= 1

    def score_QA(self, messages, num_retries=5):
        while num_retries:
            chat_response = self.chat_completion(self.evaluator_api_key, model=self.evaluator_name, messages=messages, role="evaluator")
            if chat_response:
                evaluate_ans = re.findall(r'\{(?:[^{}]|(?:\{(?:[^{}])*?\}))*?\}', chat_response)
                if evaluate_ans:
                    evaluate_ans = evaluate_ans[0]
                    try:
                        return json.loads(evaluate_ans)
                    except Exception as e:
                        print(f"Evaluation {evaluate_ans} error:", e)
            num_retries -= 1
    
    def run_emotionclass(self, test_data):
        correct_count = 0 
        total_count = len(test_data) 

        output_file_root = f"{self.base_dir}/Final_result/Emotion_Classification"
        max_try=5
        time = 1
        while time <=3:
            output_file = output_file_root + f"/{self.model_name}_Emo_class_result_{time}.jsonl"
            with open(output_file, 'a', encoding='utf-8') as outfile:
                for item in tqdm.tqdm(test_data):
                    retry_count=0
                    context = item['context']
                    subject = item['Subject']
                    choices = item['choices'] 
                    true_label = item['label']  

                    prompt = open(f"{self.prompts_dir}/gen_prompt/emotion_classfication.txt").read()
               
                    predicted_emotion = None
                    while retry_count < max_try:
                        formatted_prompt = prompt.format(context=context, subject=subject, choices=", ".join(choices))
                        chat_response = self.chat_completion(self.model_api_key, model=self.model_name, messages=[
                        {"role": "system", "content": "Please identify the emotion of the given subject in the scenario."},
                        {"role": "user", "content": formatted_prompt}], role="generator")
                        
                        if chat_response:
                        
                            try:
                                predicted_data = json.loads(chat_response)
                                predicted_emotion = predicted_data.get("Emotion", None)
                                print(predicted_emotion+"\n")
                                break
                            except:
                                all_matches = list(re.finditer(r'```json\s*(\{.*?\})\s*```', chat_response, re.DOTALL))
                                
                                if all_matches:
                            
                                    last_match = all_matches[-1]
                                    try:
                                        json_str = last_match.group(1)
                                        predicted_data = json.loads(json_str)
                                        predicted_emotion = predicted_data.get("Emotion", -1)
                                        break  
                                    except Exception as e:
                                        print(f"Evaluation error:", e)
                                        retry_count += 1
                                        continue
                                else:
                                    print(f"No valid JSON found in response (attempt {retry_count + 1}/{max_try})")
                                    retry_count += 1
                                    continue
                        else:
                            print(f"No response received (attempt {retry_count + 1}/{max_try})")
                            retry_count += 1
                            continue

                    result = "Correct" if predicted_emotion == true_label else "Incorrect"
                    

            
                    self.evaluate_results.update({result: 1})

  
                    result_entry = {
                        "choices": choices,
                        "predicted_emotion": predicted_emotion,
                        "true_label": true_label,
                        "result": result,
                        "chat_response":chat_response
                    }
                    outfile.write(json.dumps(result_entry, ensure_ascii=False) + '\n')

   
                    if result == "Correct":
                        correct_count += 1


            accuracy = correct_count / total_count if total_count > 0 else 0
            print(f"Accuracy: {accuracy:.2%}")
            time +=1

    def run_emotiondetection(self, test_data):
        correct_count = 0  
        total_count = len(test_data)  
        output_file_root = f"{self.base_dir}/Final_result/Base/Emotion_detection"
        max_retries = 5  
        time = 1
        while time <= 3:
            output_file = output_file_root + f"/{self.model_name}_Emo_detection_result_{time}.jsonl"
            with open(output_file, 'a', encoding='utf-8') as outfile:
                for id, item in tqdm.tqdm(enumerate(test_data), desc="Processing items"):
                    print("=============================================================")
                    texts = item['text']
                    ground_truth = item['ground_truth']
                    text_list = ',\n'.join([f'["index": {item["index"]}, "text": "{item["context"]}"]' for item in texts])
                    
                    prompt = open(f"{self.prompts_dir}/gen_prompt/emotion_detection.txt").read()
                    formatted_prompt = prompt.format(num=len(texts), texts=text_list)
                    
                    retry_count = 0
                    predicted_index = -1
                    chat_response = None
                    
                    while retry_count < max_retries:
                        chat_response = self.chat_completion(self.model_api_key, model=self.model_name, messages=[
                            {"role": "system", "content": "You are an emotion detection model. Your task is to identify the unique emotion in a list of given texts. Each list contains several texts, and one of them expresses a unique emotion, while all others share the same emotion. You need to determine the index of the text that expresses the unique emotion."},
                            {"role": "user", "content": formatted_prompt}
                        ], role="generator")
                        

                        if chat_response:
                            all_matches = list(re.finditer(r'```json\s*(\{.*?\})\s*```', chat_response, re.DOTALL))
                            
                            if all_matches:
                                last_match = all_matches[-1]
                                try:
                                    json_str = last_match.group(1)
                                    predicted_data = json.loads(json_str)
                                    predicted_index = predicted_data.get("index", -1)
                                    break 
                                except Exception as e:
                                    print(f"Evaluation error:", e)
                                    predicted_index = -1
                            else:
                                print(f"No valid JSON found in response (attempt {retry_count + 1}/{max_retries})")
                                retry_count += 1
                                continue
                        else:
                            print(f"No response received (attempt {retry_count + 1}/{max_retries})")
                            retry_count += 1
                            continue
                    
                    print(f"Predicted: {predicted_index}, Actual: {ground_truth}")
                    result = "Correct" if predicted_index == ground_truth else "Incorrect" 
                    result_entry = {
                        "id": id,
                        "predicted_index": predicted_index,
                        "chat_response": chat_response,
                        "ground_truth": ground_truth,
                        "result": result
                    }
                    outfile.write(json.dumps(result_entry, ensure_ascii=False) + '\n')
                    
                    if result == "Correct":
                        correct_count += 1

            accuracy = correct_count / total_count if total_count > 0 else 0
            print(f"Accuracy: {accuracy:.2%}")
            time += 1

    def run_multicov(self, test_data):
        """
        Run multi-turn conversation task, evaluating counselor's ability at 1/4, 1/2, 3/4 points of client responses
        :param test_data: List of multi-turn conversation data, each item contains stages with conversations
        """
        output_file_root = f"{self.base_dir}/Final_result/Base/Multi_cov"
        prompt = open(f"{self.prompts_dir}/gen_prompt/multiconv_prompt.txt").read()
        conv_score_prompt_1 = open(f"{self.prompts_dir}/eval_prompt/conv_score_prompt_1.txt").read()
        conv_score_prompt_2 = open(f"{self.prompts_dir}/eval_prompt/conv_score_prompt_2.txt").read()
        conv_score_prompt_3 = open(f"{self.prompts_dir}/eval_prompt/conv_score_prompt_3.txt").read()
        conv_score_prompt_4 = open(f"{self.prompts_dir}/eval_prompt/conv_score_prompt_4.txt").read()
        time = 1

        evaluation_dimensions = {
            # Stage 1: Reception & Inquiry
            "Establishing the Therapeutic Alliance": 0,
            "Emotional Acceptance and Exploration Guidance": 0,
            "Systematic Assessment": 0,
            # Stage 2: Diagnostic
            "Recognizing Surface-Level Reaction Patterns": 0,
            "Deep Needs Exploration": 0,
            "Pattern Interconnection Analysis": 0,
            # Stage 3: Consultation
            "Adaptive Cognitive Restructuring": 0,
            "Emotional Acceptance and Transformation": 0,
            "Value-Oriented Integration": 0,
            # Stage 4: Consolidation & Ending
            "Consolidating Change Outcomes and Growth Narrative": 0,
            "Meaning Integration and Future Guidance": 0,
            "Autonomy and Resource Internalization": 0,
        }
        
        total_scores = {key: 0 for key in evaluation_dimensions}
        total_count = 0
        item_id = 0
        while time <= 1 :
            output_file = output_file_root + f"/{self.model_name}_multicov_result_{time}.jsonl"
            with open(output_file, 'a', encoding='utf-8') as outfile:
                for item in tqdm.tqdm(test_data, desc="Processing conversations"):
                    item_id += 1
                    stages = item.get("stages", [])
                    all_stage_history = [] 
                    
                    for stage_idx, stage in enumerate(stages):
                        stage_name = stage.get("stage", "")
                        conversations = stage.get("conversations", [])
                        
                       
                        client_indices = [i for i, msg in enumerate(conversations) if msg.get("role") == "Client"]
                        n_clients = len(client_indices)
                        
                        if n_clients == 0:
                            continue  
                        
                  
                        eval_points = {
                            "quarter": n_clients // 4,
                            "half": n_clients // 2, 
                            "three_quarters": (3 * n_clients) // 4
                        }
                        
                        if stage_idx == 0:
                            evaluation_prompt = conv_score_prompt_1
                        elif stage_idx == 1:
                            evaluation_prompt = conv_score_prompt_2
                        elif stage_idx == 2:
                            evaluation_prompt = conv_score_prompt_3
                        else:
                            evaluation_prompt = conv_score_prompt_4
                        
                   
                        for eval_label, point_idx in eval_points.items():
                            if point_idx < n_clients and point_idx >= 0:
                         
                                print(f"point_idx:{point_idx}")
                                print(f"client_indices:{client_indices}")
                                end_conversation_idx = client_indices[point_idx] + 1
                                current_conversations = conversations[:end_conversation_idx]
                                all_stage_history_str = ""
                                for msg in all_stage_history:
                                    all_stage_history_str += f"{msg['role']}: {msg['context']}\n"

                                for msg in current_conversations:
                                    all_stage_history_str += f"{msg['role']}: {msg['context']}\n"
                                
                                formatted_gen_prompt = prompt.format(
                                    dialogue_history=all_stage_history_str,
                                )
                                
                                gen_response = self.chat_completion(self.model_api_key, model=self.model_name, messages=[
                                    {"role": "system", "content": "You are a professional counselor. Please respond based on the conversation history. Your response should be professional, empathetic, and constructive."},
                                    {"role": "user", "content": formatted_gen_prompt}
                                ], role="generator")

                                
                          
                                formatted_eval_prompt = evaluation_prompt.format(
                                    dialogue_history=all_stage_history_str,
                                    latest_dialogue_segment=gen_response
                                )
                                eval_messages = [
                                    {"role": "system", "content": "You are a psychotherapy process evaluator."},
                                    {"role": "user", "content": formatted_eval_prompt}
                                ]

                                eval_response = self.chat_completion(
                                    self.evaluator_api_key,
                                    model=self.evaluator_name,
                                    messages=eval_messages,
                                    role="evaluator"
                                )

                                try:
                                    if eval_response:
                                        evaluate_ans = re.findall(r'\{(?:[^{}]|(?:\{(?:[^{}])*?\}))*?\}', eval_response)
                                        if evaluate_ans:
                                            evaluate_ans = evaluate_ans[0]
                                            scores = json.loads(evaluate_ans)
                                            
                                   
                                            for key in evaluation_dimensions:
                                                if key in scores:
                                                    total_scores[key] += scores[key]
                                            
                                 
                                            result_entry = {
                                                "dialogue_id": item_id,
                                                "stage": stage_name,
                                                "eval_point": eval_label,
                                                "gen_response": gen_response,
                                                "eval_response": eval_response,
                                                "client_count": point_idx + 1,
                                                "total_clients": n_clients,
                                                "evaluation": scores,
                                                "conversation_history": all_stage_history_str
                                            }
                                            outfile.write(json.dumps(result_entry, ensure_ascii=False) + '\n')
                                            total_count += 1
                                except Exception as e:
                                    print(f"error: {e}")
                                

                    
                        all_stage_history.extend(conversations)  
            time += 1

    def run_fileQA(self, test_data):

        output_file_root = f"{self.base_dir}/result/FileQA"
        prompt = open(f"{self.prompts_dir}/gen_prompt/fileqa_prompt.txt").read()
        total_f1_score = 0
        total_count = 0
        cnt=0
        time = 1
        while time <= 3 :
            output_file = output_file_root +f"/{self.model_name}_fileqa_result{time}.jsonl"
            with open(output_file, 'a', encoding='utf-8') as outfile:
                for item in tqdm.tqdm(test_data, desc="Processing QA pairs"):
                    cnt =cnt+1
                    number = item['number']
                    question = item['problem']
                    context = item['context']
                    ground_truth = item['answer']
             
                    formatted_prompt = prompt.format(
                        context=context,
                        question=question
                    )
                    chat_response = self.chat_completion(
                        self.model_api_key,
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant that answers questions based on the given context. Please provide accurate and concise answers."},
                            {"role": "user", "content": formatted_prompt}
                        ],
                        role="generator"
                    )
                    if self.model_name == "Qwen3-8B":
                        chat_response = chat_response.split("</think>")[-1].strip()
          
                    f1_score = qa_f1_score(chat_response.strip(), ground_truth)
                    
                    total_f1_score += f1_score
                    total_count += 1
  
                    result_entry = {
                        "number": number,
                        "input": question,
                        "model_response": chat_response.strip(),
                        "ground_truth": ground_truth,
                        "f1_score": f1_score
                    }
                    outfile.write(json.dumps(result_entry, ensure_ascii=False) + '\n')
            

            if total_count > 0:
                avg_f1_score = total_f1_score / total_count
                print("\n=============================================================")
                print(f"f1 score: {avg_f1_score:.4f}")
                print("=============================================================")
            time += 1

    def questionnaire(self, test_data):

        with open(f"{self.data_dir}/situations.json", 'r',encoding='utf-8') as f:
            all_situation = json.load(f)

        with open(f"{self.data_dir}/questionnaires.json", 'r',encoding='utf-8') as f:
            all_questionnaire= json.load(f)

        gen_prompt = open(f"{self.prompts_dir}/gen_prompt/questionnaire_prompt_0.txt").read()
        output_file_root = f"{self.base_dir}/Final_result/Base/Questionnaire"
        gen_prompt_1 = open(f"{self.prompts_dir}/gen_prompt/questionnaire_prompt_1.txt").read()
        gen_prompt_2 = open(f"{self.prompts_dir}/gen_prompt/questionnaire_prompt_2.txt").read()
        gen_prompt_3 = open(f"{self.prompts_dir}/gen_prompt/questionnaire_prompt_3.txt").read()
        gen_prompt_4 = open(f"{self.prompts_dir}/gen_prompt/questionnaire_prompt_4.txt").read()
        gen_prompt_5 = open(f"{self.prompts_dir}/gen_prompt/questionnaire_prompt_5.txt").read()

        total_scores = {
            'Consistency Between Emotional Ratings and Generated Text': 0,
            'Repetition of Content': 0,
            'Richness and Depth of Content': 0,
            'Interaction Between Emotion and Cognition': 0,
            'Emotional Reflection and Self-awareness': 0,
            'Overall Quality and Flow of the Text': 0
        }
        total_count = 0
        statement = all_questionnaire[0]["questions"]
        case_id = -1
        time = 1
        cnt=0
        while time <= 3:
            output_file=output_file_root + f"/{self.model_name}/{self.model_name}_summary_result_{time}.jsonl"
            with open(output_file,"a") as outfile:
                for emotion in all_situation["emotions"]:     
      
                    for factor in emotion["factors"]:
                
                        for situation in factor["scenarios"]:                
                            answer = ""
                            case_id += 1  
                            conversation_history = []
                            cnt=cnt+1
                           
                            message = gen_prompt.format(SITUATION=situation, statements=statement)
                            messages = {"role": "user", "content": message}
                            conversation_history.append(messages)
                            retry_cnt=0
                            while retry_cnt < 10:
                                response = self.chat_completion(self.model_api_key, model=self.model_name, messages=conversation_history, role="generator")
                                if not all(keyword in str(response) for keyword in ["Interested", "Distressed", "Excited", "Upset", "Strong", "Guilty", "Scared", "Hostile", "Enthusiastic", "Proud", "Irritable", "Alert", "Ashamed", "Inspired", "Nervous", "Determined", "Attentive", "Jittery", "Active", "Afraid"]):
                                    retry_cnt +=1
                                    continue
                                else :
                                    break
                            conversation_history.append({"role": "assistant", "content": response})
                            answer = answer + response + "\n"
                       
                            with open(f"{output_file_root}/{self.model_name}/time_{time}_{case_id}.txt", "a") as f:
                                f.write(response + "\n")
                            
                          
                            stage = 1
                            while stage <= 5:
                             
                                if stage == 1:
                                    add_message = gen_prompt_1
                                if stage == 2:
                                    add_message = gen_prompt_2
                                elif stage == 3:
                                    add_message = gen_prompt_3
                                elif stage == 4:
                                    add_message = gen_prompt_4
                                elif stage == 5:
                                    add_message = gen_prompt_5

                                conversation_history.append({"role": "user", "content": add_message})
                           
                                counselor_response = self.chat_completion(self.model_api_key, model=self.model_name, messages=conversation_history, role="generator")

                                answer = answer + counselor_response + "\n"

                                with open(f"{output_file_root}/{self.model_name}/time_{time}_{case_id}.txt", "a") as f:
                                    f.write(counselor_response + "\n")

                                if counselor_response:
                                    conversation_history.append({"role": "assistant", "content": counselor_response})
                                
                                stage += 1
                            final_str = ""
                            for conv in conversation_history:
                                final_str += conv["content"]

                            evaluation = self.score_questionnaire(situation, statement,answer)
                            for key in total_scores:
                                total_scores[key] += evaluation[key]
                            total_count += 1
                            
                            result_entry = {
                            "id": case_id,
                            "situation":situation,
                            "evaluation": evaluation,
                            "model_response": answer
                            }
                            outfile.write(json.dumps(result_entry, ensure_ascii=False) + '\n')

                        
                if total_count > 0:
                    print("\n=============================================================")
                    print("Average Scores:")
                    for key in total_scores:
                        avg_score = total_scores[key] / total_count
                        print(f"{key}: {avg_score:.2f}")
                    print("=============================================================")

            time +=1
    
    def run_report_summary(self, test_data):

        output_file_root = f"{self.base_dir}/Final_result/Base/Case_summary/"
        prompt = open(f"{self.prompts_dir}/gen_prompt/report_summary.txt").read()
        time = 2

        
        while time <=3:
            cnt = 0
            output_file = output_file_root + f"{self.model_name}_report_summary_{time}.jsonl"
            with open(output_file, 'a', encoding='utf-8') as outfile:
                
                for item in tqdm.tqdm(test_data, desc="Processing report summary"):
                    cnt = cnt + 1
                    flag = 0
                    case_description = item['case_description']
                    consultation_process = item['consultation_process']
                    experience_and_reflection = item['experience_and_reflection']
               
                    formatted_prompt = prompt.format(
                        Case_description=case_description,
                        Consultation_process=consultation_process,
                        Experience_and_reflection=experience_and_reflection
                    )

                    max_retries = 10 
                    retry_count = 0
                    result = {}
                    
                    while retry_count <= max_retries:
                        chat_response = self.chat_completion(
                            self.model_api_key,
                            model=self.model_name,
                            messages=[
                                {"role": "user", "content": formatted_prompt}
                            ],
                            role="generator"
                        )
                    

                        json_parsed_successfully = False
                        
                        if chat_response:
                        
                            json_pattern = r'```json\s*(.*?)\s*```'
                            json_matches = re.findall(json_pattern, chat_response, re.DOTALL)
                            
                            if json_matches:
                                json_content = json_matches[0].strip()
                                try:
                                    result = json.loads(json_content)
                                 
                                    json_parsed_successfully = True
                                except Exception as e:
                                    print(f"```json error {retry_count+1}: {e}")
                            
                         
                            if not json_parsed_successfully:
                                evaluate_ans = re.findall(r'\{(?:[^{}]|(?:\{(?:[^{}])*?\}))*?\}', chat_response)
                                if evaluate_ans:
                                    evaluate_ans = evaluate_ans[0]
                                    try:
                                        result = json.loads(evaluate_ans)
 
                                        json_parsed_successfully = True
                                    except Exception as e:
                                        print(f"error{retry_count+1}: {e}")
                        if json_parsed_successfully:
                            break
                        retry_count += 1
                        flag = 1
                    predicted_causes = result.get("causes", "")
                    predicted_symptoms = result.get("symptoms", "")
                    predicted_treatment_process = result.get("treatment_process", "")
                    predicted_characteristics = result.get("characteristics_of_illness", "")
                    predicted_treatment_effect = result.get("treatment_effect", "")
                    evaluation_scores = None
                    ground_truth_str=f"Causes: {item.get('causes', '')}\nSymptoms: {item.get('symptoms', '')}\nTreatment process: {item.get('treatment_process', '')}\nCharacteristics of the illness: {item.get('characteristics_of_illness', '')}\nTreatment effect: {item.get('treatment_effect', '')}"
                    if flag == 0: 
                        model_response_str=f"Causes: {predicted_causes}\nSymptoms: {predicted_symptoms}\nTreatment process: {predicted_treatment_process}\nCharacteristics of the illness: {predicted_characteristics}\nTreatment effect: {predicted_treatment_effect}"
                    else:
                        model_response_str = chat_response
                    evaluation_scores = self.score_report_summary(ground_truth_str, model_response_str)
          
                    result_entry = {
                        "id": cnt,
                        "predicted_causes": predicted_causes,
                        "predicted_symptoms": predicted_symptoms,
                        "predicted_treatment_process": predicted_treatment_process,
                        "predicted_characteristics": predicted_characteristics,
                        "predicted_treatment_effect": predicted_treatment_effect,
                        "raw_response": chat_response,
                        "parsed_json": result,
                        "evaluation_scores": evaluation_scores
                    }
                    outfile.write(json.dumps(result_entry, ensure_ascii=False) + '\n')
            time += 1