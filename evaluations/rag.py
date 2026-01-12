import json
import re
import tqdm
import numpy as np
import faiss
from langchain_text_splitters import RecursiveCharacterTextSplitter
from FlagEmbedding import BGEM3FlagModel
from .base import BaseEvaluator
from transformers import AutoTokenizer
from .qa_score import qa_f1_score


class RAGEvaluator(BaseEvaluator):

    def __init__(self,
                 model_name,
                 model_api_key,
                 model_url,
                 evaluator_name,
                 evaluator_api_key,
                 evaluator_url,
                 CoEM_sage_model_name,
                 CoEM_api_key,
                 CoEM_url,
                 data_dir,
                 prompts_dir,
                 base_dir,
                 task):
        super().__init__(model_name, model_api_key, model_url, evaluator_name, evaluator_api_key, evaluator_url, CoEM_sage_model_name, CoEM_api_key, CoEM_url, data_dir, prompts_dir, base_dir, task)
        self.bge_model = BGEM3FlagModel('~/Bge-m3', use_fp16=True)
        self.tokenizer = AutoTokenizer.from_pretrained("~/Llama-3.1-8B-Instruct", trust_remote_code=True)
        self.text_splitter_128 = RecursiveCharacterTextSplitter(
            chunk_size=128,  
            chunk_overlap=50,
            length_function=lambda x: len(self.tokenizer.encode(x)),
            is_separator_regex=False,
        )
        self.text_splitter_256 = RecursiveCharacterTextSplitter(
            chunk_size=256,  
            chunk_overlap=50,
            length_function=lambda x: len(self.tokenizer.encode(x)),
            is_separator_regex=False,
        )
        self.text_splitter_512 = RecursiveCharacterTextSplitter(
            chunk_size=512,  
            chunk_overlap=50,
            length_function=lambda x: len(self.tokenizer.encode(x)),
            is_separator_regex=False,
        )
        self.task = task


    def retrieve_with_bge(self, query, context_chunks, top_k):


        if (len(context_chunks) <= top_k):
            return context_chunks
        query_vec = self.bge_model.encode(query)['dense_vecs'][None, :].astype('float32')
        

        context_vecs = [self.bge_model.encode(chunk)['dense_vecs'][None, :].astype('float32') for chunk in context_chunks]
        context_vecs = np.stack(context_vecs, axis=0).squeeze()
        
  
        id_list = np.array([n for n in range(len(context_vecs))])
        index = faiss.IndexIDMap(faiss.IndexFlatIP(query_vec.shape[1]))
        index.add_with_ids(context_vecs, id_list)
        

        query_vec = np.array(query_vec)
        near_ids = index.search(query_vec, top_k)[1][0].tolist()
        

        sorted_ids = sorted(near_ids)
        return [context_chunks[i] for i in sorted_ids]
    
    def run_report_summary(self, test_data):

        output_file_root = f"{self.base_dir}/Final_result/RAG/{self.task}/"
        prompt = open(f"{self.prompts_dir}/aug_prompt/Case_Summary/report_summary.txt").read()
        query_cause = open(f"{self.prompts_dir}/aug_prompt/Case_Summary/summary_query_cause.txt").read()
        query_symptoms = open(f"{self.prompts_dir}/aug_prompt/Case_Summary/summary_query_symptoms.txt").read()
        query_process = open(f"{self.prompts_dir}/aug_prompt/Case_Summary/summary_query_process.txt").read()
        query_characteristics = open(f"{self.prompts_dir}/aug_prompt/Case_Summary/summary_query_Characteristics .txt").read()
        query_effect = open(f"{self.prompts_dir}/aug_prompt/Case_Summary/summary_query_effect.txt").read()
        time = 1
        
        while time <= 3:
            cnt = 0
            output_file = output_file_root + f"{self.model_name}_report_summary_result_{time}.jsonl"
            with open(output_file, 'a', encoding='utf-8') as outfile:
                for item in tqdm.tqdm(test_data, desc="Processing report summary"):
                    cnt = cnt + 1
                    flag = 0
                    case_description = item['case_description']
                    consultation_process = item['consultation_process']
                    experience_and_reflection = item['experience_and_reflection']
                    
                    # 1 chunking
                    if isinstance(case_description, list):
                        case_description = ' '.join(str(x) for x in case_description)
                    else:
                        case_description = str(case_description)
                        
                    if isinstance(consultation_process, list):
                        consultation_process = ' '.join(str(x) for x in consultation_process)
                    else:
                        consultation_process = str(consultation_process)
                        
                    if isinstance(experience_and_reflection, list):
                        experience_and_reflection = ' '.join(str(x) for x in experience_and_reflection)
                    else:
                        experience_and_reflection = str(experience_and_reflection)
                    
                   
                    full_context = f"Case description:{case_description}\nConsultation process:{consultation_process}\nExperience and reflection:{experience_and_reflection}"
                    


                    
                    chunks_first = self.text_splitter_128.split_text(full_context)

                    # 2 initial ranking
                    
              
                    retrieval_results = {}
                    
                  
                    causes_chunks = self.retrieve_with_bge(query_cause, chunks_first, top_k=4)
                    retrieval_results['causes'] = causes_chunks
                    
             
                    symptoms_chunks = self.retrieve_with_bge(query_symptoms, chunks_first, top_k=4)
                    retrieval_results['symptoms'] = symptoms_chunks
            

                    process_chunks = self.retrieve_with_bge(query_process, chunks_first, top_k=4)
                    retrieval_results['treatment_process'] = process_chunks
                    
                  
                    characteristics_chunks = self.retrieve_with_bge(query_characteristics, chunks_first, top_k=4)
                    retrieval_results['characteristics_of_illness'] = characteristics_chunks
                    
                    
                    effect_chunks = self.retrieve_with_bge(query_effect, chunks_first, top_k=4)
                    retrieval_results['treatment_effect'] = effect_chunks   
                            
                    
                    retrieval_chunks_final = ""
                    for dimension in ['causes', 'symptoms', 'treatment_process', 'characteristics_of_illness', 'treatment_effect']:
                        retrieval_chunks_final += f"\n=== {dimension.upper()} RELATED CONTENT ===\n"
                        for i, pack in enumerate(retrieval_results[dimension]):
                            retrieval_chunks_final += f"Content {i+1}:\n{pack}\n\n"
                        retrieval_chunks_final += "-" * 50 + "\n"

                    
            
                    formatted_prompt = prompt.format(
                        Case_description=case_description,
                        Consultation_process=consultation_process,
                        Experience_and_reflection=experience_and_reflection,
                        retrieval_chunks_final=retrieval_chunks_final
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
                                    print(f"```json error (try{retry_count+1}): {e}")
                            
                
                            if not json_parsed_successfully:
                                evaluate_ans = re.findall(r'\{(?:[^{}]|(?:\{(?:[^{}])*?\}))*?\}', chat_response)
                                if evaluate_ans:
                                    evaluate_ans = evaluate_ans[0]
                                    try:
                                        result = json.loads(evaluate_ans)
       
                                        json_parsed_successfully = True
                                    except Exception as e:
                                        print(f"error {retry_count+1}: {e}")
                        
                       
                        if json_parsed_successfully:
                            break
                        
                     
                        retry_count += 1
                        if retry_count <= max_retries:
                            print(f"JSON error {retry_count+1} ...")
                        else:
                            print(f"max({max_retries+1} )")
                           
                            flag = 1

                   
                    predicted_causes = result.get("causes", "")
                    predicted_symptoms = result.get("symptoms", "")
                    predicted_treatment_process = result.get("treatment_process", "")
                    predicted_characteristics = result.get("characteristics_of_illness", "")
                    predicted_treatment_effect = result.get("treatment_effect", "")

                
                    evaluation_scores = None
                    ground_truth_str=f"Causes: {item.get('causes', '')}\nSymptoms: {item.get('symptoms', '')}\nTreatment process: {item.get('treatment_process', '')}\nCharacteristics of the illness: {item.get('characteristics_of_illness', '')}\nTreatment effect: {item.get('treatment_effect', '')}"
                    if flag == 0 :
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

    def run_emotionclass(self, test_data):
            
        correct_count = 0 
        total_count = len(test_data) 
        print(f"model_name: {self.model_name}")
        output_file_root = f"{self.base_dir}/Final_result/RAG/{self.task}"
        max_try=5
        prompt_gen = open(f"{self.prompts_dir}/aug_prompt/emo_class_gen.txt").read()
        time = 1
        while time <= 3:
            output_file = output_file_root+ f"/{self.model_name}_Emo_class_result_subject_origin_{time}_subject.jsonl"
            with open(output_file, 'a', encoding='utf-8') as outfile:
                for item in tqdm.tqdm(test_data):
                    retry_count=0
                    context = item['content']
                    subject = item['subject']
                    choices = item['choices'] 
                    true_label = item['label']  

           
                    predicted_emotion = None
                    # 1 chunking 
                    chunks = self.text_splitter_128.split_text(context)

                    first_chunks = self.retrieve_with_bge(subject,chunks,top_k=1)

                    second_chunks_str="\n".join(first_chunks)
                    while retry_count < max_try:
                        formatted_prompt = prompt_gen.format(context=context, subject=subject, choices=", ".join(choices),second_chunks=second_chunks_str)
                        chat_response = self.chat_completion(self.model_api_key, model=self.model_name, messages=[
                        {"role": "system", "content": "Please identify the emotion of the given subject in the scenario."},
                        {"role": "user", "content": formatted_prompt},], role="generator")

                        
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

    def questionnaire(self, test_data):

        with open(f"{self.data_dir}/Emotion_Expression_Situations.json", 'r',encoding='utf-8') as f:
            all_situation = json.load(f)

        with open(f"{self.data_dir}/Emotion_Expression_Questionnaires.json", 'r',encoding='utf-8') as f:
            all_questionnaire= json.load(f)

        gen_prompt = open(f"{self.prompts_dir}/gen_prompt/questionnaire_prompt_0.txt").read()
        output_file_root = f"{self.base_dir}/Final_result/RAG/{self.task}/{self.model_name}/"
        gen_prompt_1 = open(f"{self.prompts_dir}/aug_prompt/questionnaire_prompt_1.txt").read()
        gen_prompt_2 = open(f"{self.prompts_dir}/aug_prompt/questionnaire_prompt_2.txt").read()
        gen_prompt_3 = open(f"{self.prompts_dir}/aug_prompt/questionnaire_prompt_3.txt").read()
        gen_prompt_4 = open(f"{self.prompts_dir}/aug_prompt/questionnaire_prompt_4.txt").read()
        gen_prompt_5 = open(f"{self.prompts_dir}/aug_prompt/questionnaire_prompt_5.txt").read()
        query_2 = open(f"{self.prompts_dir}/aug_prompt/questionnaire_query_2.txt").read()
        query_3 = open(f"{self.prompts_dir}/aug_prompt/questionnaire_query_3.txt").read()
        query_4 = open(f"{self.prompts_dir}/aug_prompt/questionnaire_query_4.txt").read()
        query_5 = open(f"{self.prompts_dir}/aug_prompt/questionnaire_query_5.txt").read()
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
        output_file = output_file_root + f"{self.model_name}_result.jsonl"
        with open(output_file,"a") as outfile:
            for emotion in all_situation["emotions"]:     
         
                for factor in emotion["factors"]:
             
                    for situation in factor["scenarios"]:                
                        answer = ""
                        case_id += 1 
                        conversation_history = []
                        retry_cnt=0
                   
                        message = gen_prompt.format(SITUATION=situation, statements=statement)
                        messages = {"role": "user", "content": message}
                        conversation_history.append(messages)
                        while retry_cnt < 5:
                            response = self.chat_completion(self.model_api_key, model=self.model_name, messages=conversation_history, role="generator")
                            if not all(keyword in str(response) for keyword in ["Interested", "Distressed", "Excited", "Upset", "Strong", "Guilty", "Scared", "Hostile", "Enthusiastic", "Proud", "Irritable", "Alert", "Ashamed", "Inspired", "Nervous", "Determined", "Attentive", "Jittery", "Active", "Afraid"]):
                                retry_cnt +=1
                                continue
                            else :
                                break

                        conversation_history.append({"role": "assistant", "content": response})
                        answer = answer + response + "\n"
                   
                        with open(output_file_root+f"response_Generation_{case_id}.txt", "a") as f:
                            f.write(response + "\n")
                  
                        stage = 1
                        while stage <= 5:
                   
                            if stage == 1:
                                add_message = gen_prompt_1
                                conversation_history.append({"role": "user", "content": add_message})
                                
                            
                                counselor_response = self.chat_completion(self.model_api_key, model=self.model_name, messages=conversation_history, role="generator")
                                answer = answer + counselor_response +"\n"

                             
                                with open(output_file_root+f"response_Generation_{case_id}.txt", "a") as f:
                                    f.write(counselor_response + "\n")

                                conversation_history.append({"role": "assistant", "content": counselor_response})
                            else:
                                if stage == 2:
                                    add_message = gen_prompt_2
                                    query=query_2                                   
                                    n=2
                                elif stage == 3:
                                    add_message = gen_prompt_3
                                    query=query_3                                  
                                    n=4
                                elif stage == 4:
                                    add_message = gen_prompt_4
                                    query=query_4                               
                                    n=4
                                elif stage == 5:
                                    add_message = gen_prompt_5
                                    query=query_5    
                                    n=4
                              
                                chunks_first = self.text_splitter_128.split_text(answer)
                              
                                retrieve_first = self.retrieve_with_bge(query,chunks_first,top_k=n)
                             
                                final_chunks_str = "\n".join([f"Chunk{i+1}:\n{chunk}" for i, chunk in enumerate(retrieve_first)])

                                add_message_format=add_message.format(retrieved_chunk=final_chunks_str)
                                messages = {"role": "user", "content": add_message_format}
                                conversation_history.append(messages)
                                counselor_response = self.chat_completion(self.model_api_key, model=self.model_name, messages=conversation_history, role="generator")
                                answer = answer + counselor_response + "\n"

                              
                                with open(output_file_root+f"response_Generation_{case_id}.txt", "a") as f:
                                    f.write(counselor_response + "\n")

                                conversation_history.append({"role": "assistant", "content": counselor_response})
                            
                            stage += 1

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

    def run_fileQA(self, test_data):

        output_file_root = f"{self.base_dir}/Final_result/Origin_rag/{self.task}/"
        prompt_gen = open(f"{self.prompts_dir}/aug_prompt/qa_gen.txt").read()      
   
        total_f1_score = 0
        total_count = 0
        cnt=0
        time = 1
        while time <=1 :
            output_file = output_file_root + f"{self.model_name}_fileqa_result_512_16_{time}.jsonl"
            with open(output_file, 'a', encoding='utf-8') as outfile:
                for item in tqdm.tqdm(test_data, desc="Processing QA pairs"):
                    cnt += 1
                    print("=============================================================")
                    
             
                    number = item['number']
                    question = item['problem']
                    context = item['context']
                    ground_truth = item['answer']
                    chunks = self.text_splitter_512.split_text(context)
                    query = question
                    retrieved_chunks = self.retrieve_with_bge(query, chunks, top_k=16)
                    final_chunks_str = "\n".join([f"Chunk{i+1}:\n{chunk}" for i, chunk in enumerate(retrieved_chunks)])
                    formatted_gen_prompt = prompt_gen.format(
                        context = context , 
                        final_chunks = final_chunks_str,
                        question = question
                    )
                    chat_response = self.chat_completion(
                        self.model_api_key,
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant that answers questions based on the given context. Please provide accurate and concise answers."},
                            {"role": "user", "content": formatted_gen_prompt}
                        ],
                        role="generator"
                    )
                    if "Qwen" in self.model_name:
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
                print(f"F1: {avg_f1_score:.4f}")
                print("=============================================================")
            time += 1

    def run_multicov(self, test_data):
        """
        Run multi-turn conversation task, evaluating counselor's ability at 1/4, 1/2, 3/4 points of client responses
        :param test_data: List of multi-turn conversation data, each item contains stages with conversations
        """
        output_file_root = f"{self.base_dir}/Final_result/RAG/{self.task}/"
        gen_prompt = open(f"{self.prompts_dir}/aug_prompt/multi_conv_gen_origin_rag.txt").read()
        conv_score_prompt_4 = open(f"{self.prompts_dir}/eval_prompt/conv_score_prompt_4.txt").read()
        item_id = 0
        time = 1 
        N = 4
        while time <=3 :
            output_file = output_file_root + f"{self.model_name}_multicov_result_3rounds_{time}_N{N}.jsonl"
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
                        

                        evaluation_prompt = conv_score_prompt_4
                        
                      
                        for eval_label, point_idx in eval_points.items():
                            if point_idx < n_clients and point_idx >= 0:
                                if stage_idx!=3:
                                    continue
                                current_conversations = conversations[:client_indices[point_idx]-2]
                                client_last_rounds = conversations[client_indices[point_idx]-2:client_indices[point_idx]+1]
                                all_stage_history_str = ""

                                for msg in all_stage_history:
                                    all_stage_history_str += f"{msg['role']}: {msg['context']}\n"
                                for msg in current_conversations:
                                    all_stage_history_str += f"{msg['role']}: {msg['context']}\n"
                           
                                query = ""
                                for msg in client_last_rounds:
                                    query += f"{msg['role']}: {msg['context']}\n"
                                chunks = self.text_splitter_128.split_text(all_stage_history_str)

                                retrieved_chunks = self.retrieve_with_bge(query, chunks, top_k=N)
                                first_chunks = "\n\n".join([
                                    f"[Chunk {i+1}]\n{chunk}" 
                                    for i, chunk in enumerate(retrieved_chunks)
                                ])
                    
                       
                                formatted_gen_prompt = gen_prompt.format(
                                    dialogue_history=all_stage_history_str,
                                    latest_reply=query,
                                    final_chunks=first_chunks,
                                )

                                
                                gen_response = self.chat_completion(self.model_api_key, model=self.model_name, messages=[
                                    {"role": "system", "content": "You are a professional counselor. Please respond based on the conversation history. Your response should be professional, empathetic, and constructive."},
                                    {"role": "user", "content": formatted_gen_prompt}
                                ], role="generator")

                                all_stage_history_str += query
                    
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
                                            result_entry = {
                                                "dialogue_id": item_id,
                                                "stage": stage_name,
                                                "gen_response": gen_response,
                                                "eval_response": eval_response,
                                                "eval_point": eval_label,
                                                "client_count": point_idx + 1,
                                                "total_clients": n_clients,
                                                "evaluation": scores,
                                                "conversation_history": all_stage_history_str
                                            }
                                            outfile.write(json.dumps(result_entry, ensure_ascii=False) + '\n')
                                except Exception as e:
                                    print(f"error: {e}")
                        all_stage_history.extend(conversations)   
            time +=1      
                    

    def run_emotiondetection(self, test_data):
        gen_prompt = open(f"{self.prompts_dir}/aug_prompt/emo_detect_gen.txt").read()
        correct_count = 0  
        total_count = len(test_data)
        output_file_root = f"{self.base_dir}/Final_result/RAG/{self.task}"
        max_retries=5
        time = 1
        while time <= 3 :
            output_file = output_file_root + f"/{self.model_name}_Emo_detection_result_{time}.jsonl"
            with open(output_file, 'a', encoding='utf-8') as outfile:
                for id, item in tqdm.tqdm(enumerate(test_data), desc="Processing items"):
                    texts = item['text']
                    ground_truth = item['ground_truth']
                    text_list = ',\n'.join([f'["index": {seg["index"]}, "text": "{seg["context"]}"]' for seg in texts])
                    packs=[]
                 
                    chunks = [item['context'] for item in texts]
                    for i, chunk in enumerate(chunks):
                        pack = {
                            "index": i,
                            "content": chunk
                        }
                        packs.append(pack)
                               

                    if len(packs) > 8:
                        pack_contents = [
                            pack["content"]
                            for pack in packs
                        ]
                        
                     
                        pack_embeddings = []
                        for content in pack_contents:
                            embedding = self.bge_model.encode(content)['dense_vecs']
                            pack_embeddings.append(embedding)

                        pack_embeddings = np.array(pack_embeddings)

                       
                        similarity_matrix = np.dot(pack_embeddings, pack_embeddings.T)

                   
                        avg_similarities = []
                        for i in range(len(packs)):
                            similarities_with_others = np.concatenate([similarity_matrix[i][:i], similarity_matrix[i][i+1:]])
                            avg_similarity = np.mean(similarities_with_others)
                            avg_similarities.append((i, avg_similarity))

               
                        avg_similarities.sort(key=lambda x: x[1])
                        lowest_similarity_indices = [idx for idx, _ in avg_similarities[:8]]
                        selected_packs = [packs[i] for i in lowest_similarity_indices]


                    else:
           
                        selected_packs = packs
                    pack_texts = [f'["index": {seg["index"]}, "text": "{seg["content"]}"]' for seg in selected_packs]
         
                    selected_text_list_str = "\n\n".join([pack for pack in pack_texts])
                    

                    formatted_gen_prompt = gen_prompt.format(
                            num=len(texts), 
                            texts=text_list,
                            rag_reference=selected_text_list_str
                        )

                    retry_count = 0
                    predicted_index = -1
                    chat_response = None

                    while retry_count < max_retries:
                        chat_response = self.chat_completion(self.model_api_key, model=self.model_name, messages=[
                            {"role": "system", "content": "You are an emotion detection model. Your task is to identify the unique emotion in a list of given texts. Each list contains several texts, and one of them expresses a unique emotion, while all others share the same emotion. You need to determine the index of the text that expresses the unique emotion."},
                            {"role": "user", "content": formatted_gen_prompt}
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