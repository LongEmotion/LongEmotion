import json
import os
import re
import tqdm
import numpy as np
import faiss
from langchain_text_splitters import RecursiveCharacterTextSplitter
from FlagEmbedding import BGEM3FlagModel
from .base import BaseEvaluator
from transformers import AutoTokenizer
from .qa_score import qa_f1_score


class SearchO1Evaluator(BaseEvaluator):

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

        # 初始化BGE模型
        self.bge_model = BGEM3FlagModel('~/Bge-m3', use_fp16=True)
        
        
        self.tokenizer = AutoTokenizer.from_pretrained("~/Llama-3.1-8B-Instruct", trust_remote_code=True)
        # 初始化文本分割器
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
    
    def count_tokens(self,text):
        return len(self.tokenizer.encode(text, add_special_tokens=False))

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

    def _generate_with_internal_search_over_text(
        self,
        conversation_history,
        base_text_for_search,
        *,
        system_prompt,
        top_k=3,
        max_turn=5,
    ):

        reasoning = ""
        query_pattern = r"<\|begin_search_query\|>(.*?)<\|end_search_query\|>"
        result_block_pattern = r"<\|begin_search_result\|>.*?<\|end_search_result\|>"


        current_turn = 0
        while current_turn < max_turn + 2:
            current_turn += 1

            messages = (
                [{"role": "system", "content": system_prompt}]
                + conversation_history
                + [{"role": "assistant", "content": reasoning}]
            )
            reasoning = reasoning + self.chat_completion(
                self.model_api_key, model=self.model_name, messages=messages,role="generator"
            )

            queries = re.findall(query_pattern, reasoning, re.DOTALL)
            results = re.findall(result_block_pattern, reasoning, re.DOTALL)
            qn, rn = len(queries), len(results)


            if qn == rn + 1:
                query = (queries[-1] or "").strip()


                search_text = ""
                if isinstance(base_text_for_search, str) and base_text_for_search.strip():
                    search_text += base_text_for_search.strip() + "\n"
                search_text += reasoning
                chunks = self.text_splitter_128.split_text(search_text) if search_text.strip() else []

                chunks_tmp = self.retrieve_with_bge(query, chunks, top_k=top_k) if chunks else []
                chunks_tmp_str = "\n".join(
                    [f"Chunk{i+1}:\n{chunk}" for i, chunk in enumerate(chunks_tmp)]
                )
                reasoning = (
                    reasoning
                    + "<|begin_search_result|>\n"
                    + chunks_tmp_str
                    + "\n<|end_search_result|>\n"
                )
                continue

            break

        return reasoning.strip()
    
    def run_report_summary(self, test_data):

        output_file_root = f"{self.base_dir}/Final_result/Search-o1/{self.task}/"
        max_try = 5
        prompt_gen_system = open(f"{self.prompts_dir}/search_o1/report_summary_system.txt").read()
        gen_user = open(f"{self.prompts_dir}/search_o1/report_summary_user.txt").read()
        max_turn = 5
        time = 1
        cnt = 0

        while time <= 1:
            output_file = output_file_root + f"{self.model_name}_report_summary_result_subject_{time}.jsonl"
            with open(output_file, 'a', encoding='utf-8') as outfile:
                for item in tqdm.tqdm(test_data, desc="Processing report summary with search-o1"):
                    cnt += 1
                    retry_count = 0
                    success = False
                    flag = 0

                    case_description = item['case_description']
                    consultation_process = item['consultation_process']
                    experience_and_reflection = item['experience_and_reflection']


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

                    
                    full_context = f"Case Description: {case_description}\nConsultation Process: {consultation_process}\nExperience and Reflection: {experience_and_reflection}"

                    chunks = self.text_splitter_128.split_text(full_context)

                    result = {}
                    chat_response = ""

                    while retry_count < max_try and not success:
                        reasoning = ""
                        retry_count = retry_count + 1
                        predicted_causes = ""
                        predicted_symptoms = ""
                        predicted_treatment_process = ""
                        predicted_characteristics = ""
                        predicted_treatment_effect = ""
                        current_turn = 0

                        while current_turn < max_turn + 2:
                            current_turn += 1
                            prompt_gen_user = gen_user.format(
                                Case_description=case_description,
                                Consultation_process=consultation_process,
                                Experience_and_reflection=experience_and_reflection
                            )

                            messages = [
                                {"role": "system", "content": prompt_gen_system},
                                {"role": "user", "content": prompt_gen_user},
                                {"role": "assistant", "content": reasoning}
                            ]
                            reasoning = reasoning + self.chat_completion(self.model_api_key, model=self.model_name, messages=messages,role="generator")




                            all_matches = list(re.finditer(r'```json\s*(\{.*?\})\s*```', reasoning, re.DOTALL))
                            json_extracted = False

                            if all_matches:
                                last_match = all_matches[-1]
                                try:
                                    json_str = last_match.group(1)
                                    predicted_data = json.loads(json_str)
                                    predicted_causes = predicted_data.get("causes", "")
                                    predicted_symptoms = predicted_data.get("symptoms", "")
                                    predicted_treatment_process = predicted_data.get("treatment_process", "")
                                    predicted_characteristics = predicted_data.get("characteristics_of_illness", "")
                                    predicted_treatment_effect = predicted_data.get("treatment_effect", "")
                                    success = True
                                    json_extracted = True
                                    break  
                                except Exception as e:
                                    print(f"Evaluation error: {e}")


                            if not json_extracted:
                                json_matches = list(re.finditer(r'\{[^{}]*"causes"\s*:\s*"[^"]*"[^}]*\}', reasoning, re.DOTALL))
                                if json_matches:
                                    last_json_match = json_matches[-1]
                                    try:
                                        json_str = last_json_match.group(0)
                                        predicted_data = json.loads(json_str)
                                        predicted_causes = predicted_data.get("causes", "")
                                        predicted_symptoms = predicted_data.get("symptoms", "")
                                        predicted_treatment_process = predicted_data.get("treatment_process", "")
                                        predicted_characteristics = predicted_data.get("characteristics_of_illness", "")
                                        predicted_treatment_effect = predicted_data.get("treatment_effect", "")
                                        success = True
                                        json_extracted = True
                                        break  
                                    except Exception as e:
                                        print(f"Evaluation error (fallback): {e}")

                  
                            if not json_extracted:
                                if "<|end_search_query|>" in reasoning:
                              
                                    begin_pattern = r'<\|begin_search_query\|>(.*?)<\|end_search_query\|>'
                                    matches = re.findall(begin_pattern, reasoning, re.DOTALL)
                                    query = ""
                                    if matches:
                                        query = matches[-1]  
                                    else:
                                        query = ""  
                                    chunks_tmp = self.retrieve_with_bge(query, chunks, top_k=3)
                                    chunks_tmp_str = "\n".join([f"Chunk{i+1}:\n{chunk}" for i, chunk in enumerate(chunks_tmp)])
                                    reasoning = reasoning + "<|begin_search_result|>\n" + chunks_tmp_str + "<|end_search_result|>\n"
                                    continue
                                else:
                                    continue

                        chat_response = reasoning

                    if not success:
                        flag = 1


                    evaluation_scores = None
                    ground_truth_str = f"Causes: {item.get('causes', '')}\nSymptoms: {item.get('symptoms', '')}\nTreatment process: {item.get('treatment_process', '')}\nCharacteristics of the illness: {item.get('characteristics_of_illness', '')}\nTreatment effect: {item.get('treatment_effect', '')}"
                    if flag == 0:
                        model_response_str = f"Causes: {predicted_causes}\nSymptoms: {predicted_symptoms}\nTreatment process: {predicted_treatment_process}\nCharacteristics of the illness: {predicted_characteristics}\nTreatment effect: {predicted_treatment_effect}"
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
        output_file_root = f"{self.base_dir}/Final_result/Search-o1/{self.task}/"
        max_try=5
        prompt_gen_system = open(f"{self.prompts_dir}/search_o1/emotion_classification_system.txt").read()
        gen_user = open(f"{self.prompts_dir}/search_o1/emotion_classification_user.txt").read()
        max_turn = 5
        time = 1
        cnt = 0
        while time <= 1:
            output_file = output_file_root + f"{self.model_name}_Emo_class_NEW_result_subject_{time}.jsonl"
            with open(output_file, 'a', encoding='utf-8') as outfile:
                for item in tqdm.tqdm(test_data):
                    cnt +=1
                    retry_count=0
                    success = False
                    context = item['content']
                    subject = item['subject']
                    choices = item['choices']  
                    true_label = item['label']
                    while retry_count < max_try and not success:
                        reasoning = ""
                        retry_count = retry_count + 1
                        predicted_emotion = ""
                        current_turn = 0

                        chunks = self.text_splitter_128.split_text(context)
                        while current_turn < max_turn+2:
                            current_turn += 1
                            prompt_gen_user = gen_user.format(context=context, subject=subject, choices=", ".join(choices))

                            messages = [
                                {"role": "system", "content": prompt_gen_system},
                                {"role": "user", "content": prompt_gen_user},
                                {"role": "assistant", "content": reasoning}
                            ]
                            reasoning = reasoning + self.chat_completion(self.model_api_key, model=self.model_name, messages=messages,role="generator")

                        
    
                            all_matches = list(re.finditer(r'```json\s*(\{.*?\})\s*```', reasoning, re.DOTALL))
                            json_extracted = False

                            if all_matches:
                                last_match = all_matches[-1]
                                try:
                                    json_str = last_match.group(1)
                                    predicted_data = json.loads(json_str)
                                    predicted_emotion = predicted_data.get("Emotion", "")
                                    success = True
                                    json_extracted = True
                                    break 
                                except Exception as e:
                                    print(f"Evaluation error:", e)

                       
                            if not json_extracted:
                                json_matches = list(re.finditer(r'\{[^{}]*"Emotion"\s*:\s*"[^"]*"[^}]*\}', reasoning, re.DOTALL))
                                if json_matches:
                                    last_json_match = json_matches[-1]
                                    try:
                                        json_str = last_json_match.group(0)
                                        predicted_data = json.loads(json_str)
                                        predicted_emotion = predicted_data.get("Emotion", "")
                                        success = True
                                        json_extracted = True
                                        break 
                                    except Exception as e:
                                        print(f"Evaluation error (fallback):", e)

                            # 如果没有提取到JSON，检查是否需要搜索查询
                            if not json_extracted:
                                if "<|end_search_query|>" in reasoning:
 
                                    begin_pattern = r'<\|begin_search_query\|>(.*?)<\|end_search_query\|>'
                                    matches = re.findall(begin_pattern, reasoning, re.DOTALL)
                                    query = matches[-1] if matches else ""
                                    chunks_tmp = self.retrieve_with_bge(query, chunks, top_k=1)
                                    chunks_tmp_str = "\n".join(
                                        [f"Chunk{i+1}:\n{chunk}" for i, chunk in enumerate(chunks_tmp)]
                                    )
                                    reasoning = (
                                        reasoning
                                        + "<|begin_search_result|>\n"
                                        + chunks_tmp_str
                                        + "<|end_search_result|>\n"
                                    )
                                    continue
                            else:
                                continue

                    result = "Correct" if predicted_emotion == true_label else "Incorrect"
                    self.evaluate_results.update({result: 1})
                    result_entry = {
                        "choices": choices,
                        "predicted_emotion": predicted_emotion,
                        "true_label": true_label,
                        "result": result,
                        "chat_response":reasoning
                    }
                    outfile.write(json.dumps(result_entry, ensure_ascii=False) + '\n')

                    if result == "Correct":
                        correct_count += 1
            accuracy = correct_count / total_count if total_count > 0 else 0
            time += 1 

    def questionnaire(self, test_data):

        with open(f"{self.data_dir}/Emotion_Expression_Situations.json", 'r',encoding='utf-8') as f:
            all_situation = json.load(f)

        with open(f"{self.data_dir}/Emotion_Expression_Questionnaires.json", 'r',encoding='utf-8') as f:
            all_questionnaire= json.load(f)

        gen_prompt = open(f"{self.prompts_dir}/search_o1/questionnaire_user.txt").read()
        output_file_root = f"{self.base_dir}/Final_result/Search-o1/Emotion_Expression/{self.model_name}/"
        gen_prompt_1 = open(f"{self.prompts_dir}/search_o1/questionnaire_stage_1.txt").read()
        gen_prompt_2 = open(f"{self.prompts_dir}/search_o1/questionnaire_stage_2.txt").read()
        gen_prompt_3 = open(f"{self.prompts_dir}/search_o1/questionnaire_stage_3.txt").read()
        gen_prompt_4 = open(f"{self.prompts_dir}/search_o1/questionnaire_stage_4.txt").read()
        gen_prompt_5 = open(f"{self.prompts_dir}/search_o1/questionnaire_stage_5.txt").read()
        # 旧逻辑：Stage2-5 使用静态 query_xxx.txt 做检索
        # 新逻辑：移植 search-o1，模型在生成中自动产出 query（<|begin_search_query|>...）
        prompt_searcho1_system = open(f"{self.prompts_dir}/search_o1/questionnaire_system.txt").read()
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
        output_file = output_file_root + f"{self.model_name}_questionnaire_result.jsonl"
        with open(output_file,"a") as outfile:
            for emotion in all_situation["emotions"]:     
                # 遍历每个因素
                for factor in emotion["factors"]:
                    # 遍历每个场景
                    for situation in factor["scenarios"]:                
                        answer = ""
                        case_id += 1  # 修复赋值操作符
                        conversation_history = []
                        retry_cnt=0
                        # 第一阶段
                        message = gen_prompt.format(SITUATION=situation, statements=statement)
                        messages = {"role": "user", "content": message}
                        conversation_history.append(messages)
                        while retry_cnt < 5:
                            response = self.chat_completion(
                                self.model_api_key,
                                model=self.model_name,
                                messages=[{"role": "system", "content": prompt_searcho1_system}] + conversation_history,
                                role="generator"
                            )
                            if not all(keyword in str(response) for keyword in ["Interested", "Distressed", "Excited", "Upset", "Strong", "Guilty", "Scared", "Hostile", "Enthusiastic", "Proud", "Irritable", "Alert", "Ashamed", "Inspired", "Nervous", "Determined", "Attentive", "Jittery", "Active", "Afraid"]):
                                retry_cnt +=1
                                continue
                            else :
                                break

                        conversation_history.append({"role": "assistant", "content": response})
                        answer = answer + response + "\n"
                        # 保存第0阶段响应
                        with open(output_file_root+f"response_Generation_{case_id}.txt", "a") as f:
                            f.write(response + "\n")
                        # 后续阶段
                        stage = 1
                        while stage <= 5:
                            # 选择对应阶段的提示词
                            if stage == 1:
                                add_message = gen_prompt_1    # 添加用户消息
                                conversation_history.append({"role": "user", "content": add_message})
                                
                                # 获取模型响应
                                counselor_response = self.chat_completion(self.model_api_key, model=self.model_name, messages=conversation_history,role="generator")
                                answer = answer + counselor_response +"\n"

                                # 保存响应
                                with open(output_file_root+f"response_Generation_{case_id}.txt", "a") as f:
                                    f.write(counselor_response + "\n")

                                conversation_history.append({"role": "assistant", "content": counselor_response})
                            else:
                                if stage == 2:
                                    add_message = gen_prompt_2
                                    n=2
                                elif stage == 3:
                                    add_message = gen_prompt_3
                                    n=4
                                elif stage == 4:
                                    add_message = gen_prompt_4
                                    n=4
                                elif stage == 5:
                                    add_message = gen_prompt_5
                                    n=4

                                conversation_history.append({"role": "user", "content": add_message})

                                counselor_response = self._generate_with_internal_search_over_text(
                                    conversation_history,
                                    answer,
                                    system_prompt=prompt_searcho1_system,
                                    top_k=n,
                                    max_turn=5,
                                )
                                answer = answer + counselor_response + "\n"

                                # 保存响应
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

        output_file_root = f"{self.base_dir}/Final_result/Search-o1/Emotion_QA/"
        os.makedirs(output_file_root, exist_ok=True)

        prompt_gen_system = open(f"{self.prompts_dir}/search_o1/fileqa_system.txt").read()
        gen_user = open(f"{self.prompts_dir}/search_o1/fileqa_user.txt").read()
        max_try = 5
        max_turn = 5

        total_f1_score = 0
        total_count = 0
        cnt=0
        time = 1
        while time <= 1:
            output_file = output_file_root + f"{self.model_name}_fileqa_searcho1_{time}.jsonl"
            with open(output_file, 'a', encoding='utf-8') as outfile:
                for item in tqdm.tqdm(test_data, desc="Processing QA pairs"):
                    cnt += 1
                    retry_count = 0
                    success = False

                    number = item['number']
                    question = item['problem']
                    context = item['context']
                    ground_truth = item['answer']
             
                    chunks = self.text_splitter_128.split_text(context)
                    final_answer = ""
                    reasoning = ""
                    while retry_count < max_try and not success:
                        reasoning = ""
                        final_answer = ""
                        retry_count += 1
                        current_turn = 0

                        while current_turn < max_turn + 2:
                            current_turn += 1
                            prompt_gen_user = gen_user.format(context=context,question=question)
                            
                            messages = [
                                {"role": "system", "content": prompt_gen_system},
                                {"role": "user", "content": prompt_gen_user},
                                {"role": "assistant", "content": reasoning}
                            ]
                            reasoning = reasoning + self.chat_completion(self.model_api_key, model=self.model_name, messages=messages,role="generator")
                            all_matches = list(re.finditer(r'```json\s*(\{.*?\})\s*```', reasoning, re.DOTALL))
                            json_extracted = False
                            if all_matches:
                                last_match = all_matches[-1]
                                try:
                                    json_str = last_match.group(1)
                                    predicted_data = json.loads(json_str)
                                    final_answer = predicted_data.get("answer", "")
                                    if isinstance(final_answer, str) and final_answer.strip():
                                        success = True
                                        json_extracted = True
                                        break
                                except Exception as e:
                                    print(f"Evaluation error:", e)

                            if not json_extracted:
                                json_matches = list(re.finditer(r'\{[^{}]*"answer"\s*:\s*"[^"]*"[^}]*\}', reasoning, re.DOTALL))
                                if json_matches:
                                    last_json_match = json_matches[-1]
                                    try:
                                        json_str = last_json_match.group(0)
                                        predicted_data = json.loads(json_str)
                                        final_answer = predicted_data.get("answer", "")
                                        if isinstance(final_answer, str) and final_answer.strip():
                                            success = True
                                            json_extracted = True
                                            break
                                    except Exception as e:
                                        print(f"Evaluation error (fallback):", e)
                            if not json_extracted and "<|end_search_query|>" in reasoning:
                                begin_pattern = r'<\|begin_search_query\|>(.*?)<\|end_search_query\|>'
                                matches = re.findall(begin_pattern, reasoning, re.DOTALL)
                                query = matches[-1] if matches else ""
                                chunks_tmp = self.retrieve_with_bge(query, chunks, top_k=4)
                                positions_by_chunk = {}
                                for i, c in enumerate(chunks):
                                    positions_by_chunk.setdefault(c, []).append(i)

                                retrieved = []
                                for c in chunks_tmp:
                                    pos_list = positions_by_chunk.get(c)
                                    if not pos_list:
                                        continue
                                    i = pos_list.pop(0)
                                    retrieved.append((i, c))
                                retrieved.sort(key=lambda x: x[0])

                                chunks_tmp_str = "\n".join(
                                    [f"chunk {i+1} (index {i}): {c}" for i, c in retrieved]
                                )
                                reasoning = reasoning + "<|begin_search_result|>\n" + chunks_tmp_str + "\n<|end_search_result|>\n"
                                continue
                    if not isinstance(final_answer, str) or not final_answer.strip():
                        final_answer = reasoning.split("</think>")[-1].strip()

                    f1_score = qa_f1_score(final_answer.strip(), ground_truth)
                    total_f1_score += f1_score
                    total_count += 1
                    result_entry = {
                        "number": number,
                        "input": question,
                        "model_response": final_answer.strip(),
                        "ground_truth": ground_truth,
                        "f1_score": f1_score,
                        "chat_response": reasoning
                    }
                    outfile.write(json.dumps(result_entry, ensure_ascii=False) + '\n')
            time += 1


    def run_multicov(self, test_data):
        output_file_root = f"{self.base_dir}/Final_result/Search-o1/Emotion_Conversation/"
        prompt_gen_system = open(f"{self.prompts_dir}/search_o1/multicov_system.txt").read()
        gen_user = open(f"{self.prompts_dir}/search_o1/multicov_user.txt").read()
        conv_score_prompt_4 = open(f"{self.prompts_dir}/eval_prompt/conv_score_prompt_4.txt").read()
        item_id = 0
        time = 1 
        N = 4
        max_turn = 5 
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
                                prompt_gen_user = gen_user.format(
                                    dialogue_history=all_stage_history_str,
                                    latest_reply=query,
                                )

                                reasoning = ""
                                gen_response = ""
                                current_turn = 0
                                begin_pattern = r'<\|begin_search_query\|>(.*?)<\|end_search_query\|>'
                                result_block_pattern = r'<\|begin_search_result\|>.*?<\|end_search_result\|>'

                                while current_turn < max_turn + 2:
                                    current_turn += 1
                                    messages = [
                                        {"role": "system", "content": prompt_gen_system},
                                        {"role": "user", "content": prompt_gen_user},
                                        {"role": "assistant", "content": reasoning},
                                    ]
                                    new_text = self.chat_completion(
                                        self.model_api_key, model=self.model_name, messages=messages,role="generator"
                                    )
                                    reasoning = reasoning + (new_text or "")

                                    all_matches = list(
                                        re.finditer(r"```json\s*(\{.*?\})\s*```", reasoning, re.DOTALL)
                                    )
                                    json_extracted = False
                                    if all_matches:
                                        last_match = all_matches[-1]
                                        try:
                                            json_str = last_match.group(1)
                                            predicted_data = json.loads(json_str)
                                            gen_response = str(predicted_data.get("response", "")).strip()
                                            json_extracted = True
                                            break
                                        except Exception as e:
                                            print("MultiCov JSON parse error:", e)

                                    if not json_extracted:
                                        json_matches = list(
                                            re.finditer(
                                                r'\{[^{}]*"response"\s*:\s*"[\s\S]*?"[^{}]*\}',
                                                reasoning,
                                                re.DOTALL,
                                            )
                                        )
                                        if json_matches:
                                            last_json_match = json_matches[-1]
                                            try:
                                                json_str = last_json_match.group(0)
                                                predicted_data = json.loads(json_str)
                                                gen_response = str(predicted_data.get("response", "")).strip()
                                                json_extracted = True
                                                break
                                            except Exception as e:
                                                print("MultiCov JSON parse error (fallback):", e)

                                    if not json_extracted and "<|end_search_query|>" in reasoning:
                                        matches = re.findall(begin_pattern, reasoning, re.DOTALL)
                                        result_blocks = re.findall(result_block_pattern, reasoning, re.DOTALL)

                                        if len(matches) > len(result_blocks):
                                            query_search = matches[-1] if matches else ""
                                            chunks_tmp = self.retrieve_with_bge(query_search, chunks, top_k=N)
                                            chunks_tmp_str = "\n".join(
                                                [f"Chunk{i+1}:\n{chunk}" for i, chunk in enumerate(chunks_tmp)]
                                            )
                                            reasoning = (
                                                reasoning
                                                + "<|begin_search_result|>\n"
                                                + chunks_tmp_str
                                                + "\n<|end_search_result|>\n"
                                            )
                                            continue

                                    continue

                                if not isinstance(gen_response, str) or not gen_response.strip():
                                    gen_response = reasoning.strip()

                                all_stage_history_str += query

                                formatted_eval_prompt = evaluation_prompt.format(
                                    dialogue_history=all_stage_history_str,
                                    latest_dialogue_segment=gen_response
                                )
                                
                                eval_messages = [
                                    {"role": "system", "content": "You are a psychotherapy process evaluator."},
                                    {"role": "user", "content": formatted_eval_prompt}
                                ]

                                eval_response = self.chat_completion(self.evaluator_api_key, model=self.evaluator_name, messages=eval_messages,role="evaluator")

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
        output_file_root = f"{self.base_dir}/Final_result/Search-o1/Emotion_Detection/"
        max_try=5
        prompt_gen_system = open(f"{self.prompts_dir}/search_o1/emotion_detection_system.txt").read()
        gen_user = open(f"{self.prompts_dir}/search_o1/emotion_detection_user.txt").read()
        max_turn = 5
        time =1
        cnt = 0
        while time <= 1 :
            output_file = output_file_root + f"{self.model_name}_Emo_detection_result_{time}.jsonl"
            with open(output_file, 'a', encoding='utf-8') as outfile:
                for id, item in tqdm.tqdm(enumerate(test_data), desc="Processing items"):
                    cnt +=1
                    retry_count=0
                    success = False
                    texts = item['text']
                    ground_truth = item['ground_truth']
                    text_list = ',\n'.join([f'["index": {seg["index"]}, "text": "{seg["context"]}"]' for seg in texts])

                    while retry_count < max_try and not success:
                        reasoning = ""
                        retry_count = retry_count + 1
                        predicted_index = -1
                        current_turn = 0

                        chunks = [seg['context'] for seg in texts]

                        while current_turn < max_turn+2:
                            current_turn += 1
                            text_list = ',\n'.join([f'["index": {seg["index"]}, "text": "{seg["context"]}"]' for seg in texts])
                            prompt_gen_user = gen_user.format( num=len(texts), texts=text_list)

                            messages = [
                                {"role": "system", "content": prompt_gen_system},
                                {"role": "user", "content": prompt_gen_user},
                                {"role": "assistant", "content": reasoning}
                            ]
                            reasoning = reasoning + self.chat_completion(self.model_api_key, model=self.model_name, messages=messages,role="generator")
                            all_matches = list(re.finditer(r'```json\s*(\{.*?\})\s*```', reasoning, re.DOTALL))
                            json_extracted = False

                            if all_matches:
                                last_match = all_matches[-1]
                                try:
                                    json_str = last_match.group(1)
                                    predicted_data = json.loads(json_str)
                                    predicted_index = predicted_data.get("index", -1)
                                    success = True
                                    json_extracted = True
                                    break
                                except Exception as e:
                                    print(f"Evaluation error:", e)

                            if not json_extracted:
                                json_matches = list(re.finditer(r'\{[^{}]*"index"\s*:\s*\d+[^}]*\}', reasoning, re.DOTALL))
                                if json_matches:
                                    last_json_match = json_matches[-1]
                                    try:
                                        json_str = last_json_match.group(0)
                                        predicted_data = json.loads(json_str)
                                        predicted_index = predicted_data.get("index", -1)
                                        success = True
                                        json_extracted = True
                                        break  
                                    except Exception as e:
                                        print(f"Evaluation error (fallback):", e)
                            if not json_extracted:
                                if "<|end_search_query|>" in reasoning:
                                    begin_pattern = r'<\|begin_search_query\|>(.*?)<\|end_search_query\|>'
                                matches = re.findall(begin_pattern, reasoning, re.DOTALL)
                                query=""
                                if matches:
                                    query = matches[-1]  
                                else:
                                    query = ""  
                                chunks_tmp = self.retrieve_with_bge(query,chunks,top_k=4)
                                positions_by_chunk = {}
                                for i, c in enumerate(chunks):
                                    positions_by_chunk.setdefault(c, []).append(i)

                                retrieved = []
                                for c in chunks_tmp:
                                    pos_list = positions_by_chunk.get(c)
                                    if not pos_list:
                                        continue
                                    i = pos_list.pop(0)
                                    orig_index = texts[i]["index"]
                                    retrieved.append((orig_index, i, c))

                                retrieved.sort(key=lambda x: (x[0], x[1]))
                                chunks_tmp_str = ',\n'.join(
                                    [f'["index": {orig_index}, "text": "{c}"]' for orig_index, _, c in retrieved]
                                )
                                reasoning = reasoning + "<|begin_search_result|>\n" + chunks_tmp_str + "<|end_search_result|>\n"
                                continue
                            else:
                                continue

                    print(f"Predicted: {predicted_index}, Actual: {ground_truth}")
                    result = "Correct" if predicted_index == ground_truth else "Incorrect"

                    self.evaluate_results.update({result: 1})

                    result_entry = {
                        "id": id,
                        "predicted_index": predicted_index,
                        "ground_truth": ground_truth,
                        "result": result,
                        "chat_response":reasoning
                    }
                    outfile.write(json.dumps(result_entry, ensure_ascii=False) + '\n')

                    if result == "Correct":
                        correct_count += 1

            time += 1