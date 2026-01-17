<h1>
  <img src="LongEmotion-logo.png" alt="LongEmotion Logo" width="80" style="vertical-align: middle; margin-right: 8px;">
  LongEmotion: Measuring Emotional Intelligence of Large Language Models in Long-Context Interaction
</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2509.07403"><img src="https://img.shields.io/badge/arXiv-2509.07403-b31b1b.svg" alt="arXiv"></a>
  <a href="https://huggingface.co/LongEmotion"><img src="https://img.shields.io/badge/🤗-Dataset-yellow.svg" alt="Dataset"></a>
</p>

---

## 🌟 Overview 

**LongEmotion** is a comprehensive benchmark designed to evaluate the **Emotional Intelligence (EI)** of Large Language Models (LLMs) in **long-context scenarios**.  
It includes six carefully constructed tasks that test emotion recognition, psychological knowledge application, and empathetic generation — areas crucial for emotionally coherent and human-aligned AI systems.

---

## 🚀 Key Features

- 🧩 **Comprehensive Benchmark** — Six complementary tasks covering three EI dimensions: recognition, knowledge, empathy.  
- 🧠 **Long-Context Evaluation** — Average context length exceeds **15,000 tokens**, simulating realistic emotional dialogues and narratives.  
- 🤝 **Collaborative Emotional Modeling (CoEM)** — A novel multi-agent RAG-based architecture to enhance long-context EI reasoning.  
---

## 🧩 Tasks

| Category | Task | Description | Metric |
|-----------|------|-------------|---------|
| **Emotion Recognition** | Emotion Classification (EC) | Identify the emotional category of a target entity in long, noisy text | Accuracy |
| **Emotion Recognition** | Emotion Detection (ED) | Detect the distinct emotional segment among N+1 segments | Accuracy |
| **Knowledge Application** | Emotion QA (QA) | Answer psychology-related questions from long-context literature | F1 Score |
| **Knowledge Application** | Emotion Summary (ES) | Summarize causes, symptoms, treatment, and effects from psychological reports | LLM-as-Judge |
| **Empathetic Generation** | Emotion Conversation (MC) | Simulate long counseling dialogues to assess empathy and guidance | LLM-as-Judge |
| **Empathetic Generation** | Emotion Expression (EE) | Generate structured emotional self-narratives across five phases | LLM-as-Judge |

---

## 🏗️ Framework: Collaborative Emotional Modeling (CoEM)

The **CoEM** framework integrates **Retrieval-Augmented Generation (RAG)** with **multi-agent emotional reasoning**, designed to improve EI under long-context conditions.

1. **Chunking** – Segment long texts into context chunks.  
2. **Initial Ranking** – Retrieve semantically relevant chunks via cosine similarity.  
3. **Multi-Agent Enrichment** – Use CoEM-Sage (e.g., GPT-4o, DeepSeek-V3) to add emotional and psychological insights.  
4. **Re-Ranking** – Refine enriched chunks for emotional and factual relevance.  
5. **Emotional Ensemble Generation** – Generate the final task-specific response with CoEM-Core.

---

## 💻 How to start

### 1️⃣ Environment Installation
```
conda create -n LongEmotion python==3.10
cd ~/LongEmotion
pip install -r requirements.txt
```

### 2️⃣ Preparation

Before running the benchmark, please complete the **model configuration** in `evaluate.sh`.

### ⚙️ Model Configuration

If your model is hosted locally, set `"model_url"` to `"local"`.

- **`model_name`** — The main model to be evaluated.  
- **`model_name_coem_sage`** — The auxiliary “CoEM-Sage” model used in the **CoEM Framework** for multi-agent collaboration.  
- **`evaluator_name`** — The evaluation model used for **LLM-as-Judge** tasks (e.g., GPT-4o).

Example configuration in `evaluate.sh`:

```bash
#!/usr/bin/env bash
# =====================================
# LongEmotion Evaluation Configuration
# =====================================

WORK_DIR=$(dirname $(readlink -f $0))
echo "Working Directory: ${WORK_DIR}"
UUID=$(uuidgen)
PID=$BASHPID
echo "UUID: ${UUID}"
echo "PID: ${PID}"

METHOD=$1   # baseline | rag | coem | self-rag | search-o1
TASK=$2     # e.g., Emotion_Summary, Emotion_QA, etc.

OUTPUT_DIR="${WORK_DIR}"
mkdir -p "${OUTPUT_DIR}"
LOG_FILE="${OUTPUT_DIR}/logs.txt"
exec &> >(tee -a "$LOG_FILE")

PYTHONPATH="${WORK_DIR}/src" python "${WORK_DIR}/evaluate.py" \
  --task "${TASK}" \
  --method "${METHOD}" \
  --data_dir "${WORK_DIR}/data" \
  --prompts_dir "${WORK_DIR}/prompts" \
  --base_dir "${OUTPUT_DIR}" \
  --model_name "deepseek-chat" \
  --model_api_key "your_api_key" \
  --model_url "your_url" \
  --model_name_coem_sage "deepseek-chat" \
  --model_api_key_coem_sage "your_api_key" \
  --model_url_coem_sage "your_url" \
  --evaluator_name "gpt-4o" \
  --evaluator_api_key "your_api_key" \
  --evaluator_url "your_url"
```
### 🧩 Running Tasks

Add the following script to run.sh to execute specific tasks or methods:
```bash
#!/usr/bin/env bash
# =====================================
# LongEmotion Run Script
# =====================================
# Available Methods:
#   1. baseline
#   2. rag
#   3. coem
#   4. self-rag
#   5. search-o1
#
# Available Tasks:
#   1. Emotion_Summary
#   2. Emotion_Classification_Emobench
#   3. Emotion_Classification_Finentity
#   4. Emotion_Detection
#   5. Emotion_QA
#   6. Emotion_Expression
#   7. Emotion_Conversations
# =====================================

export CUDA_VISIBLE_DEVICES=0

# Example: Run the baseline method on Emotion_Summary
bash evaluate.sh baseline Emotion_Summary
```


If you are testing under the Self-RAG setting, start a separate inference server before running run.sh:
```bash
vllm serve ~/selfrag_llama2_7b --gpu-memory-utilization 0.5 --dtype float16 --port 8010
```

After finishing all the settings, run **"bash run.sh"** to start evaluation!

### 🔧 Alternative Evaluation with lm_eval

You can also evaluate models using the `lm_eval` framework. 

#### Installation

First, install the lm_eval package:

```bash
pip install lm-eval
```

#### Running Tasks

Below are the commands for different tasks:

##### Emotion Classification (EC)

```bash
lm_eval --model vllm --model_args pretrained=meta-llama/Llama-3.1-8B-Instruct,tensor_parallel_size=8 --tasks longemotion_emotion_classification --include_path ./tasks --output_path results/emobench/ --log_samples --apply_chat_template --system_instruction "Please identify the emotion of the given subject in the scenario."
```

##### Emotion Detection (ED)

```bash
lm_eval --model vllm --model_args pretrained=meta-llama/Llama-3.1-8B-Instruct,tensor_parallel_size=8 --tasks longemotion_emotion_detection --include_path ./tasks --output_path results/emobench/ --log_samples --apply_chat_template --system_instruction "You are an emotion detection model. Your task is to identify the unique emotion in a list of given texts. Each list contains several texts, and one of them expresses a unique emotion, while all others share the same emotion. You need to determine the index of the text that expresses the unique emotion."
```

##### Emotion QA (QA)

```bash
lm_eval --model vllm --model_args pretrained=meta-llama/Llama-3.1-8B-Instruct,tensor_parallel_size=8 --tasks longemotion_fileqa --include_path ./tasks --output_path results/emobench/ --log_samples --apply_chat_template --system_instruction "You are a helpful assistant that answers questions based on the given context. Please provide accurate and concise answers."
```

##### Emotion Summary (ES)

```bash
lm_eval --model vllm --model_args pretrained=meta-llama/Llama-3.1-8B-Instruct,tensor_parallel_size=8 --tasks longemotion_report_summary --include_path ./tasks --output_path results/emobench/ --log_samples --apply_chat_template
```

##### Multi-turn Conversation (MC-4)

```bash
lm_eval --model vllm --model_args pretrained=meta-llama/Llama-3.1-8B-Instruct,tensor_parallel_size=8 --tasks longemotion_multiconv --include_path ./tasks --output_path results/emobench/ --log_samples --apply_chat_template --system_instruction "You are a professional counselor. Please respond based on the conversation history. Your response should be professional, empathetic, and constructive."
```

---

## 💡 Citation
```
@article{liu2025longemotion,
  title={LongEmotion: Measuring Emotional Intelligence of Large Language Models in Long-Context Interaction},
  author={Liu, Weichu and Xiong, Jing and Hu, Yuxuan and Li, Zixuan and Tan, Minghuan and Mao, Ningning and Zhao, Chenyang and Wan, Zhongwei and Tao, Chaofan and Xu, Wendong and others},
  journal={arXiv preprint arXiv:2509.07403},
  year={2025}
}
```
