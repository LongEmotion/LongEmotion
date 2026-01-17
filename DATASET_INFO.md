# LongEmotion 数据集详细信息

本文档提供 LongEmotion 数据集的详细统计信息和使用说明。

## 📦 数据集来源

- **HuggingFace 地址**: https://huggingface.co/datasets/LongEmotion/LongEmotion
- **本地路径**: `/home/xiongjing/LongEmotion/hf_dataset/`

## 📊 数据集概览

LongEmotion 包含 **6 大任务类型**，涵盖情绪识别、心理知识应用和共情生成三大维度。

### 总体统计

| 任务 | 样本数 | 平均上下文长度 | 文件格式 |
|------|--------|----------------|----------|
| Emotion Classification (Emobench) | 200 | ~19,345 tokens | JSONL |
| Emotion Classification (Finentity) | 200 | ~43,588 tokens | JSONL |
| Emotion Detection | 136 | ~4,592 tokens | JSONL |
| Emotion QA | 120 | N/A | JSONL |
| Emotion Conversation | 100 (400 轮次) | N/A | JSONL |
| Emotion Summary | 150 | N/A | JSONL |
| Emotion Expression (Situations) | 8 情绪类型 | N/A | JSON |
| Emotion Expression (Questionnaires) | 1 问卷 | N/A | JSON |

---

## 📁 任务详细说明

### 1. Emotion Classification - Emobench

**任务类型**: 情绪识别  
**文件路径**: `hf_dataset/Emotion Classification/Emotion_Classification_Emobench.jsonl`

- **样本数**: 200
- **平均长度**: 19,344.58 tokens
- **任务描述**: 在长篇且带有噪音的文本中识别目标实体的情绪类别
- **评估指标**: Accuracy

**情绪类别** (部分示例):
- 基础情绪: Acceptance, Admiration, Amusement, Anger, Caring, Delight, Disappointment, Disgust, Excitement, Gratitude, Guilt, Joy, Love, Pride, Sadness, Surprise
- 复合情绪: Admiration & Disapproval, Anger & Love, Excitement & Delight & Embarrassment 等

**数据字段**:
```json
{
  "id": int,
  "content": "长文本内容",
  "subject": "目标实体",
  "label": "情绪标签",
  "source": "数据来源",
  "choices": ["选项1", "选项2", ...],
  "length": int
}
```

---

### 2. Emotion Classification - Finentity

**任务类型**: 情绪识别  
**文件路径**: `hf_dataset/Emotion Classification/Emotion_Classification_Finentity.jsonl`

- **样本数**: 200
- **平均长度**: 43,587.77 tokens (最长上下文)
- **任务描述**: 在金融实体文本中进行情感分类
- **评估指标**: Accuracy

**情绪类别**:
- Positive (积极)
- Neutral (中性)
- Negative (消极)

**数据字段**:
```json
{
  "id": int,
  "content": "长文本内容",
  "subject": "目标实体",
  "label": "情绪标签",
  "source": "数据来源",
  "token_length": int,
  "choices": ["Positive", "Neutral", "Negative"]
}
```

---

### 3. Emotion Detection

**任务类型**: 情绪识别  
**文件路径**: `hf_dataset/Emotion Detection/Emotion_Detection.jsonl`

- **样本数**: 136
- **平均长度**: 4,592.07 tokens
- **任务描述**: 从 N+1 个文本片段中检测出情绪不同的片段
- **评估指标**: Accuracy

**数据字段**:
```json
{
  "text": "文本内容",
  "label": "标签",
  "length": int,
  "ground_truth": "正确答案"
}
```

---

### 4. Emotion QA

**任务类型**: 知识应用  
**文件路径**: `hf_dataset/Emotion QA/Emotion_QA.jsonl`

- **样本数**: 120
- **任务描述**: 基于长篇心理学文献回答相关问题
- **评估指标**: F1 Score
- **来源**: 30 篇心理学相关学术文献

**数据字段**:
```json
{
  "number": int,
  "problem": "问题",
  "answer": "答案",
  "source": "来源文献",
  "context": "上下文"
}
```

---

### 5. Emotion Conversation

**任务类型**: 共情生成  
**文件路径**: `hf_dataset/Emotion Conversation/Emotion_Conversations.jsonl`

- **对话数**: 100
- **总轮次**: 400 (平均每个对话 4 轮)
- **任务描述**: 模拟长篇心理咨询对话，评估共情能力和引导能力
- **评估指标**: LLM-as-Judge

**数据字段**:
```json
{
  "id": int,
  "stages": [
    {"stage": 1, "content": "对话内容"},
    ...
  ],
  "description": "场景描述"
}
```

---

### 6. Emotion Summary

**任务类型**: 知识应用  
**文件路径**: `hf_dataset/Emotion Summary/Emotion_Summary.jsonl`

- **样本数**: 150
- **任务描述**: 从心理咨询报告中总结病因、症状、治疗过程和效果
- **评估指标**: LLM-as-Judge

**数据字段**:
```json
{
  "id": int,
  "case_description": "案例描述",
  "consultation_process": "咨询过程",
  "experience_and_reflection": "经验和反思",
  "causes": "病因",
  "symptoms": "症状",
  "treatment_process": "治疗过程",
  "characteristics_of_illness": "疾病特征",
  "treatment_effect": "治疗效果"
}
```

**注意**: `Emotion_Summary_origin.jsonl` 是原始版本，`Emotion_Summary.jsonl` 是处理后版本。

---

### 7. Emotion Expression - Situations

**任务类型**: 共情生成  
**文件路径**: `hf_dataset/Emotion Expression/Emotion_Expression_Situations.json`

- **情绪类型数**: 8
- **任务描述**: 基于特定情境生成结构化的情绪自我叙述
- **评估指标**: LLM-as-Judge

**数据结构**:
```json
{
  "emotions": [
    {
      "emotion_name": "情绪名称",
      "situations": ["情境1", "情境2", ...]
    },
    ...
  ]
}
```

---

### 8. Emotion Expression - Questionnaires

**任务类型**: 共情生成  
**文件路径**: `hf_dataset/Emotion Expression/Emotion_Expression_Questionnaires.json`

- **问卷数**: 1
- **任务描述**: 通过问卷形式评估情绪表达的五个阶段
- **评估指标**: LLM-as-Judge

**数据结构**:
```json
[
  {
    "name": "问卷名称",
    "questions": ["问题1", "问题2", ...],
    "compute_mode": "计算模式",
    "prompt": "提示词",
    "inner_setting": "内部设置",
    "scale": "量表",
    "reverse": "反向计分项",
    "categories": "类别"
  }
]
```

---

## 🔧 数据使用示例

### 加载 JSONL 文件

```python
import json

# 读取情绪分类数据
with open('hf_dataset/Emotion Classification/Emotion_Classification_Emobench.jsonl', 'r', encoding='utf-8') as f:
    data = [json.loads(line) for line in f]
    
print(f"加载了 {len(data)} 个样本")
print(f"第一个样本: {data[0]}")
```

### 加载 JSON 文件

```python
import json

# 读取情绪表达数据
with open('hf_dataset/Emotion Expression/Emotion_Expression_Situations.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    
print(f"情绪类型: {len(data['emotions'])}")
```

---

## 📋 评估方法

### 自动评估任务
- **Emotion Classification (EC)**: Accuracy
- **Emotion Detection (ED)**: Accuracy
- **Emotion QA**: F1 Score

### LLM-as-Judge 任务
- **Emotion Summary (ES)**: 使用评估模型（如 GPT-4o）评分
- **Emotion Conversation (MC)**: 评估共情和引导质量
- **Emotion Expression (EE)**: 评估情绪表达的完整性和准确性

---

## 🚀 快速开始

1. **环境准备**
```bash
conda create -n LongEmotion python==3.10
pip install -r requirements.txt
```

2. **配置模型** (编辑 `evaluate.sh`)
```bash
--model_name "your-model"
--model_api_key "your-api-key"
--evaluator_name "gpt-4o"
```

3. **运行评估**
```bash
bash evaluate.sh baseline Emotion_Classification_Emobench
```

---

## 📖 引用

如果您使用了 LongEmotion 数据集，请引用：

```bibtex
@article{liu2025longemotion,
  title={LongEmotion: Measuring Emotional Intelligence of Large Language Models in Long-Context Interaction},
  author={Liu, Weichu and Xiong, Jing and Hu, Yuxuan and Li, Zixuan and Tan, Minghuan and Mao, Ningning and Zhao, Chenyang and Wan, Zhongwei and Tao, Chaofan and Xu, Wendong and others},
  journal={arXiv preprint arXiv:2509.07403},
  year={2025}
}
```

---

## 🔗 相关链接

- **论文**: [arXiv:2509.07403](https://arxiv.org/abs/2509.07403)
- **HuggingFace**: [LongEmotion/LongEmotion](https://huggingface.co/datasets/LongEmotion/LongEmotion)
- **GitHub**: (待补充)

---

*最后更新: 2026-01-17*
