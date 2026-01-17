# LongEmotion 数据集说明

<div align="center">
  <img src="LongEmotion-logo.png" alt="LongEmotion Logo" width="200">
  
  <h3>测量大语言模型在长上下文交互中的情商</h3>
  
  [![Paper](https://img.shields.io/badge/arXiv-2509.07403-b31b1b.svg)](https://arxiv.org/abs/2509.07403)
  [![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace-yellow)](https://huggingface.co/datasets/LongEmotion/LongEmotion)
  [![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
</div>

---

## 📚 目录

- [数据集简介](#数据集简介)
- [数据统计](#数据统计)
- [任务说明](#任务说明)
- [数据格式](#数据格式)
- [下载与使用](#下载与使用)
- [评估框架](#评估框架)
- [常见问题](#常见问题)

---

## 数据集简介

**LongEmotion** 是首个专门用于评估大语言模型（LLMs）在**长上下文场景**下**情商（Emotional Intelligence, EI）**的综合性基准测试。

### 🎯 核心特点

- **长上下文**: 平均上下文长度超过 **15,000 tokens**，最长达 **43,588 tokens**
- **全面评估**: 涵盖情绪识别、心理知识应用、共情生成三大维度
- **多样化任务**: 6 大任务类型，8 个子任务
- **真实场景**: 数据来源于心理咨询案例、学术文献、情绪对话等真实场景

### 📊 三大评估维度

| 维度 | 任务 | 评估指标 |
|------|------|---------|
| **情绪识别** | Emotion Classification (EC)<br>Emotion Detection (ED) | Accuracy |
| **知识应用** | Emotion QA (QA)<br>Emotion Summary (ES) | F1 Score<br>LLM-as-Judge |
| **共情生成** | Emotion Conversation (MC)<br>Emotion Expression (EE) | LLM-as-Judge |

---

## 数据统计

### 整体统计

```
📦 LongEmotion 数据集
├── 总任务数: 8
├── 总样本数: 1,106+
├── 平均上下文长度: 15,000+ tokens
└── 最长上下文: 43,588 tokens
```

### 详细统计

| 任务名称 | 类型 | 样本数 | 平均长度 | 数据来源 |
|---------|------|--------|----------|----------|
| **EC-Emobench** | 分类 | 200 | 19,345 | BookCorpus |
| **EC-Finentity** | 分类 | 200 | 43,588 | 金融文档 |
| **ED** | 检测 | 136 | 4,592 | 混合来源 |
| **QA** | 问答 | 120 | - | 30篇学术论文 |
| **MC** | 对话 | 100 (400轮) | - | 心理咨询对话 |
| **ES** | 摘要 | 150 | - | 心理咨询报告 |
| **EE-Situations** | 生成 | 8类 | - | 情境描述 |
| **EE-Questionnaires** | 生成 | 1问卷 | - | 心理量表 |

---

## 任务说明

### 1️⃣ 情绪分类 (Emotion Classification)

#### EC-Emobench
- **目标**: 在长篇小说片段中识别角色的复杂情绪
- **难点**: 文本长、干扰信息多、情绪细粒度且可能混合
- **情绪类别**: 80+ 种（包括单一和复合情绪）
  - 单一情绪: Joy, Sadness, Anger, Fear, Surprise, Disgust, etc.
  - 复合情绪: "Joy & Gratitude", "Anger & Disappointment", etc.

**示例**:
```json
{
  "id": 1,
  "content": "[20,000+ token的长篇文本]",
  "subject": "Elizabeth",
  "label": "Delight",
  "choices": ["Delight", "Disappointment", "Anger", "Pessimism", "Remorse", "Anticipation"],
  "length": 20082
}
```

#### EC-Finentity
- **目标**: 在极长的金融文档中判断实体的情感倾向
- **难点**: 超长上下文（平均 43k+ tokens）
- **情绪类别**: Positive, Neutral, Negative

---

### 2️⃣ 情绪检测 (Emotion Detection)

- **目标**: 从 N+1 个文本片段中找出情绪不同的那一个
- **难点**: 需要理解多个文本片段并进行对比
- **任务类型**: N选1 多选题

**示例**:
```json
{
  "text": {
    "option_A": "文本片段A",
    "option_B": "文本片段B",
    "option_C": "文本片段C",
    "option_D": "文本片段D"
  },
  "label": "A",
  "ground_truth": "选项A的情绪与其他不同"
}
```

---

### 3️⃣ 情绪问答 (Emotion QA)

- **目标**: 基于心理学文献回答专业问题
- **难点**: 需要精准理解学术文献并提取关键信息
- **来源**: 30 篇心理健康相关的学术论文
- **评估**: F1 Score（与标准答案对比）

**主题示例**:
- 压力与健康行为的关系
- 手机使用与心理健康
- 身体活动对心理健康的影响
- AI技术对心理健康的影响

---

### 4️⃣ 情绪摘要 (Emotion Summary)

- **目标**: 从心理咨询案例报告中提取结构化信息
- **输出字段**:
  - `causes`: 病因分析
  - `symptoms`: 症状描述
  - `treatment_process`: 治疗过程
  - `treatment_effect`: 治疗效果

**示例**:
```json
{
  "id": 1,
  "case_description": "来访者是一位35岁的女性...",
  "consultation_process": "咨询共进行了12次...",
  "causes": "童年创伤、工作压力...",
  "symptoms": "焦虑、失眠、情绪低落...",
  "treatment_process": "认知行为疗法...",
  "treatment_effect": "症状明显改善..."
}
```

---

### 5️⃣ 情绪对话 (Emotion Conversation)

- **目标**: 模拟心理咨询对话，展现共情能力
- **对话结构**: 每个案例 4 轮对话
  - 第1轮: 来访者初次描述问题
  - 第2轮: 咨询师回应与引导
  - 第3轮: 来访者深入表达
  - 第4轮: 咨询师总结与建议

**评估维度**:
- 共情能力 (Empathy)
- 引导能力 (Guidance)
- 专业性 (Professionalism)

---

### 6️⃣ 情绪表达 (Emotion Expression)

#### EE-Situations
- **目标**: 基于给定情境生成情绪叙述
- **情绪类型**: 8 种基本情绪（喜悦、悲伤、愤怒等）
- **叙述阶段**: 5 个阶段
  1. 情绪触发事件
  2. 生理反应
  3. 认知评估
  4. 行为表现
  5. 长期影响

#### EE-Questionnaires
- **目标**: 通过心理量表评估情绪表达能力
- **形式**: 标准化问卷

---

## 数据格式

### 文件结构

```
hf_dataset/
├── Emotion Classification/
│   ├── Emotion_Classification_Emobench.jsonl      # 200条
│   └── Emotion_Classification_Finentity.jsonl     # 200条
├── Emotion Detection/
│   └── Emotion_Detection.jsonl                    # 136条
├── Emotion QA/
│   └── Emotion_QA.jsonl                          # 120条
├── Emotion Conversation/
│   └── Emotion_Conversations.jsonl               # 100条
├── Emotion Summary/
│   ├── Emotion_Summary.jsonl                     # 150条
│   └── Emotion_Summary_origin.jsonl              # 150条（原始版）
├── Emotion Expression/
│   ├── Emotion_Expression_Situations.json        # 8种情绪
│   └── Emotion_Expression_Questionnaires.json    # 1份问卷
└── README.md
```

### 数据加载

#### Python 示例

```python
import json

# 加载 JSONL 文件
def load_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

# 加载 JSON 文件
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# 使用示例
ec_data = load_jsonl('hf_dataset/Emotion Classification/Emotion_Classification_Emobench.jsonl')
ee_data = load_json('hf_dataset/Emotion Expression/Emotion_Expression_Situations.json')

print(f"EC样本数: {len(ec_data)}")
print(f"EE情绪类型: {len(ee_data['emotions'])}")
```

#### HuggingFace Datasets 库

```python
from huggingface_hub import snapshot_download

# 下载整个数据集
local_dir = snapshot_download(
    repo_id='LongEmotion/LongEmotion',
    repo_type='dataset',
    local_dir='./LongEmotion_data'
)

print(f"数据集已下载到: {local_dir}")
```

---

## 下载与使用

### 方法1: 直接从 HuggingFace 下载

```bash
# 使用 huggingface_hub
pip install huggingface_hub

python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='LongEmotion/LongEmotion',
    repo_type='dataset',
    local_dir='./LongEmotion_data'
)
"
```

### 方法2: 使用 Git LFS

```bash
git lfs install
git clone https://huggingface.co/datasets/LongEmotion/LongEmotion
```

### 方法3: 在线浏览

访问 [HuggingFace 数据集页面](https://huggingface.co/datasets/LongEmotion/LongEmotion) 在线浏览数据。

---

## 评估框架

### CoEM (Collaborative Emotional Modeling)

LongEmotion 提供了一个创新的评估框架 **CoEM**，结合了 RAG 和多智能体协作：

```
┌─────────────────────────────────────────────────┐
│              长上下文输入文本                      │
└────────────────┬────────────────────────────────┘
                 │
                 ▼
         ┌──────────────┐
         │  文本分块     │
         └──────┬───────┘
                │
                ▼
         ┌──────────────┐
         │  初始检索     │  (基于语义相似度)
         └──────┬───────┘
                │
                ▼
         ┌──────────────┐
         │  多智能体增强 │  (CoEM-Sage: GPT-4o/DeepSeek-V3)
         └──────┬───────┘
                │
                ▼
         ┌──────────────┐
         │  重排序       │  (情绪相关性)
         └──────┬───────┘
                │
                ▼
         ┌──────────────┐
         │  集成生成     │  (CoEM-Core)
         └──────┬───────┘
                │
                ▼
         ┌──────────────┐
         │  最终答案     │
         └──────────────┘
```

### 评估方法

1. **Baseline**: 直接使用 LLM 处理全文
2. **RAG**: 检索增强生成
3. **CoEM**: 多智能体协作情绪建模
4. **Self-RAG**: 自适应检索
5. **Search-O1**: 搜索优化

---

## 常见问题

### Q1: 数据集的许可协议是什么？
A: 请参考项目 LICENSE 文件。学术研究使用请引用原论文。

### Q2: 为什么 HuggingFace 上显示数据集加载错误？
A: 这是由于部分数据文件字段不一致导致的（`length` vs `token_length`）。建议使用 `snapshot_download` 下载原始文件后直接读取 JSONL/JSON 文件。

### Q3: 如何评估我的模型？
A: 参考项目中的 `evaluate.py` 和 `evaluate.sh`，配置你的模型 API 后运行评估脚本。

### Q4: 支持哪些模型？
A: 支持所有兼容 OpenAI API 格式的模型，包括：
- OpenAI GPT系列
- Anthropic Claude系列
- DeepSeek系列
- 本地部署的开源模型（如 LLaMA, Qwen 等）

### Q5: 评估需要多长时间？
A: 取决于模型和任务。完整评估所有任务预计需要数小时到一天。

### Q6: 数据集可以商用吗？
A: 请查看具体的许可协议。部分数据来源可能有使用限制。

---

## 🙏 致谢

感谢所有为 LongEmotion 数据集贡献的研究人员和数据提供者。

---

## 📧 联系方式

如有问题或建议，请：
- 提交 GitHub Issue
- 发送邮件至项目作者
- 在 HuggingFace 讨论区留言

---

## 📖 引用

如果您在研究中使用了 LongEmotion，请引用：

```bibtex
@article{liu2025longemotion,
  title={LongEmotion: Measuring Emotional Intelligence of Large Language Models in Long-Context Interaction},
  author={Liu, Weichu and Xiong, Jing and Hu, Yuxuan and Li, Zixuan and Tan, Minghuan and Mao, Ningning and Zhao, Chenyang and Wan, Zhongwei and Tao, Chaofan and Xu, Wendong and others},
  journal={arXiv preprint arXiv:2509.07403},
  year={2025}
}
```

---

<div align="center">
  <p>⭐ 如果这个项目对您有帮助，欢迎 Star！</p>
  <p>Made with ❤️ by LongEmotion Team</p>
</div>
