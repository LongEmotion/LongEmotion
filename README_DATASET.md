# LongEmotion 数据集

<div align="center">
  <img src="LongEmotion-logo.png" alt="LongEmotion Logo" width="200">
  
  <h3>测量大语言模型在长上下文交互中的情商</h3>
  
  [![Paper](https://img.shields.io/badge/arXiv-2509.07403-b31b1b.svg)](https://arxiv.org/abs/2509.07403)
  [![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace-yellow)](https://huggingface.co/datasets/LongEmotion/LongEmotion)
</div>

---

## 🌟 数据集简介

**LongEmotion** 是首个专门评估大语言模型（LLMs）在**长上下文场景**下**情商（Emotional Intelligence）**的综合性基准。

### 核心特点

- 🎯 **长上下文**: 平均长度 15,000+ tokens，最长达 43,588 tokens
- 📊 **全面评估**: 情绪识别、心理知识应用、共情生成三大维度
- 🔬 **多样任务**: 6 大任务类型，8 个子任务，1,156+ 样本
- 🌐 **真实数据**: 来自心理咨询案例、学术文献、金融文档等

---

## 📊 数据集统计

| 任务 | 类型 | 样本数 | 平均长度 | 评估指标 |
|------|------|--------|----------|----------|
| **EC-Emobench** | 分类 | 200 | 19,345 tokens | Accuracy |
| **EC-Finentity** | 分类 | 200 | **43,588 tokens** | Accuracy |
| **Emotion Detection** | 检测 | 136 | 4,592 tokens | Accuracy |
| **Emotion QA** | 问答 | 120 | - | F1 Score |
| **Emotion Conversation** | 对话 | 100 (400轮) | - | LLM-Judge |
| **Emotion Summary** | 摘要 | 150 | - | LLM-Judge |
| **Emotion Expression** | 生成 | 8类+1卷 | - | LLM-Judge |

**总计**: 1,156+ 样本，173MB 数据

---

## 🚀 快速开始

### 1️⃣ 下载数据集

```bash
# 方法1: 使用提供的脚本（推荐）
python download_dataset.py --output_dir ./hf_dataset

# 方法2: 手动下载
pip install huggingface_hub
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='LongEmotion/LongEmotion',
    repo_type='dataset',
    local_dir='./hf_dataset'
)
"
```

### 2️⃣ 验证数据

```bash
# 运行测试脚本
python test_load_data.py --data_dir hf_dataset
```

预期输出：
```
🎉 所有测试通过！数据集可以正常使用。
```

### 3️⃣ 加载数据示例

```python
import json

# 加载情绪分类数据
with open('hf_dataset/Emotion Classification/Emotion_Classification_Emobench.jsonl', 'r') as f:
    data = [json.loads(line) for line in f]

print(f"样本数: {len(data)}")
print(f"第一个样本: {data[0]['subject']} - {data[0]['label']}")
```

---

## 📂 数据集结构

```
hf_dataset/
├── Emotion Classification/
│   ├── Emotion_Classification_Emobench.jsonl      # 200样本, 84种情绪
│   └── Emotion_Classification_Finentity.jsonl     # 200样本, 3类情感
├── Emotion Detection/
│   └── Emotion_Detection.jsonl                    # 136样本, 检测异常情绪
├── Emotion QA/
│   └── Emotion_QA.jsonl                          # 120问答, 30篇文献
├── Emotion Conversation/
│   └── Emotion_Conversations.jsonl               # 100对话, 400轮次
├── Emotion Summary/
│   ├── Emotion_Summary.jsonl                     # 150案例摘要
│   └── Emotion_Summary_origin.jsonl              # 原始版本
└── Emotion Expression/
    ├── Emotion_Expression_Situations.json        # 8种情绪类型
    └── Emotion_Expression_Questionnaires.json    # PANAS问卷
```

---

## 📖 任务说明

### 情绪识别 (Emotion Recognition)

#### 1. Emotion Classification - Emobench
- **任务**: 在长篇小说中识别角色的细粒度情绪
- **挑战**: 84种情绪类别（含复合情绪）
- **示例**: "Delight", "Anger & Disappointment", "Joy & Gratitude"

#### 2. Emotion Classification - Finentity
- **任务**: 在极长金融文档中判断实体情感
- **挑战**: 平均43k+ tokens，最长可达80k+
- **类别**: Positive, Neutral, Negative

#### 3. Emotion Detection
- **任务**: 从多个片段中检测情绪异常的片段
- **格式**: N选1 多选题

### 知识应用 (Knowledge Application)

#### 4. Emotion QA
- **任务**: 基于心理学文献回答专业问题
- **来源**: 30篇心理健康领域学术论文
- **主题**: 压力管理、手机使用、身体活动、AI影响等

#### 5. Emotion Summary
- **任务**: 从心理咨询报告提取结构化信息
- **字段**: 病因、症状、治疗过程、治疗效果

### 共情生成 (Empathetic Generation)

#### 6. Emotion Conversation
- **任务**: 模拟心理咨询对话（4轮）
- **评估**: 共情能力、引导能力、专业性

#### 7. Emotion Expression
- **任务**: 基于情境生成情绪叙述
- **阶段**: 触发事件 → 生理反应 → 认知评估 → 行为表现 → 长期影响

---

## 🔧 评估方法

LongEmotion 支持多种评估方法：

| 方法 | 说明 | 适用场景 |
|------|------|---------|
| **Baseline** | 直接处理全文 | 短文本或大上下文窗口 |
| **RAG** | 检索增强生成 | 长文本信息检索 |
| **CoEM** | 协作情绪建模 | 长文本情绪任务（推荐）|
| **Self-RAG** | 自适应检索 | 动态决策场景 |
| **Search-O1** | 搜索优化 | 复杂推理任务 |

### CoEM 框架

```
输入文本 → 分块 → 初始检索 → 多智能体增强 → 重排序 → 集成生成 → 输出
```

---

## 📚 文档导航

本项目提供完整的文档体系：

| 文档 | 内容 | 适用场景 |
|------|------|---------|
| [**QUICKSTART.md**](QUICKSTART.md) | 快速上手指南 | 首次使用必读 |
| [**DATASET_INFO.md**](DATASET_INFO.md) | 数据集详细技术文档 | 深入了解数据结构 |
| [**DATASET_README_CN.md**](DATASET_README_CN.md) | 中文完整说明 | 全面介绍和FAQ |
| [**DATA_DOWNLOAD_SUMMARY.md**](DATA_DOWNLOAD_SUMMARY.md) | 下载完成报告 | 查看下载状态 |

### 工具脚本

| 脚本 | 功能 |
|------|------|
| `download_dataset.py` | 自动下载和验证数据集 |
| `test_load_data.py` | 测试数据加载 |
| `evaluate.py` | 评估模型性能 |

---

## 💻 运行评估

### 配置模型

编辑 `evaluate.sh`：

```bash
--model_name "gpt-4o"
--model_api_key "your_api_key"
--model_url "https://api.openai.com/v1"
```

### 运行单个任务

```bash
bash evaluate.sh baseline Emotion_Classification_Emobench
```

### 批量评估

```bash
bash run.sh
```

---

## 📊 数据样例

### Emotion Classification

```json
{
  "id": 1,
  "content": "[20,000+ tokens 长文本]",
  "subject": "Elizabeth",
  "label": "Delight",
  "choices": ["Delight", "Disappointment", "Anger", "Pessimism"],
  "length": 20082
}
```

### Emotion QA

```json
{
  "number": "File14-2",
  "problem": "What is intrinsic capacity?",
  "answer": "A comprehensive indicator...",
  "source": "Association Between Daily Internet Use...",
  "context": "[长篇学术论文]"
}
```

### Emotion Conversation

```json
{
  "id": 1,
  "description": "患者因躯体症状反复住院...",
  "stages": [
    {"stage": "Reception", "conversations": "..."},
    {"stage": "Inquiry", "conversations": "..."},
    {"stage": "Treatment", "conversations": "..."},
    {"stage": "Follow-up", "conversations": "..."}
  ]
}
```

---

## 🔗 相关链接

- **📄 论文**: [arXiv:2509.07403](https://arxiv.org/abs/2509.07403)
- **🤗 HuggingFace**: [LongEmotion/LongEmotion](https://huggingface.co/datasets/LongEmotion/LongEmotion)
- **💻 GitHub**: (待补充)

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

## ❓ 常见问题

### Q: HuggingFace 显示数据集加载错误？
**A**: 这是由于字段不一致导致的。请使用 `download_dataset.py` 下载原始文件，然后直接读取 JSONL/JSON。

### Q: 如何验证数据完整性？
**A**: 运行 `python test_load_data.py`，所有测试通过即表示数据正常。

### Q: 支持哪些模型？
**A**: 支持所有 OpenAI API 兼容的模型，包括 GPT、Claude、DeepSeek、本地部署的开源模型等。

### Q: 评估需要多久？
**A**: 完整评估约需数小时到一天，取决于模型和任务。

---

## 🙏 致谢

感谢所有为 LongEmotion 数据集贡献的研究人员和数据提供者。

---

<div align="center">
  <p>⭐ 如果这个项目对您有帮助，欢迎 Star！</p>
  <p>Made with ❤️ by LongEmotion Team</p>
  <p><i>最后更新: 2026-01-17</i></p>
</div>
