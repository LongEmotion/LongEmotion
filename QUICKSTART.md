# LongEmotion 快速开始指南

本文档帮助你快速上手 LongEmotion 数据集和评估框架。

## 📋 目录

- [环境准备](#环境准备)
- [下载数据集](#下载数据集)
- [数据加载示例](#数据加载示例)
- [运行评估](#运行评估)
- [查看结果](#查看结果)

---

## 环境准备

### 1. 创建 Python 环境

```bash
# 创建 conda 环境（推荐）
conda create -n LongEmotion python=3.10
conda activate LongEmotion

# 或使用 venv
python3 -m venv longemotion_env
source longemotion_env/bin/activate  # Linux/Mac
# longemotion_env\Scripts\activate  # Windows
```

### 2. 安装依赖

```bash
cd LongEmotion
pip install -r requirements.txt
```

### 3. 验证安装

```bash
python -c "import torch; import transformers; print('✓ 环境准备完成')"
```

---

## 下载数据集

### 方法1: 使用提供的脚本（推荐）

```bash
python download_dataset.py --output_dir ./hf_dataset
```

脚本会自动：
- ✓ 从 HuggingFace 下载数据集
- ✓ 验证数据完整性
- ✓ 生成统计报告

### 方法2: 手动下载

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id='LongEmotion/LongEmotion',
    repo_type='dataset',
    local_dir='./hf_dataset'
)
```

### 方法3: 使用已有数据

如果你已经有数据文件，可以验证完整性：

```bash
python download_dataset.py --skip_download --output_dir ./hf_dataset
```

---

## 数据加载示例

### Python 示例

创建 `test_load.py`：

```python
import json
from pathlib import Path

# 数据目录
DATA_DIR = Path("hf_dataset")

# 1. 加载情绪分类数据
def load_emotion_classification():
    print("="*60)
    print("加载 Emotion Classification (Emobench)")
    print("="*60)
    
    file_path = DATA_DIR / "Emotion Classification/Emotion_Classification_Emobench.jsonl"
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    
    print(f"总样本数: {len(data)}")
    print(f"\n第一个样本:")
    print(f"  ID: {data[0]['id']}")
    print(f"  Subject: {data[0]['subject']}")
    print(f"  Label: {data[0]['label']}")
    print(f"  Content Length: {data[0]['length']} tokens")
    print(f"  Choices: {data[0]['choices']}")
    print(f"  Content Preview: {data[0]['content'][:200]}...")
    
    return data

# 2. 加载情绪问答数据
def load_emotion_qa():
    print("\n" + "="*60)
    print("加载 Emotion QA")
    print("="*60)
    
    file_path = DATA_DIR / "Emotion QA/Emotion_QA.jsonl"
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    
    print(f"总样本数: {len(data)}")
    print(f"\n第一个样本:")
    print(f"  Number: {data[0]['number']}")
    print(f"  Problem: {data[0]['problem'][:100]}...")
    print(f"  Answer: {data[0]['answer'][:100]}...")
    print(f"  Source: {data[0]['source'][:80]}...")
    
    return data

# 3. 加载情绪对话数据
def load_emotion_conversation():
    print("\n" + "="*60)
    print("加载 Emotion Conversation")
    print("="*60)
    
    file_path = DATA_DIR / "Emotion Conversation/Emotion_Conversations.jsonl"
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    
    print(f"总对话数: {len(data)}")
    print(f"\n第一个对话:")
    print(f"  ID: {data[0]['id']}")
    print(f"  Description: {data[0]['description'][:100]}...")
    print(f"  Stages: {len(data[0]['stages'])} 轮")
    
    for stage in data[0]['stages'][:2]:  # 显示前两轮
        print(f"    Stage {stage['stage']}: {stage['content'][:80]}...")
    
    return data

# 4. 加载情绪表达数据
def load_emotion_expression():
    print("\n" + "="*60)
    print("加载 Emotion Expression")
    print("="*60)
    
    # Situations
    file_path = DATA_DIR / "Emotion Expression/Emotion_Expression_Situations.json"
    with open(file_path, 'r', encoding='utf-8') as f:
        situations = json.load(f)
    
    print(f"情绪类型数: {len(situations['emotions'])}")
    print(f"情绪类型: {[e['emotion_name'] for e in situations['emotions'][:3]]}")
    
    return situations

# 主函数
if __name__ == "__main__":
    print("🚀 LongEmotion 数据加载示例\n")
    
    ec_data = load_emotion_classification()
    qa_data = load_emotion_qa()
    conv_data = load_emotion_conversation()
    ee_data = load_emotion_expression()
    
    print("\n" + "="*60)
    print("✓ 所有数据加载成功！")
    print("="*60)
```

运行测试：

```bash
python test_load.py
```

---

## 运行评估

### 1. 配置模型

编辑 `evaluate.sh`，设置你的模型配置：

```bash
#!/usr/bin/env bash

WORK_DIR=$(dirname $(readlink -f $0))
METHOD=$1   # baseline | rag | coem | self-rag | search-o1
TASK=$2     # 任务名称

PYTHONPATH="${WORK_DIR}/src" python "${WORK_DIR}/evaluate.py" \
  --task "${TASK}" \
  --method "${METHOD}" \
  --data_dir "${WORK_DIR}/hf_dataset" \
  --prompts_dir "${WORK_DIR}/prompts" \
  --base_dir "${WORK_DIR}/evaluations" \
  --model_name "gpt-4o" \
  --model_api_key "your_api_key_here" \
  --model_url "https://api.openai.com/v1" \
  --model_name_coem_sage "gpt-4o" \
  --model_api_key_coem_sage "your_api_key_here" \
  --model_url_coem_sage "https://api.openai.com/v1" \
  --evaluator_name "gpt-4o" \
  --evaluator_api_key "your_api_key_here" \
  --evaluator_url "https://api.openai.com/v1"
```

**支持的模型配置：**

#### OpenAI
```bash
--model_name "gpt-4o"
--model_api_key "sk-..."
--model_url "https://api.openai.com/v1"
```

#### DeepSeek
```bash
--model_name "deepseek-chat"
--model_api_key "sk-..."
--model_url "https://api.deepseek.com/v1"
```

#### Claude (通过 OpenAI-compatible endpoint)
```bash
--model_name "claude-3-5-sonnet-20241022"
--model_api_key "sk-..."
--model_url "your_endpoint"
```

#### 本地模型
```bash
--model_name "Qwen/Qwen2.5-72B-Instruct"
--model_url "local"
```

### 2. 选择评估方法

LongEmotion 支持 5 种评估方法：

| 方法 | 说明 | 适用场景 |
|------|------|---------|
| `baseline` | 直接处理全文 | 短文本或小模型上下文窗口足够 |
| `rag` | 检索增强生成 | 长文本，需要信息检索 |
| `coem` | 协作情绪建模 | 长文本情绪任务（推荐） |
| `self-rag` | 自适应检索 | 需要动态决策的场景 |
| `search-o1` | 搜索优化 | 复杂推理任务 |

### 3. 选择评估任务

可用任务列表：

```bash
# 情绪分类
Emotion_Classification_Emobench
Emotion_Classification_Finentity

# 情绪检测
Emotion_Detection

# 情绪问答
Emotion_QA

# 情绪对话
Emotion_Conversations

# 情绪摘要
Emotion_Summary

# 情绪表达
Emotion_Expression
```

### 4. 运行评估

编辑 `run.sh`：

```bash
#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

# 运行单个任务
bash evaluate.sh baseline Emotion_Classification_Emobench

# 运行多个任务
# bash evaluate.sh baseline Emotion_QA
# bash evaluate.sh coem Emotion_Conversations
```

执行：

```bash
bash run.sh
```

### 5. 批量评估所有任务

创建 `run_all.sh`：

```bash
#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

METHODS=("baseline" "rag" "coem")
TASKS=(
    "Emotion_Classification_Emobench"
    "Emotion_Classification_Finentity"
    "Emotion_Detection"
    "Emotion_QA"
    "Emotion_Conversations"
    "Emotion_Summary"
)

for method in "${METHODS[@]}"; do
    for task in "${TASKS[@]}"; do
        echo "=========================================="
        echo "运行: $method - $task"
        echo "=========================================="
        bash evaluate.sh "$method" "$task"
    done
done

echo "✓ 全部评估完成！"
```

运行：

```bash
chmod +x run_all.sh
bash run_all.sh
```

---

## 查看结果

### 结果目录结构

```
evaluations/
├── Emotion_Classification_Emobench/
│   ├── baseline/
│   │   ├── results.json
│   │   └── metrics.json
│   ├── rag/
│   └── coem/
├── Emotion_QA/
└── logs.txt
```

### 查看评估指标

```python
import json

# 读取结果
with open('evaluations/Emotion_Classification_Emobench/baseline/metrics.json', 'r') as f:
    metrics = json.load(f)

print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"Total Samples: {metrics['total']}")
print(f"Correct: {metrics['correct']}")
```

### 对比不同方法

创建 `compare_results.py`：

```python
import json
from pathlib import Path
from tabulate import tabulate

def compare_methods(task):
    methods = ['baseline', 'rag', 'coem']
    results = []
    
    for method in methods:
        metrics_file = Path(f'evaluations/{task}/{method}/metrics.json')
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            results.append([
                method,
                f"{metrics.get('accuracy', 0):.4f}",
                metrics.get('total', 0),
                metrics.get('correct', 0)
            ])
    
    print(f"\n{'='*60}")
    print(f"任务: {task}")
    print(f"{'='*60}")
    print(tabulate(results, headers=['Method', 'Accuracy', 'Total', 'Correct']))

# 对比所有任务
tasks = [
    'Emotion_Classification_Emobench',
    'Emotion_Classification_Finentity',
    'Emotion_Detection',
    'Emotion_QA',
]

for task in tasks:
    compare_results(task)
```

---

## 💡 小贴士

### 1. 节省 API 成本

```bash
# 先在小数据集上测试
head -10 hf_dataset/Emotion\ QA/Emotion_QA.jsonl > test_data.jsonl

# 或者修改评估脚本，限制样本数
python evaluate.py --task Emotion_QA --max_samples 10
```

### 2. 使用缓存

评估脚本会自动缓存 embedding 和检索结果，重复运行会更快。

### 3. 并行评估

如果有多个 GPU，可以并行运行不同任务：

```bash
# Terminal 1
CUDA_VISIBLE_DEVICES=0 bash evaluate.sh baseline Emotion_QA &

# Terminal 2
CUDA_VISIBLE_DEVICES=1 bash evaluate.sh baseline Emotion_Detection &
```

### 4. 监控进度

```bash
# 实时查看日志
tail -f evaluations/logs.txt
```

---

## 🐛 常见问题

### Q1: API 调用失败

```bash
# 检查 API key 是否正确
echo $OPENAI_API_KEY

# 测试连接
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY"
```

### Q2: 内存不足

```bash
# 减小 batch size
# 在 evaluate.py 中修改 batch_size 参数
```

### Q3: 数据路径错误

```bash
# 确保数据路径正确
ls -la hf_dataset/Emotion\ Classification/

# 修改 evaluate.sh 中的 --data_dir 参数
```

### Q4: Self-RAG 模型启动失败

```bash
# 确保模型路径正确
ls -la ~/selfrag_llama2_7b

# 启动 vLLM 服务器
vllm serve ~/selfrag_llama2_7b \
    --gpu-memory-utilization 0.5 \
    --dtype float16 \
    --port 8010
```

---

## 📚 下一步

- 阅读 [DATASET_INFO.md](DATASET_INFO.md) 了解数据集详细信息
- 阅读 [DATASET_README_CN.md](DATASET_README_CN.md) 查看完整文档
- 查看论文: https://arxiv.org/abs/2509.07403
- 访问 HuggingFace: https://huggingface.co/datasets/LongEmotion/LongEmotion

---

## 🙏 获取帮助

遇到问题？
- 查看 `evaluations/logs.txt` 日志文件
- 提交 GitHub Issue
- 联系项目作者

---

祝你评估顺利！🚀
