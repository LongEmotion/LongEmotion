#!/usr/bin/env python3
"""
LongEmotion 数据集下载和验证脚本

使用方法:
    python download_dataset.py --output_dir ./hf_dataset

功能:
    1. 从 HuggingFace 下载 LongEmotion 数据集
    2. 验证数据完整性
    3. 生成统计报告
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Any

try:
    from huggingface_hub import snapshot_download
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("警告: huggingface_hub 未安装。请运行: pip install huggingface_hub")


class LongEmotionDownloader:
    """LongEmotion 数据集下载器"""
    
    REPO_ID = "LongEmotion/LongEmotion"
    
    EXPECTED_FILES = {
        "Emotion Classification/Emotion_Classification_Emobench.jsonl": 200,
        "Emotion Classification/Emotion_Classification_Finentity.jsonl": 200,
        "Emotion Detection/Emotion_Detection.jsonl": 136,
        "Emotion QA/Emotion_QA.jsonl": 120,
        "Emotion Conversation/Emotion_Conversations.jsonl": 100,
        "Emotion Summary/Emotion_Summary.jsonl": 150,
        "Emotion Summary/Emotion_Summary_origin.jsonl": 150,
        "Emotion Expression/Emotion_Expression_Situations.json": None,  # JSON文件
        "Emotion Expression/Emotion_Expression_Questionnaires.json": None,  # JSON文件
    }
    
    def __init__(self, output_dir: str = "./hf_dataset"):
        self.output_dir = Path(output_dir)
        self.stats = {}
        
    def download(self) -> bool:
        """下载数据集"""
        if not HF_AVAILABLE:
            print("错误: 需要安装 huggingface_hub")
            print("运行: pip install huggingface_hub")
            return False
        
        print(f"正在从 HuggingFace 下载 {self.REPO_ID} ...")
        print(f"保存到: {self.output_dir.absolute()}")
        
        try:
            local_dir = snapshot_download(
                repo_id=self.REPO_ID,
                repo_type='dataset',
                local_dir=str(self.output_dir),
            )
            print(f"✓ 下载成功！数据保存在: {local_dir}")
            return True
        except Exception as e:
            print(f"✗ 下载失败: {e}")
            return False
    
    def validate(self) -> bool:
        """验证数据完整性"""
        print("\n" + "="*80)
        print("验证数据完整性...")
        print("="*80)
        
        all_valid = True
        
        for file_path, expected_count in self.EXPECTED_FILES.items():
            full_path = self.output_dir / file_path
            
            if not full_path.exists():
                print(f"✗ 文件缺失: {file_path}")
                all_valid = False
                continue
            
            # 读取文件并统计
            try:
                if file_path.endswith('.jsonl'):
                    with open(full_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        actual_count = len(lines)
                        
                    # 验证每行都是有效的JSON
                    for i, line in enumerate(lines[:5], 1):  # 检查前5行
                        try:
                            json.loads(line)
                        except json.JSONDecodeError as e:
                            print(f"✗ {file_path} 第{i}行JSON格式错误: {e}")
                            all_valid = False
                    
                    if expected_count and actual_count != expected_count:
                        print(f"⚠ {file_path}: 预期{expected_count}条，实际{actual_count}条")
                    else:
                        print(f"✓ {file_path}: {actual_count}条数据")
                    
                    self.stats[file_path] = actual_count
                    
                else:  # JSON文件
                    with open(full_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        
                    if isinstance(data, list):
                        count = len(data)
                    elif isinstance(data, dict):
                        count = f"{len(data)} 字段"
                    else:
                        count = "1 对象"
                    
                    print(f"✓ {file_path}: {count}")
                    self.stats[file_path] = str(count)
                    
            except Exception as e:
                print(f"✗ {file_path} 读取失败: {e}")
                all_valid = False
        
        print("="*80)
        if all_valid:
            print("✓ 所有文件验证通过！")
        else:
            print("⚠ 部分文件验证失败，请检查")
        
        return all_valid
    
    def analyze(self):
        """分析数据集统计信息"""
        print("\n" + "="*80)
        print("数据集统计分析")
        print("="*80)
        
        # Emotion Classification - Emobench
        self._analyze_emotion_classification_emobench()
        
        # Emotion Classification - Finentity
        self._analyze_emotion_classification_finentity()
        
        # Emotion Detection
        self._analyze_emotion_detection()
        
        # Emotion QA
        self._analyze_emotion_qa()
        
        # Emotion Conversation
        self._analyze_emotion_conversation()
        
        # Emotion Summary
        self._analyze_emotion_summary()
        
        print("="*80)
    
    def _load_jsonl(self, file_path: str) -> List[Dict]:
        """加载JSONL文件"""
        full_path = self.output_dir / file_path
        with open(full_path, 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f]
    
    def _load_json(self, file_path: str) -> Any:
        """加载JSON文件"""
        full_path = self.output_dir / file_path
        with open(full_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _analyze_emotion_classification_emobench(self):
        """分析 Emotion Classification Emobench 数据"""
        try:
            data = self._load_jsonl("Emotion Classification/Emotion_Classification_Emobench.jsonl")
            avg_length = sum(d.get('length', 0) for d in data) / len(data)
            emotions = set(d['label'] for d in data)
            
            print(f"\n📊 Emotion Classification (Emobench)")
            print(f"   样本数: {len(data)}")
            print(f"   平均长度: {avg_length:.2f} tokens")
            print(f"   情绪类别数: {len(emotions)}")
            print(f"   情绪示例: {', '.join(list(emotions)[:5])}, ...")
        except Exception as e:
            print(f"✗ 分析 Emobench 失败: {e}")
    
    def _analyze_emotion_classification_finentity(self):
        """分析 Emotion Classification Finentity 数据"""
        try:
            data = self._load_jsonl("Emotion Classification/Emotion_Classification_Finentity.jsonl")
            avg_length = sum(d.get('token_length', 0) for d in data) / len(data)
            emotions = set(d['label'] for d in data)
            
            print(f"\n📊 Emotion Classification (Finentity)")
            print(f"   样本数: {len(data)}")
            print(f"   平均长度: {avg_length:.2f} tokens")
            print(f"   情绪类别: {', '.join(sorted(emotions))}")
        except Exception as e:
            print(f"✗ 分析 Finentity 失败: {e}")
    
    def _analyze_emotion_detection(self):
        """分析 Emotion Detection 数据"""
        try:
            data = self._load_jsonl("Emotion Detection/Emotion_Detection.jsonl")
            avg_length = sum(d.get('length', 0) for d in data) / len(data)
            
            print(f"\n📊 Emotion Detection")
            print(f"   样本数: {len(data)}")
            print(f"   平均长度: {avg_length:.2f} tokens")
        except Exception as e:
            print(f"✗ 分析 Emotion Detection 失败: {e}")
    
    def _analyze_emotion_qa(self):
        """分析 Emotion QA 数据"""
        try:
            data = self._load_jsonl("Emotion QA/Emotion_QA.jsonl")
            sources = set(d['source'] for d in data)
            
            print(f"\n📊 Emotion QA")
            print(f"   样本数: {len(data)}")
            print(f"   来源文献数: {len(sources)}")
        except Exception as e:
            print(f"✗ 分析 Emotion QA 失败: {e}")
    
    def _analyze_emotion_conversation(self):
        """分析 Emotion Conversation 数据"""
        try:
            data = self._load_jsonl("Emotion Conversation/Emotion_Conversations.jsonl")
            total_stages = sum(len(d.get('stages', [])) for d in data)
            
            print(f"\n📊 Emotion Conversation")
            print(f"   对话数: {len(data)}")
            print(f"   总轮次: {total_stages}")
            print(f"   平均轮次: {total_stages / len(data):.2f}")
        except Exception as e:
            print(f"✗ 分析 Emotion Conversation 失败: {e}")
    
    def _analyze_emotion_summary(self):
        """分析 Emotion Summary 数据"""
        try:
            data = self._load_jsonl("Emotion Summary/Emotion_Summary.jsonl")
            
            print(f"\n📊 Emotion Summary")
            print(f"   样本数: {len(data)}")
            print(f"   字段: causes, symptoms, treatment_process, treatment_effect")
        except Exception as e:
            print(f"✗ 分析 Emotion Summary 失败: {e}")
    
    def generate_report(self, output_file: str = "dataset_report.txt"):
        """生成数据集报告"""
        report_path = self.output_dir / output_file
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("LongEmotion 数据集报告\n")
            f.write("="*80 + "\n\n")
            
            f.write("文件统计:\n")
            f.write("-"*80 + "\n")
            for file_path, count in self.stats.items():
                f.write(f"{file_path}: {count}\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write(f"报告生成于: {self.output_dir.absolute()}\n")
            f.write("="*80 + "\n")
        
        print(f"\n✓ 报告已保存: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="LongEmotion 数据集下载和验证工具"
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./hf_dataset',
        help='数据集保存目录 (默认: ./hf_dataset)'
    )
    parser.add_argument(
        '--skip_download',
        action='store_true',
        help='跳过下载，仅验证已有数据'
    )
    parser.add_argument(
        '--skip_analysis',
        action='store_true',
        help='跳过详细分析'
    )
    
    args = parser.parse_args()
    
    downloader = LongEmotionDownloader(args.output_dir)
    
    # 下载
    if not args.skip_download:
        if not downloader.download():
            return
    
    # 验证
    if not downloader.validate():
        print("\n⚠ 数据验证失败，请检查数据完整性")
        return
    
    # 分析
    if not args.skip_analysis:
        downloader.analyze()
    
    # 生成报告
    downloader.generate_report()
    
    print("\n" + "="*80)
    print("✓ 全部完成！")
    print(f"数据位置: {Path(args.output_dir).absolute()}")
    print("="*80)


if __name__ == "__main__":
    main()
