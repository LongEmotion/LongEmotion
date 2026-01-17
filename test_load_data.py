#!/usr/bin/env python3
"""
LongEmotion 数据加载测试脚本

用途: 验证数据集是否正确下载并可以正常加载
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any


class DatasetTester:
    """数据集测试器"""
    
    def __init__(self, data_dir: str = "hf_dataset"):
        self.data_dir = Path(data_dir)
        self.all_passed = True
        
    def test_emotion_classification_emobench(self):
        """测试 Emotion Classification Emobench"""
        print("\n" + "="*80)
        print("测试 1: Emotion Classification (Emobench)")
        print("="*80)
        
        try:
            file_path = self.data_dir / "Emotion Classification/Emotion_Classification_Emobench.jsonl"
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = [json.loads(line) for line in f]
            
            # 验证数据量
            assert len(data) == 200, f"预期 200 样本，实际 {len(data)}"
            
            # 验证字段
            required_fields = ['id', 'content', 'subject', 'label', 'source', 'choices', 'length']
            for field in required_fields:
                assert field in data[0], f"缺少字段: {field}"
            
            # 显示示例
            sample = data[0]
            print(f"✓ 加载成功: {len(data)} 个样本")
            print(f"\n示例数据:")
            print(f"  ID: {sample['id']}")
            print(f"  Subject: {sample['subject']}")
            print(f"  Label: {sample['label']}")
            print(f"  Length: {sample['length']} tokens")
            print(f"  Choices: {len(sample['choices'])} 个选项")
            print(f"  Content preview: {sample['content'][:150]}...")
            
            print("\n✓ 测试通过")
            return True
            
        except Exception as e:
            print(f"\n✗ 测试失败: {e}")
            self.all_passed = False
            return False
    
    def test_emotion_classification_finentity(self):
        """测试 Emotion Classification Finentity"""
        print("\n" + "="*80)
        print("测试 2: Emotion Classification (Finentity)")
        print("="*80)
        
        try:
            file_path = self.data_dir / "Emotion Classification/Emotion_Classification_Finentity.jsonl"
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = [json.loads(line) for line in f]
            
            assert len(data) == 200, f"预期 200 样本，实际 {len(data)}"
            
            sample = data[0]
            print(f"✓ 加载成功: {len(data)} 个样本")
            print(f"\n示例数据:")
            print(f"  ID: {sample['id']}")
            print(f"  Subject: {sample['subject']}")
            print(f"  Label: {sample['label']}")
            print(f"  Token Length: {sample.get('token_length', 'N/A')} tokens")
            print(f"  Choices: {sample['choices']}")
            
            # 检查超长上下文
            avg_length = sum(s.get('token_length', 0) for s in data) / len(data)
            print(f"\n  平均长度: {avg_length:.2f} tokens (超长上下文！)")
            
            print("\n✓ 测试通过")
            return True
            
        except Exception as e:
            print(f"\n✗ 测试失败: {e}")
            self.all_passed = False
            return False
    
    def test_emotion_detection(self):
        """测试 Emotion Detection"""
        print("\n" + "="*80)
        print("测试 3: Emotion Detection")
        print("="*80)
        
        try:
            file_path = self.data_dir / "Emotion Detection/Emotion_Detection.jsonl"
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = [json.loads(line) for line in f]
            
            assert len(data) == 136, f"预期 136 样本，实际 {len(data)}"
            
            sample = data[0]
            print(f"✓ 加载成功: {len(data)} 个样本")
            print(f"\n示例数据:")
            print(f"  Label: {sample['label']}")
            print(f"  Length: {sample.get('length', 'N/A')} tokens")
            print(f"  Text options: {list(sample['text'].keys()) if isinstance(sample['text'], dict) else 'N/A'}")
            
            print("\n✓ 测试通过")
            return True
            
        except Exception as e:
            print(f"\n✗ 测试失败: {e}")
            self.all_passed = False
            return False
    
    def test_emotion_qa(self):
        """测试 Emotion QA"""
        print("\n" + "="*80)
        print("测试 4: Emotion QA")
        print("="*80)
        
        try:
            file_path = self.data_dir / "Emotion QA/Emotion_QA.jsonl"
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = [json.loads(line) for line in f]
            
            assert len(data) == 120, f"预期 120 样本，实际 {len(data)}"
            
            sample = data[0]
            sources = set(s['source'] for s in data)
            
            print(f"✓ 加载成功: {len(data)} 个样本")
            print(f"\n示例数据:")
            print(f"  Number: {sample['number']}")
            print(f"  Problem: {sample['problem'][:100]}...")
            print(f"  Answer: {sample['answer'][:100]}...")
            print(f"  Source: {sample['source'][:60]}...")
            print(f"\n  来源文献数: {len(sources)}")
            
            print("\n✓ 测试通过")
            return True
            
        except Exception as e:
            print(f"\n✗ 测试失败: {e}")
            self.all_passed = False
            return False
    
    def test_emotion_conversation(self):
        """测试 Emotion Conversation"""
        print("\n" + "="*80)
        print("测试 5: Emotion Conversation")
        print("="*80)
        
        try:
            file_path = self.data_dir / "Emotion Conversation/Emotion_Conversations.jsonl"
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = [json.loads(line) for line in f]
            
            assert len(data) == 100, f"预期 100 样本，实际 {len(data)}"
            
            sample = data[0]
            total_stages = sum(len(d.get('stages', [])) for d in data)
            
            print(f"✓ 加载成功: {len(data)} 个对话")
            print(f"\n示例数据:")
            print(f"  ID: {sample['id']}")
            print(f"  Description: {sample['description'][:100]}...")
            print(f"  Stages: {len(sample['stages'])} 轮")
            
            if sample['stages']:
                print(f"\n  第一轮对话:")
                stage_info = sample['stages'][0]
                print(f"    Stage {stage_info['stage']}")
                # conversations 字段可能是字符串或列表
                conversations = stage_info.get('conversations', '')
                if isinstance(conversations, str):
                    print(f"    Conversations: {conversations[:100]}...")
                elif isinstance(conversations, list) and conversations:
                    print(f"    Conversations: {len(conversations)} 条消息")
            
            print(f"\n  总对话轮次: {total_stages}")
            print(f"  平均轮次: {total_stages / len(data):.2f}")
            
            print("\n✓ 测试通过")
            return True
            
        except Exception as e:
            print(f"\n✗ 测试失败: {e}")
            self.all_passed = False
            return False
    
    def test_emotion_summary(self):
        """测试 Emotion Summary"""
        print("\n" + "="*80)
        print("测试 6: Emotion Summary")
        print("="*80)
        
        try:
            file_path = self.data_dir / "Emotion Summary/Emotion_Summary.jsonl"
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = [json.loads(line) for line in f]
            
            assert len(data) == 150, f"预期 150 样本，实际 {len(data)}"
            
            sample = data[0]
            required_fields = ['causes', 'symptoms', 'treatment_process', 'treatment_effect']
            
            print(f"✓ 加载成功: {len(data)} 个样本")
            print(f"\n示例数据:")
            print(f"  ID: {sample['id']}")
            
            # case_description 可能是字典
            case_desc = sample.get('case_description', '')
            if isinstance(case_desc, dict):
                print(f"  Case Description: (dict with {len(case_desc)} keys)")
            elif isinstance(case_desc, str):
                print(f"  Case Description: {case_desc[:100]}...")
            
            print(f"\n  关键字段:")
            for field in required_fields:
                if field in sample:
                    content = sample[field]
                    if isinstance(content, dict):
                        print(f"    {field}: (dict with {len(content)} keys)")
                    elif isinstance(content, str):
                        print(f"    {field}: {content[:60]}...")
                    else:
                        print(f"    {field}: {str(content)[:60]}...")
            
            print("\n✓ 测试通过")
            return True
            
        except Exception as e:
            print(f"\n✗ 测试失败: {e}")
            self.all_passed = False
            return False
    
    def test_emotion_expression_situations(self):
        """测试 Emotion Expression Situations"""
        print("\n" + "="*80)
        print("测试 7: Emotion Expression (Situations)")
        print("="*80)
        
        try:
            file_path = self.data_dir / "Emotion Expression/Emotion_Expression_Situations.json"
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            assert 'emotions' in data, "缺少 'emotions' 字段"
            
            emotions = data['emotions']
            print(f"✓ 加载成功: {len(emotions)} 种情绪类型")
            
            if emotions:
                print(f"\n示例情绪:")
                for i, emotion in enumerate(emotions[:3], 1):
                    print(f"  {i}. {emotion.get('emotion_name', 'N/A')}")
                    situations = emotion.get('situations', [])
                    if situations:
                        print(f"     情境数: {len(situations)}")
                        print(f"     示例: {situations[0][:60]}...")
            
            print("\n✓ 测试通过")
            return True
            
        except Exception as e:
            print(f"\n✗ 测试失败: {e}")
            self.all_passed = False
            return False
    
    def test_emotion_expression_questionnaires(self):
        """测试 Emotion Expression Questionnaires"""
        print("\n" + "="*80)
        print("测试 8: Emotion Expression (Questionnaires)")
        print("="*80)
        
        try:
            file_path = self.data_dir / "Emotion Expression/Emotion_Expression_Questionnaires.json"
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            assert isinstance(data, list), "应该是列表格式"
            assert len(data) > 0, "问卷列表为空"
            
            questionnaire = data[0]
            print(f"✓ 加载成功: {len(data)} 份问卷")
            
            print(f"\n问卷信息:")
            print(f"  Name: {questionnaire.get('name', 'N/A')}")
            print(f"  Questions: {len(questionnaire.get('questions', []))} 个问题")
            print(f"  Compute Mode: {questionnaire.get('compute_mode', 'N/A')}")
            print(f"  Scale: {questionnaire.get('scale', 'N/A')}")
            
            print("\n✓ 测试通过")
            return True
            
        except Exception as e:
            print(f"\n✗ 测试失败: {e}")
            self.all_passed = False
            return False
    
    def run_all_tests(self):
        """运行所有测试"""
        print("\n" + "█"*80)
        print("█" + " "*78 + "█")
        print("█" + "  LongEmotion 数据集加载测试".center(78) + "█")
        print("█" + " "*78 + "█")
        print("█"*80)
        
        tests = [
            self.test_emotion_classification_emobench,
            self.test_emotion_classification_finentity,
            self.test_emotion_detection,
            self.test_emotion_qa,
            self.test_emotion_conversation,
            self.test_emotion_summary,
            self.test_emotion_expression_situations,
            self.test_emotion_expression_questionnaires,
        ]
        
        passed = 0
        failed = 0
        
        for test in tests:
            if test():
                passed += 1
            else:
                failed += 1
        
        # 总结
        print("\n" + "="*80)
        print("测试总结")
        print("="*80)
        print(f"总测试数: {len(tests)}")
        print(f"✓ 通过: {passed}")
        print(f"✗ 失败: {failed}")
        
        if self.all_passed:
            print("\n🎉 所有测试通过！数据集可以正常使用。")
            print("="*80)
            return 0
        else:
            print("\n⚠️  部分测试失败，请检查数据集。")
            print("="*80)
            return 1


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="测试 LongEmotion 数据集加载")
    parser.add_argument(
        '--data_dir',
        type=str,
        default='hf_dataset',
        help='数据集目录 (默认: hf_dataset)'
    )
    
    args = parser.parse_args()
    
    tester = DatasetTester(args.data_dir)
    exit_code = tester.run_all_tests()
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
