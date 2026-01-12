import argparse
import json


from evaluations.base import BaseEvaluator
from evaluations.rag import RAGEvaluator
from evaluations.selfrag import SelfRAGEvaluator
from evaluations.searcho1 import SearchO1Evaluator
from evaluations.coem import COEMEvaluator
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="APTNESS evaluator")
    datasets = ['ed', 'extes']
    parser.add_argument("-d", "--task", default="ed", type=str,
                        help="one of: {}".format(", ".join(sorted(datasets))))
    parser.add_argument("-dd", "--data_dir", default="data", type=str,
                        help="data directory")
    parser.add_argument("-pd", "--prompts_dir", default="prompts", type=str,
                        help="prompts directory")
    strategies = ['esconv', 'extes']
    parser.add_argument("-s", "--strategy", default="extes", type=str,
                        help="one of: {}".format(", ".join(sorted(strategies))))
    methods = ['baseline', 'rag', 'self-rag', 'search-o1', 'coem']
    parser.add_argument("-m", "--method", default="baseline", type=str,
                        help="Methods used for generation. one of: {}".format(", ".join(sorted(methods))))

    parser.add_argument('--model_name', type=str, help='Model name for generation')
    parser.add_argument('--model_api_key', type=str, help='API key for generation model')
    parser.add_argument('--model_url', type=str, default=None, help='API URL for generation model')
    parser.add_argument('--model_name_coem', type=str, default=None, help='Model name for COEM method')
    parser.add_argument('--model_api_key_coem', type=str, default=None, help='API key for COEM model')
    parser.add_argument('--model_url_coem', type=str, default=None, help='API URL for COEM model')
    parser.add_argument('--model_name_coem_sage', type=str, default=None, help='CoEM sage model name')
    parser.add_argument('--model_api_key_coem_sage', type=str, default=None, help='CoEM sage API key')
    parser.add_argument('--model_url_coem_sage', type=str, default=None, help='CoEM sage API URL')
    parser.add_argument('--evaluator_name', type=str, help='Model name for evaluation')
    parser.add_argument('--evaluator_api_key', type=str, help='API key for evaluation model')
    parser.add_argument('--evaluator_url', type=str, default=None, help='API URL for evaluation model')
    parser.add_argument('--base_dir', type=str, default="output", help='Base directory for output files')

    args = parser.parse_args()

    print(args)
    test_data = []
    if args.task == 'Emotion_Expression':
        test_data=[]
    else:
        jsonl_file_path = f"{args.data_dir}/{args.task}.jsonl"
        try:
            with open(jsonl_file_path, 'r', encoding='utf-8') as fd:
                for line in fd:
                    test_data.append(json.loads(line.strip()))
        except FileNotFoundError:
            print(f"Error: Data file not found: {jsonl_file_path}")
            print("Please check if the data file exists.")
            exit(1)
        except json.JSONDecodeError as e:
            print(f"Error: Failed to parse JSON from {jsonl_file_path}: {e}")
            exit(1) 

    if args.method == 'baseline':
        evaluator = BaseEvaluator(
            model_name=args.model_name,
            model_api_key=args.model_api_key,
            model_url=args.model_url or "",
            evaluator_name=args.evaluator_name,
            evaluator_api_key=args.evaluator_api_key,
            evaluator_url=args.evaluator_url or "",
            CoEM_sage_model_name=args.model_name_coem_sage or "",
            CoEM_api_key=args.model_api_key_coem_sage or "",
            CoEM_url=args.model_url_coem_sage or "",
            data_dir=args.data_dir,
            prompts_dir=args.prompts_dir,
            base_dir=args.base_dir,
            task=args.task,
        )
    elif args.method == 'rag':
        evaluator = RAGEvaluator(
            model_name=args.model_name,
            model_api_key=args.model_api_key,
            model_url=args.model_url or "",
            evaluator_name=args.evaluator_name,
            evaluator_api_key=args.evaluator_api_key,
            evaluator_url=args.evaluator_url or "",
            CoEM_sage_model_name=args.model_name_coem_sage or "",
            CoEM_api_key=args.model_api_key_coem_sage or "",
            CoEM_url=args.model_url_coem_sage or "",
            data_dir=args.data_dir,
            prompts_dir=args.prompts_dir,
            base_dir=args.base_dir,
            task=args.task,
        )
    elif args.method == 'coem':
        evaluator = COEMEvaluator(
            model_name=args.model_name_coem or args.model_name,
            model_api_key=args.model_api_key_coem or args.model_api_key,
            model_url=args.model_url_coem or args.model_url or "",
            evaluator_name=args.evaluator_name,
            evaluator_api_key=args.evaluator_api_key,
            evaluator_url=args.evaluator_url or "",
            CoEM_sage_model_name=args.model_name_coem_sage or "",
            CoEM_api_key=args.model_api_key_coem_sage or "",
            CoEM_url=args.model_url_coem_sage or "",
            data_dir=args.data_dir,
            prompts_dir=args.prompts_dir,
            base_dir=args.base_dir,
            task=args.task,
        )
    elif args.method == 'self-rag':
        evaluator = SelfRAGEvaluator(
            model_name=args.model_name,
            model_api_key=args.model_api_key,
            model_url=args.model_url or "",
            evaluator_name=args.evaluator_name,
            evaluator_api_key=args.evaluator_api_key,
            evaluator_url=args.evaluator_url or "",
            CoEM_sage_model_name=args.model_name_coem_sage or "",
            CoEM_api_key=args.model_api_key_coem_sage or "",
            CoEM_url=args.model_url_coem_sage or "",
            data_dir=args.data_dir,
            prompts_dir=args.prompts_dir,
            base_dir=args.base_dir,
            task=args.task,
        )
    elif args.method == 'search-o1':
        evaluator = SearchO1Evaluator(
            model_name=args.model_name,
            model_api_key=args.model_api_key,
            model_url=args.model_url or "",
            evaluator_name=args.evaluator_name,
            evaluator_api_key=args.evaluator_api_key,
            evaluator_url=args.evaluator_url or "",
            CoEM_sage_model_name=args.model_name_coem_sage or "",
            CoEM_api_key=args.model_api_key_coem_sage or "",
            CoEM_url=args.model_url_coem_sage or "",
            data_dir=args.data_dir,
            prompts_dir=args.prompts_dir,
            base_dir=args.base_dir,
            task=args.task,
        )
    else:
        raise NotImplementedError

    if args.task=='Emotion_Classification_Emobench':
        evaluator.run_emotionclass(test_data)
    elif args.task=='Emotion_Classification_Finentity':
        evaluator.run_emotionclass(test_data)
    elif args.task=='Emotion_Detection':
        evaluator.run_emotiondetection(test_data)
    elif args.task=='Emotion_QA':
        evaluator.run_fileQA(test_data)
    elif args.task=='Emotion_Summary':
        evaluator.run_report_summary(test_data)
    elif args.task=='Emotion_Conversations':
        evaluator.run_multicov(test_data)
    elif args.task=='Emotion_Expression':
        evaluator.questionnaire(test_data)
    else:
        raise NotImplementedError
