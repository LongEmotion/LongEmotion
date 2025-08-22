import argparse
import json


from src.evaluations.base import BaseEvaluator
from src.evaluations.coem import COEMEvaluator
from src.evaluations.rag import RAGEvaluator

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LongEmotion evaluator")
    parser.add_argument("-d", "--task", type=str,
                        help="task name")
    parser.add_argument("-dd", "--data_dir", default="data", type=str,
                        help="data directory")
    parser.add_argument("-pd", "--prompts_dir", default="prompts", type=str,
                        help="prompts directory")
    parser.add_argument("-bd", "--base_dir", default="LongEmotion", type=str,
                        help="LongEmotion directory")
    methods = ['baseline', 'rag', 'coem']
    parser.add_argument("-m", "--method", default="baseline", type=str,
                        help="Methods used for generation.")

    parser.add_argument('--model_name', type=str, help='bar help')
    parser.add_argument('--model_api_key', type=str, help='bar help')
    parser.add_argument('--model_url', type=str, help='bar help')

    parser.add_argument('--evaluator_name', type=str, help='bar help')
    parser.add_argument('--evaluator_api_key', type=str, help='bar help')
    parser.add_argument('--evaluator_url', type=str, help='bar help')

    parser.add_argument('--CoEM_Sage_name', type=str, help='bar help')
    parser.add_argument('--CoEM_Sage_api_key', type=str, help='bar help')
    parser.add_argument('--CoEM_Sage_url', type=str, help='bar help')

    args = parser.parse_args()

    print(args)
    test_data = []
    if args.task == 'Questionnaire':
        test_data=[]
    else:
        jsonl_file_path = f"{args.data_dir}/{args.task}.jsonl"
        with open(jsonl_file_path, 'r', encoding='utf-8') as fd:
            for line in fd:
                test_data.append(json.loads(line.strip())) 

    if args.method == 'baseline':
        evaluator = BaseEvaluator(
            model_name=args.model_name,
            model_api_key=args.model_api_key,
            model_url=args.model_url,
            evaluator_name=args.evaluator_name,
            evaluator_api_key=args.evaluator_api_key,
            evaluator_url=args.evaluator_url,
            CoEM_sage_model_name=args.CoEM_Sage_name,
            CoEM_api_key=args.CoEM_Sage_api_key,
            CoEM_url=args.CoEM_Sage_url,
            data_dir=args.data_dir,
            prompts_dir=args.prompts_dir,
            base_dir = args.base_dir
        )
    elif args.method == 'rag':
        evaluator = RAGEvaluator(
            model_name=args.model_name,
            model_api_key=args.model_api_key,
            model_url=args.model_url,
            evaluator_name=args.evaluator_name,
            evaluator_api_key=args.evaluator_api_key,
            evaluator_url=args.evaluator_url,
            CoEM_sage_model_name=args.CoEM_Sage_name,
            CoEM_api_key=args.CoEM_Sage_api_key,
            CoEM_url=args.CoEM_Sage_url,
            data_dir=args.data_dir,
            prompts_dir=args.prompts_dir,
            base_dir = args.base_dir
        )
    elif args.method == 'coem':
        evaluator = COEMEvaluator(
            model_name=args.model_name,
            model_api_key=args.model_api_key,
            model_url=args.model_url,
            evaluator_name=args.evaluator_name,
            evaluator_api_key=args.evaluator_api_key,
            evaluator_url=args.evaluator_url,
            CoEM_sage_model_name=args.CoEM_Sage_name,
            CoEM_api_key=args.CoEM_Sage_api_key,
            CoEM_url=args.CoEM_Sage_url,
            data_dir=args.data_dir,
            prompts_dir=args.prompts_dir,
            base_dir = args.base_dir
        )
    else:
        raise NotImplementedError

    if args.task=='Emotion_Classification':
        evaluator.run_emotionclass(test_data)
    elif args.task=='Emotion_Detection':
        evaluator.run_emotiondetection(test_data)
    elif args.task=='QA':
        evaluator.run_fileQA(test_data)
    elif args.task=='Reports_Summary':
        evaluator.run_report_summary(test_data)
    elif args.task=='Conversations':
        evaluator.run_multicov(test_data)
    elif args.task=='Questionnaire':
        evaluator.questionnaire(test_data)