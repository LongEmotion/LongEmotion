
## EC

```
uv run lm_eval --model vllm     --model_args pretrained=meta-llama/Llama-3.1-8B-Instruct,tensor_parallel_size=8     --tasks longemotion_emotion_classification     --include_path ./tasks     --output_path results/emobench/ --log_samples --apply_chat_template --system_instruction "Please identify the emotion of the given subject in the scenario."
```

## ED

```
uv run lm_eval --model vllm     --model_args pretrained=meta-llama/Llama-3.1-8B-Instruct,tensor_parallel_size=8     --tasks longemotion_emotion_detection     --include_path ./tasks     --output_path results/emobench/ --log_samples --apply_chat_template --system_instruction "You are an emotion detection model. Your task is to identify the unique emotion in a list of given texts. Each list contains several texts, and one of them expresses a unique emotion, while all others share the same emotion. You need to determine the index of the text that expresses the unique emotion."
```

# QA

```
uv run lm_eval --model vllm     --model_args pretrained=meta-llama/Llama-3.1-8B-Instruct,tensor_parallel_size=8     --tasks longemotion_fileqa     --include_path ./tasks     --output_path results/emobench/ --log_samples --apply_chat_template --system_instruction "You are a helpful assistant that answers questions based on the given context. Please provide accurate and concise answers."
```