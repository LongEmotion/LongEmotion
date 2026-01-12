#!/usr/bin/env bash
WORK_DIR=$(dirname $(readlink -f $0))
echo "${WORK_DIR}"
UUID=$(uuidgen)
echo "${UUID}"
PID=$BASHPID
echo "$PID"

METHOD=$1
TASK=$2

OUTPUT_DIR="${WORK_DIR}"
mkdir -p "${OUTPUT_DIR}"
log_file="${OUTPUT_DIR}"/logs.txt
exec &> >(tee -a "$log_file")

PYTHONPATH="${WORK_DIR}"/src python "${WORK_DIR}"/evaluate.py \
  --task "${TASK}" --method "${METHOD}" \
  --data_dir "${WORK_DIR}"/data --prompts_dir "${WORK_DIR}"/prompts \
  --base_dir "${OUTPUT_DIR}" \
  --model_name "deepseek-chat" \
  --model_api_key "your_api_key" \
  --model_url "your_url" \
  --model_name_coem_sage deepseek-chat \
  --model_api_key_coem_sage your_api_key \
  --model_url_coem_sage your_url \
  --evaluator_name gpt-4o \
  --evaluator_api_key "your_api_key" \
  --evaluator_url "your_url"