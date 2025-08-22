#!/usr/bin/env bash
WORK_DIR=$(dirname $(readlink -f $0))
echo "${WORK_DIR}"
UUID=$(uuidgen)
echo "${UUID}"
PID=$BASHPID
echo "$PID"

METHOD=$1
TASK=$2

OUTPUT_DIR="${WORK_DIR}"/output/${UUID}
mkdir -p "${OUTPUT_DIR}"
log_file="${OUTPUT_DIR}"/logs.txt
exec &> >(tee -a "$log_file")


PYTHONPATH="${WORK_DIR}"/src python "${WORK_DIR}"/evaluate.py \
  --task "${TASK}" --method "${METHOD}" \
  --data_dir "${WORK_DIR}"/data --prompts_dir "${WORK_DIR}"/prompts --base_dir "${WORK_DIR}" \
  --model_name Qwen/Qwen3-8B \
  --model_api_key local \
  --model_url local \
  --evaluator_name gpt-4o \
  --evaluator_api_key your_key \
  --evaluator_url your_url \
  --CoEM_Sage_name gpt-4o \
  --CoEM_Sage_api_key your_key \
  --CoEM_Sage_url your_url \
