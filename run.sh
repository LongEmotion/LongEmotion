#!/usr/bin/env bash
# Method list:
# 1. baseline
# 2. rag
# 3. coem
# 4. self-rag
# 5. search-o1

# Task list: 
# 1. Emotion_Summary 
# 2. Emotion_Classification_Emobench 
# 3. Emotion_Classification_Finentity 
# 4. Emotion_Detection 
# 5. Emotion_QA 
# 6. Emotion_Expression
# 7. Emotion_Conversations 

export CUDA_VISIBLE_DEVICES=0
bash evaluate.sh search-o1 Emotion_Conversations

