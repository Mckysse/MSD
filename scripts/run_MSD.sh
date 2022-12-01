#!/bin/bash

GPU_IDS=0
# TGT_LANG=(es de nl)

tgt_lan=de
# tgt_lan=es
# tgt_lan=nl
SRC_LANG=en

SEEDS=(22 40 42 43 98 100)

for seed in ${SEEDS[@]}; do

    # STEP1: train teacher model (English: en)
    # ''random'' means to randomly shuffle the tgt_lan datasets and name the files from 1 to 100

    python3 main.py \
        --do_train \
        --evaluate_during_training \
        --gpu_ids ${GPU_IDS} \
        --seed ${seed} \
        --learning_rate 5e-5 \
        --data_dir ./data/ner/conll/${SRC_LANG} \
        --mmd_data_dir ./data/ner/random/${tgt_lan} \
        --output_dir conll-model-${seed}/mono-src-${SRC_LANG}

    # STEP2: teacher-student learning

    # for tgt_lan in ${TGT_LANG[@]}; do

    python3 main.py \
      --do_train \
      --evaluate_during_training \
      --do_KD \
      --gpu_ids ${GPU_IDS} \
      --seed ${seed} \
      --data_dir ./data/ner/conll/${tgt_lan} \
      --mmd_en_data_dir ./data/ner/random/${SRC_LANG} \
      --src_langs ${SRC_LANG} \
      --src_model_dir_prefix mono-src- \
      --src_model_dir conll-model-${seed} \
      --output_dir conll-model-${seed}/ts-${SRC_LANG}-${tgt_lan}

    python3 main.py \
        --do_predict \
        --gpu_ids ${GPU_IDS} \
        --seed ${seed} \
        --data_dir ./data/ner/conll/${tgt_lan} \
        --output_dir conll-model-${seed}/ts-${SRC_LANG}-${tgt_lan}
    # done


    # python3 main.py \
    #     --do_train \
    #     --do_KD \
    #     --gpu_ids ${GPU_IDS} \
    #     --seed ${seed} \
    #     --data_dir ./data/ner/conll/${TGT_LANG} \
    #     --src_langs ${SRC_LANG} \
    #     --src_model_dir_prefix mono-src- \
    #     --src_model_dir conll-model-${seed} \
    #     --output_dir conll-model-${seed}/ts-${SRC_LANG}-${TGT_LANG}

    # python3 main.py \
    #     --do_predict \
    #     --gpu_ids ${GPU_IDS} \
    #     --seed ${seed} \
    #     --data_dir ./data/ner/conll/${TGT_LANG} \
    #     --output_dir conll-model-${seed}/ts-${SRC_LANG}-${TGT_LANG}
done
