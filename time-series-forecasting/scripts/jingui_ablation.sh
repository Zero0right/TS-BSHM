#  FreTS SegRNN iTransformer Informer Reformer PatchMixer TSMixer TIDE GRU
# RPTK RPTM iTransformer Informer Reformer TIDE PatchMixer TSMixer SegRNN GRU PatchTST ModernTCN FreTS NLinear DLinear BlockRNN
for model_name in MLP MLP_REV MLP_Patch MLP_TCN MLP_REV_Patch MLP_Patch_TCN MLP_REV_TCN RPTM
do
    root_path_name=./dataset/
    data_path_name=jingui_10min_interval.csv
    model_id_name=Jingui
    data_name=custom

    seq_len=96
    for pred_len in 96
    do
        python -u run.py \
        --is_training 1 \
        --root_path $root_path_name \
        --data_path $data_path_name \
        --model_id $model_id_name'_'$seq_len'_'$pred_len \
        --model $model_name \
        --data $data_name \
        --features J \
        --seq_len $seq_len \
        --pred_len $pred_len \
        --patch_len 16 \
        --stride 16 \
        --enc_in 78 \
        --d_model 512 \
        --dropout 0.5 \
        --train_epochs 50 \
        --patience 3 \
        --dec_in 78 \
        --c_out 78 \
        --itr 1 --batch_size 32 --learning_rate 0.001
    done
done
