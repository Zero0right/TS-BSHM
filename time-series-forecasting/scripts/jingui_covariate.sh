#  FreTS SegRNN iTransformer Informer Reformer PatchMixer TSMixer TIDE GRU
# RPTK RPTM iTransformer Informer Reformer TIDE PatchMixer TSMixer SegRNN GRU PatchTST ModernTCN FreTS NLinear
for model_name in RPTM iTransformer
do
    root_path_name=./dataset/
    data_path_name=jingui_10min_interval_covariate.csv
    model_id_name=Jingui
    data_name=custom

    seq_len=96
    for pred_len in 96 192 336 720
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
        --enc_in 8 \
        --d_model 512 \
        --dropout 0.5 \
        --train_epochs 50 \
        --patience 10 \
        --dec_in 8 \
        --c_out 8 \
        --itr 1 --batch_size 32 --learning_rate 0.001
    done
done
