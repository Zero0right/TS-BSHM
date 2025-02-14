for model_name in REV_TCN_CFFN_KAN SegRNN iTransformer PatchTST PatchMixer TSMixer TIDE FreTS ModernTCN
do
    root_path_name=./dataset/
    data_path_name=ETTm1.csv
    model_id_name=ETTm1
    data_name=ETTm1

    seq_len=720
    for pred_len in 96 192 336 720
    do
        python -u run.py \
        --is_training 1 \
        --root_path $root_path_name \
        --data_path $data_path_name \
        --model_id $model_id_name'_'$seq_len'_'$pred_len \
        --model $model_name \
        --data $data_name \
        --features M \
        --seq_len $seq_len \
        --pred_len $pred_len \
        --patch_len 16 \
        --stride 16 \
        --enc_in 7 \
        --d_model 512 \
        --dropout 0.5 \
        --train_epochs 30 \
        --patience 10 \
        --dec_in 7 \
        --c_out 7 \
        --itr 1 --batch_size 256 --learning_rate 0.002
    done
done

# REV_TCN_KAN RPTM2 RPTK RPTK2 SegRNN RPTM iTransformer Informer Reformer PatchTST PatchMixer TSMixer TIDE GRU