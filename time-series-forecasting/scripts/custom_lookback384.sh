
for model_name in LSTM GRU BlockRNN TCN Reformer
do
    root_path_name=./dataset/
    data_path_name=resampled_sensor_data_5.csv
    model_id_name=custom
    data_name=custom

    seq_len=384
    for pred_len in 48 96 144 192 288 576 720
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
        --patch_len 48 \
        --stride 48 \
        --enc_in 78 \
        --d_model 512 \
        --dropout 0.5 \
        --train_epochs 30 \
        --patience 3 \
        --itr 1 --batch_size 32 --learning_rate 0.001
       
    done
done


