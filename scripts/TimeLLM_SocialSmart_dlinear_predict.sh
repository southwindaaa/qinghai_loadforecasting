model_name=DLinear
train_epochs=51
learning_rate=0.01

master_port=00097
num_process=1
batch_size=32
d_model=32
d_ff=32

comment='TimeLLM-SocalSmart'

python run_main.py \
  --task_name long_term_forecast \
  --is_training 0 \
  --root_path ./data/ \
  --data_path  load_data.csv \
  --model_id Qinghai \
  --model $model_name \
  --data qinghaidata \
  --seq_len 336 \
  --label_len 336 \
  --pred_len 76 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --model_comment $comment \
  --train_date 20240120 \
  --checkpoint 10 \
  --predict_batch_size 1
