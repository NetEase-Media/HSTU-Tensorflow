export CUDA_VISIBLE_DEVICES='0'

set -x

python main.py \
--dataset=hstu-ml-1m \
--remap_hstu_ml_1m=True \
--load_timestamp=True \
--train_dir=tmp/${SUBSTRING} \
--num_epochs=100 \
--eval_every=1 \
--eval_item_not_in_history=True \
--embedding_initializer='truncated_normal' \
--maxlen=200 \
--dropout_rate=0.2 \
--num_blocks=8 \
--num_heads=2 \
--pre_norm=False \
--ffn=False \
--normalize_query=True \
--overwrite_key_with_query=True \
--qkv_projection_initializer=normal \
--qkv_projection_bias=False \
--qkv_projection_activation=silu \
--attention_type=time_interval_bias \
--compute_hstu_time_interval=True \
--hstu_time_interval_divisor=0.301 \
--time_interval_attention_max_interval=128 \
--relative_position_bias_add_item_interaction=True \
--scale_attention=False \
--attention_activation=silu \
--attention_normalization=real_length \
--attention_dropout=False \
--u_projection=True \
--u_projection_initializer=normal \
--u_projection_bias=False \
--linear_projection_and_dropout=True \
--dropout_before_linear_projection=True \
--normalize_prediction_embedding=True \
--normalize_test_embedding=True \
--scale_logits=20. \
--loss_type=sparse_ce \
