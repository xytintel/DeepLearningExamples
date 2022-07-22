rm -rf results/checkpoints

# fp16
python3 -m torch.distributed.launch /workspace/bert/run_pretraining.py --input_dir=/workspace/bert/data/pretrain/phase2/bin_size_64/parquet/ --output_dir=/workspace/bert/results/checkpoints \
--config_file=bert_configs/large.json --vocab_file=vocab/vocab --train_batch_size=16 \
--max_seq_length=512 --max_predictions_per_seq=80 --max_steps=1563 --warmup_proportion=0.128 --num_steps_per_checkpoint=200 \
--learning_rate=4e-3 --seed=12439 \
--fp16 --gradient_accumulation_steps=1 --allreduce_post_accumulation --allreduce_post_accumulation_fp16 --do_train --phase2 --phase1_end_step=7038 --json-summary /workspace/bert/results/dllogger.json --disable_progress_bar --num_workers=1 \
--no_dense_sequence_output --disable_jit_fusions

# fp16 profile trace
python3 -m torch.distributed.launch /workspace/bert/run_pretraining.py --input_dir=/workspace/bert/data/pretrain/phase2/bin_size_64/parquet/ --output_dir=/workspace/bert/results/checkpoints \
--config_file=bert_configs/large.json --vocab_file=vocab/vocab --train_batch_size=16 \
--max_seq_length=512 --max_predictions_per_seq=80 --max_steps=1563 --warmup_proportion=0.128 --num_steps_per_checkpoint=200 \
--learning_rate=4e-3 --seed=12439 \
--fp16 --gradient_accumulation_steps=1 --allreduce_post_accumulation --allreduce_post_accumulation_fp16 --do_train --phase2 --phase1_end_step=7038 --json-summary /workspace/bert/results/dllogger.json --disable_progress_bar --num_workers=1 \
--no_dense_sequence_output --disable_jit_fusions --profile --profile_trace

# fp32/tf32
# python3 -m torch.distributed.launch /workspace/bert/run_pretraining.py --input_dir=/workspace/bert/data/pretrain/phase2/bin_size_64/parquet/ --output_dir=/workspace/bert/results/checkpoints \
# --config_file=bert_configs/large.json --vocab_file=vocab/vocab --train_batch_size=8 \
# --max_seq_length=512 --max_predictions_per_seq=80 --max_steps=1563 --warmup_proportion=0.128 --num_steps_per_checkpoint=200 \
# --learning_rate=4e-3 --seed=12439 \
# --gradient_accumulation_steps=1 --allreduce_post_accumulation --do_train --phase2 --phase1_end_step=7038 --json-summary /workspace/bert/results/dllogger.json --disable_progress_bar --num_workers=1 \
# --no_dense_sequence_output --disable_jit_fusions
