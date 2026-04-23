CUDA_VISIBLE_DEVICES=0 python continual_train.py \
  --logs-dir logs-res-setting2/ \
  --setting 2 \
  -b 32 \
  -j 4 \
  --grad-accum-steps 2 \
  --amp
