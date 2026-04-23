CUDA_VISIBLE_DEVICES=0 python continual_train.py \
  --logs-dir logs-res-setting1/ \
  --setting 1 \
  -b 32 \
  -j 4 \
  --grad-accum-steps 2 \
  --amp
