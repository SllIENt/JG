export CUDA_VISIBLE_DEVICES=0

# 训练频域增强模型（优化参数）
python main.py --anormly_ratio 0.5 --num_epochs 15 --batch_size 8 --mode train \
  --dataset SMD --data_path dataset/SMD --input_c 38 \
  --use_freq True --freq_loss_weight 0.2 --win_size 150 --lr 0.00005

# 测试
python main.py --anormly_ratio 0.5 --num_epochs 15 --batch_size 8 --mode test \
  --dataset SMD --data_path dataset/SMD --input_c 38 \
  --use_freq True --win_size 150
