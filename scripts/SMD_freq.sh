export CUDA_VISIBLE_DEVICES=0

# 训练频域增强模型
python main.py --anormly_ratio 0.5 --num_epochs 10 --batch_size 256 --mode train --dataset SMD --data_path dataset/SMD --input_c 38 --use_freq True --freq_loss_weight 0.1

# 测试
python main.py --anormly_ratio 0.5 --num_epochs 10 --batch_size 256 --mode test --dataset SMD --data_path dataset/SMD --input_c 38 --pretrained_model 20 --use_freq True
