# DynamicalRecurrentClassifier
We show how recurrent connectivity in early vision together with eye motion helps to cope with limited sensor's spatial resolution.
### Steps to run:
#### Teacher Training:
Choose one of the following 

teacher of version 1:
python cifar_style.py --stopping_patience 30 --learning_patience 10 --resblocks 5 --no-last_maxpool_en --nl elu --width_shift_range 0.15 --height_shift_range 0.15 --rotation_range 5

teacher of versions 2:
python cifar_style.py --stopping_patience 30 --learning_patience 10 --resblocks8 3 --resblocks16 3 --no-last_maxpool_en --nl elu --width_shift_range 0.15 --height_shift_range 0.15 --rotation_range 5 --network_topology v2

#### DRC Training:
python full_learning_extended_multi_traject109g.py --student_nl relu --teacher_net TEACHER_NET_PATH --dropout 0.0 --rnn_dropout 0.0 --conv_rnn_type gru --n_samples 5 --epochs 100 --int_epochs 1 --trajectories_num -1 --student_block_size 2 --broadcast 1 --style spiral_2dir2  --noise 0.5 --max_length 5 --time_pool average_pool --student_version 3 --upsample 7 --resnet_mode  --decoder_optimizer SGD --val_set_mult 1 --n_classes 10
