# DynamicalRecurrentClassifier
We show how recurrent connectivity in early vision together with eye motion helps to cope with limited sensor's spatial resolution.

Dynamical phenomena, such as recurrent neuronal activity  and perpetual motion of the eye, are typically overlooked in models of bottom-up visual perception. Recent experiments suggest that tiny inter-saccadic eye motion ("fixational drift") enhances visual  acuity beyond the limit imposed by the density of retinal photoreceptors. Here we hypothesize that such an enhancement is enabled by recurrent neuronal computations in early visual areas. Specifically, we explore a setting involving a low-resolution dynamical sensor that moves with respect to a static scene, with drift-like tiny steps. This setting mimics a dynamical eye viewing objects in perceptually-challenging conditions. The dynamical sensory input is classified by a convolutional neural network with recurrent connectivity added to its lower layers, in analogy to recurrent connectivity in early visual areas.  Applying our system to CIFAR-10 and CIFAR-100 datasets down-sampled via 8x8 sensor, we found that (i) classification accuracy, which is drastically reduced by this down-sampling, is mostly restored to its 32x32 baseline level when using a moving sensor and recurrent connectivity, (ii) in this setting, neurons in the early layers exhibit a wide repertoire of selectivity patterns, spanning the spatiotemporal selectivity space, with neurons preferring different combinations of spatial and temporal patterning, and (iii) curved sensor's trajectories improve  visual acuity compared to straight trajectories, echoing recent experimental findings involving eye-tracking in challenging conditions. Our work sheds light on the possible role of recurrent connectivity in early vision as well as the roles of fixational drift and temporal-frequency selective cells in the visual system. It also proposes a solution for artificial image recognition in settings with limited resolution and multiple time samples, such as in edge AI applications.


!(https://github.com/orram/DynamicalRecurrentClassifier/iclr2022_fig1_small_ver.png?raw=true)

### Steps to run:
#### Teacher Training:
Choose one of the following 

###### teacher of version 1:
python cifar_style.py --stopping_patience 30 --learning_patience 10 --resblocks 5 --no-last_maxpool_en --nl elu --width_shift_range 0.15 --height_shift_range 0.15 --rotation_range 5

###### teacher of versions 2:
python cifar_style.py --stopping_patience 30 --learning_patience 10 --resblocks8 3 --resblocks16 3 --no-last_maxpool_en --nl elu --width_shift_range 0.15 --height_shift_range 0.15 --rotation_range 5 --network_topology v2

###### teacher is ResNet50:
python cifar_style.py --stopping_patience 30 --learning_patience 10 --width_shift_range 0.15 --height_shift_range 0.15 --rotation_range 5 --network_topology resnet50_on_imagenet

#### DRC Training:
Choose one of the following 

###### For teacher version 1 or 2:
python DRC_training.py --student_nl elu --teacher_net TEACHER_NET_PATH --dropout 0.0 --rnn_dropout 0.0 --conv_rnn_type gru --n_samples 10 --epochs 100 --int_epochs 1 --trajectories_num -1 --student_block_size 2 --broadcast 0 --noise 0.5 --max_length 10 --time_pool average_pool --rnn_layer2 128 --rnn_layer1 64 --student_version 3 --style spiral_2dir2

###### For ResNet50 teacher:
python  DRC_training.py --student_nl relu --teacher_net TEACHER_NET_PATH --dropout 0.0 --rnn_dropout 0.0 --conv_rnn_type gru --n_samples 5 --epochs 100 --int_epochs 1 --trajectories_num -1 --student_block_size 2 --broadcast 1 --style spiral_2dir2  --noise 0.5 --max_length 5 --time_pool average_pool --student_version 3 --upsample 7 --resnet_mode  --decoder_optimizer SGD --val_set_mult 1 --n_classes 10
