# DynamicalRecurrentClassifier
We show how recurrent connectivity in early vision together with eye motion helps to cope with limited sensor's spatial resolution.

Dynamical phenomena, such as recurrent neuronal activity  and perpetual motion of the eye, are typically overlooked in models of bottom-up visual perception. Recent experiments suggest that tiny inter-saccadic eye motion ("fixational drift") enhances visual  acuity beyond the limit imposed by the density of retinal photoreceptors. Here we hypothesize that such an enhancement is enabled by recurrent neuronal computations in early visual areas. Specifically, we explore a setting involving a low-resolution dynamical sensor that moves with respect to a static scene, with drift-like tiny steps. This setting mimics a dynamical eye viewing objects in perceptually-challenging conditions. The dynamical sensory input is classified by a convolutional neural network with recurrent connectivity added to its lower layers, in analogy to recurrent connectivity in early visual areas.  Applying our system to CIFAR-10 and CIFAR-100 datasets down-sampled via 8x8 sensor, we found that (i) classification accuracy, which is drastically reduced by this down-sampling, is mostly restored to its 32x32 baseline level when using a moving sensor and recurrent connectivity, (ii) in this setting, neurons in the early layers exhibit a wide repertoire of selectivity patterns, spanning the spatiotemporal selectivity space, with neurons preferring different combinations of spatial and temporal patterning, and (iii) curved sensor's trajectories improve  visual acuity compared to straight trajectories, echoing recent experimental findings involving eye-tracking in challenging conditions. Our work sheds light on the possible role of recurrent connectivity in early vision as well as the roles of fixational drift and temporal-frequency selective cells in the visual system. It also proposes a solution for artificial image recognition in settings with limited resolution and multiple time samples, such as in edge AI applications.


![text](https://github.com/orram/DynamicalRecurrentClassifier/blob/main/iclr2022_fig1_small_ver.png)

### run

python DRC_training.py

Using ResNet50 as teacher.

#### Using smaller teacher as reference:

Change network network_topology to v2 or default 

python DRC_training.py --network_topology v2

And change other configs to:

python DRC_training.py --network_topology v2 --broadcast 0 --rnn_layer2 128 --rnn_layer1 64 --no-resnet_mode --upsample 8 --decoder_optimizer Adam

