
import os 
import sys

import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd
import pickle
from utils.vanvalenlab_convolutional_recurrent import ConvGRU2D
import argparse


def net_weights_reinitializer(model):
    for ix, layer in enumerate(model.layers):
        if hasattr(model.layers[ix], 'kernel_initializer') and \
                hasattr(model.layers[ix], 'bias_initializer'):
            weight_initializer = model.layers[ix].kernel_initializer
            bias_initializer = model.layers[ix].bias_initializer

            old_weights, old_biases = model.layers[ix].get_weights()
            ev=keras.backend.eval #evaluate everything to comply with keras1
            model.layers[ix].set_weights([
                ev(weight_initializer(ev(old_weights.shape))),
                ev(bias_initializer(ev(old_biases.shape)))])



def traject_learning_dataframe_update(train_accur,
                                    test_accur, 
                                 decoder_history,
                                 #full_history,
                                 net, parameters, name = '_'):
    '''
    TODO combine all write to dataframe functions! 
    
    '''
    file_name = 'summary_dataframe_{}.pkl'.format(name)
    path = os.getcwd() 
    if os.path.isfile(os.getcwd()  + '{}'.format(file_name)):
        dataframe = pd.read_pickle(path +' {}'.format(file_name))
    else:
        dataframe = pd.DataFrame()
    values_to_add = parameters
    values_to_add['net_name'] = net.name
    values_to_add['student_min_test_error'] = min(test_accur)
    values_to_add['student_min_train_error'] = min(train_accur)
    values_to_add['student_train_error'] = train_accur
    values_to_add['student_test_error'] = test_accur
    values_to_add['decoder_max_test_error'] = max(decoder_history.history['val_sparse_categorical_accuracy'])
    values_to_add['decoder_max_train_error'] = max(decoder_history.history['sparse_categorical_accuracy'])
    values_to_add['decoder_train_error'] = [decoder_history.history['sparse_categorical_accuracy']]
    values_to_add['decoder_test_error'] = [decoder_history.history['val_sparse_categorical_accuracy']]
    # values_to_add['full_max_test_error'] = max(full_history.history['val_sparse_categorical_accuracy'])
    # values_to_add['full_max_train_error'] = max(full_history.history['sparse_categorical_accuracy'])
    # values_to_add['full_train_error'] = [full_history.history['sparse_categorical_accuracy']]
    # values_to_add['full_test_error'] = [full_history.history['val_sparse_categorical_accuracy']]
    dataframe = dataframe.append(values_to_add, ignore_index = True)

    dataframe.to_pickle(path + '{}'.format(file_name))

def save_model(net,path,parameters,checkpoint = True):
    feature = parameters['feature']
    traject = parameters['trajectory_index']
    home_folder = path + '{}_{}_saved_models/'.format(feature, traject)
    os.mkdir(home_folder)
    if checkpoint:
        child_folder = home_folder + 'checkpoint/'
    else:
        child_folder = home_folder + 'end_of_run_model/'
    os.mkdir(child_folder)
    
    #Saving using net.save method
    model_save_path = child_folder + '{}_keras_save'.format(feature)
    os.mkdir(model_save_path)
    net.save(model_save_path)
    #LOADING WITH - keras.models.load_model(path)
    
    #Saving weights as numpy array
    numpy_weights_path = child_folder + '{}_numpy_weights/'.format(feature)
    os.mkdir(numpy_weights_path)
    all_weights = net.get_weights()
    with open(numpy_weights_path + 'numpy_weights_{}_{}'.format(feature,traject), 'wb') as file_pi:
        pickle.dump(all_weights, file_pi)
    #LOAD WITH - pickle.load - and load manualy to model.get_layer.set_weights()
    
    #save weights with keras
    keras_weights_path = child_folder + '{}_keras_weights/'.format(feature)
    os.mkdir(keras_weights_path)
    net.save_weights(keras_weights_path + 'keras_weights_{}_{}'.format(feature,traject))
    #LOADING WITH - load_status = sequential_model.load_weights("ckpt")
    



def student3(sample = 10, res = 8, activation = 'tanh', dropout = 0.0, rnn_dropout = 0.0, upsample = 0,
             num_feature = 1, layer_norm = False ,batch_norm = False, n_layers=3, conv_rnn_type='lstm',block_size = 1,
             add_coordinates = False, time_pool = False, coordinate_mode=1, attention_net_size=64, attention_net_depth=1,
             rnn_layer1=32,
             rnn_layer2=64,
             dense_interface=False,
            loss="mean_squared_error",
             **kwargs):
    #TO DO add option for different block sizes in every convcnn
    #TO DO add skip connections in the block
    #coordinate_mode 1 - boardcast,
    #coordinate_mode 2 - add via attention block
    if time_pool == '0':
        time_pool = 0
    inputA = keras.layers.Input(shape=(sample, res,res,3))
    if add_coordinates and coordinate_mode==1:
        inputB = keras.layers.Input(shape=(sample,res,res,2))
    else:
        inputB = keras.layers.Input(shape=(sample,2))
    if conv_rnn_type == 'lstm':
        Our_RNN_cell = keras.layers.ConvLSTM2D
    elif  conv_rnn_type == 'gru':
        Our_RNN_cell = ConvGRU2D
    else:
        error("not supported type of conv rnn cell")

    #Broadcast the coordinates to a [res,res,2] matrix and concat to x
    if add_coordinates:
        if coordinate_mode==1:
            x = keras.layers.Concatenate()([inputA,inputB])
        elif coordinate_mode==2:
            x = inputA
            a = keras.layers.GRU(attention_net_size,input_shape=(sample, None),return_sequences=True)(inputB)
            for ii in range(attention_net_depth-1):
                a = keras.layers.GRU(attention_net_size, input_shape=(sample, None), return_sequences=True)(a)
    else:
        x = inputA

    if upsample != 0:
        x = keras.layers.TimeDistributed(keras.layers.UpSampling2D(size=(upsample, upsample)))(x)

    print(x.shape)
    for ind in range(block_size):
        x = Our_RNN_cell(rnn_layer1,(3,3), padding = 'same', return_sequences=True,
                                dropout = dropout,recurrent_dropout=rnn_dropout,
                            name = 'convLSTM1{}'.format(ind))(x)
    for ind in range(block_size):
        x = Our_RNN_cell(rnn_layer2,(3,3), padding = 'same', return_sequences=True,
                            name = 'convLSTM2{}'.format(ind),
                            dropout = dropout,recurrent_dropout=rnn_dropout,)(x)
        if add_coordinates and coordinate_mode==2:
            a_ = keras.layers.TimeDistributed(keras.layers.Dense(64,activation="tanh"))(a)
            a_ = keras.layers.Reshape((sample, 1, 1, -1))(a_)
            x = x * a_
    for ind in range(block_size):
        if ind == block_size - 1:
            if time_pool:
                return_seq = True
            else:
                return_seq = False
        else:
            return_seq = True
        x = Our_RNN_cell(num_feature,(3,3), padding = 'same', return_sequences=return_seq,
                            name = 'convLSTM3{}'.format(ind), activation=activation,
                            dropout = dropout,recurrent_dropout=rnn_dropout,)(x)
        if dense_interface:
            if return_seq:
                x = keras.layers.TimeDistributed(keras.layers.Conv2D(64, (3, 3), padding='same',
                                        name='anti_sparse'))(x)
            else:
                x = keras.layers.Conv2D(64, (3, 3), padding='same',
                                        name='anti_sparse')(x)
    print(return_seq)
    if time_pool:
        print(time_pool)
        if time_pool == 'max_pool':
            x = tf.keras.layers.MaxPooling3D(pool_size=(sample, 1, 1))(x)
        elif time_pool == 'average_pool':
            x = tf.keras.layers.AveragePooling3D(pool_size=(sample, 1, 1))(x)
        x = tf.squeeze(x,1)
    if layer_norm:
        x = keras.layers.LayerNormalization(axis=3)(x)
    if batch_norm:
        x = keras.layers.BatchNormalization()(x)

    print(x.shape)
    model = keras.models.Model(inputs=[inputA,inputB],outputs=x, name = 'student_3')
    opt=tf.keras.optimizers.Adam(lr=1e-3)

    model.compile(
        optimizer=opt,
        loss=loss,
        metrics=["mean_squared_error", "mean_absolute_error", "cosine_similarity"],
    )
    return model

def default_parameters():
    parser = argparse.ArgumentParser(add_help=False)
    #general parameters
    parser.add_argument('--run_name_prefix', default='DRC_training', type=str, help='path to pretrained teacher net')
    parser.add_argument('--run_index', default=10, type=int, help='run_index')
    parser.add_argument('--verbose', default=2, type=int, help='run_index')
    
    parser.add_argument('--testmode', dest='testmode', action='store_true')
    parser.add_argument('--no-testmode', dest='testmode', action='store_false')
    
    ### student parameters
    parser.add_argument('--epochs', default=100, type=int, help='num training epochs')
    parser.add_argument('--int_epochs', default=1, type=int, help='num internal training epochs')
    parser.add_argument('--decoder_epochs', default=10, type=int, help='num of decoder retraining epochs')
    parser.add_argument('--num_feature', default=64, type=int, help='legacy to be discarded')
    parser.add_argument('--rnn_layer1', default=32, type=int, help='legacy to be discarded')
    parser.add_argument('--rnn_layer2', default=64, type=int, help='legacy to be discarded')
    parser.add_argument('--time_pool', default='average_pool', help='time dimention pooling to use - max_pool, average_pool, 0')
    
    parser.add_argument('--upsample', default=7, type=int, help='spatial upsampling of input 0 for no')
    
    
    parser.add_argument('--conv_rnn_type', default='gru', type=str, help='conv_rnn_type')
    parser.add_argument('--student_nl', default='relu', type=str, help='non linearity')
    parser.add_argument('--dropout', default=0.0, type=float, help='dropout1')
    parser.add_argument('--rnn_dropout', default=0.0, type=float, help='dropout1')
    parser.add_argument('--pretrained_student_path', default=None, type=str, help='pretrained student, works only with student3')
    
    parser.add_argument('--decoder_optimizer', default='SGD', type=str, help='Adam or SGD')
    
    parser.add_argument('--skip_student_training', dest='skip_student_training', action='store_true')
    parser.add_argument('--no-skip_student_training', dest='skip_student_training', action='store_false')
    
    parser.add_argument('--fine_tune_student', dest='fine_tune_student', action='store_true')
    parser.add_argument('--no-fine_tune_student', dest='fine_tune_student', action='store_false')
    
    parser.add_argument('--layer_norm_student', dest='layer_norm_student', action='store_true')
    parser.add_argument('--no-layer_norm_student', dest='layer_norm_student', action='store_false')
    
    parser.add_argument('--batch_norm_student', dest='batch_norm_student', action='store_true')
    parser.add_argument('--no-batch_norm_student', dest='batch_norm_student', action='store_false')
    
    parser.add_argument('--val_set_mult', default=5, type=int, help='repetitions of validation dataset to reduce trajectory noise')
    
    
    ### syclop parameters
    parser.add_argument('--trajectory_index', default=0, type=int, help='trajectory index - set to 0 because we use multiple trajectories')
    
    parser.add_argument('--res', default=8, type=int, help='resolution')
    parser.add_argument('--trajectories_num', default=-1, type=int, help='number of trajectories to use')
    parser.add_argument('--broadcast', default=1, type=int, help='1-integrate the coordinates by broadcasting them as extra dimentions, 2- add coordinates as an extra input')
    
    parser.add_argument('--loss', default='mean_squared_error', type=str, help='loss type for student')
    parser.add_argument('--noise', default=0.5, type=float, help='added noise to the const_p_noise style')
    parser.add_argument('--max_length', default=5, type=int, help='choose syclops max trajectory length')
    
    
    ### teacher network parameters
    
    parser.add_argument('--network_topology', default='resnet50_on_imagenet', type=str, help='default, v2 or resnet50_on_imagenet')
 
    parser.add_argument('--resblocks', default=3, type=int, help='resblocks')
    parser.add_argument('--student_version', default=3, type=int, help='student version')
    
    parser.add_argument('--last_layer_size', default=128, type=int, help='last_layer_size')
    
    
    parser.add_argument('--dropout1', default=0.2, type=float, help='dropout1')
    parser.add_argument('--dropout2', default=0.0, type=float, help='dropout2')
    parser.add_argument('--dataset_norm', default=128.0, type=float, help='dropout2')
    parser.add_argument('--dataset_center', dest='dataset_center', action='store_true')
    parser.add_argument('--no-dataset_center', dest='dataset_center', action='store_false')
    
    parser.add_argument('--dense_interface', dest='dense_interface', action='store_true')
    parser.add_argument('--no-dense_interface', dest='dense_interface', action='store_false')
    
    parser.add_argument('--layer_norm_res', dest='layer_norm_res', action='store_true')
    parser.add_argument('--no-layer_norm_res', dest='layer_norm_res', action='store_false')
    
    parser.add_argument('--layer_norm_2', dest='layer_norm_2', action='store_true')
    parser.add_argument('--no-layer_norm_2', dest='layer_norm_2', action='store_false')
    
    parser.add_argument('--skip_conn', dest='skip_conn', action='store_true')
    parser.add_argument('--no-skip_conn', dest='skip_conn', action='store_false')
    
    parser.add_argument('--last_maxpool_en', dest='last_maxpool_en', action='store_true')
    parser.add_argument('--no-last_maxpool_en', dest='last_maxpool_en', action='store_false')
    
    
    parser.add_argument('--resnet_mode', dest='resnet_mode', action='store_true')
    parser.add_argument('--no-resnet_mode', dest='resnet_mode', action='store_false')
    
    parser.add_argument('--nl', default='relu', type=str, help='non linearity')
    
    parser.add_argument('--stopping_patience', default=10, type=int, help='stopping patience')
    parser.add_argument('--learning_patience', default=5, type=int, help='stopping patience')
    parser.add_argument('--manual_suffix', default='', type=str, help='manual suffix')
    
    parser.add_argument('--data_augmentation', dest='data_augmentation', action='store_true')
    parser.add_argument('--no-data_augmentation', dest='data_augmentation', action='store_false')
    
    parser.add_argument('--rotation_range', default=0.0, type=float, help='dropout1')
    parser.add_argument('--width_shift_range', default=0.1, type=float, help='dropout2')
    parser.add_argument('--height_shift_range', default=0.1, type=float, help='dropout2')
    
    ##advanced trajectory parameters
    parser.add_argument('--time_sec', default=0.3, type=float, help='time for realistic trajectory')
    parser.add_argument('--traj_out_scale', default=4.0, type=float, help='scaling to match receptor size')
    
    parser.add_argument('--snellen', dest='snellen', action='store_true')
    parser.add_argument('--no-snellen', dest='snellen', action='store_false')
    
    parser.add_argument('--vm_kappa', default=0., type=float, help='factor for emulating sub and super diffusion')
    
    parser.set_defaults(data_augmentation=True,
                    layer_norm_res=True,
                    layer_norm_student=True,
                    batch_norm_student=False,
                    layer_norm_2=True,
                    skip_conn=True,
                    last_maxpool_en=True,
                    testmode=False,
                    dataset_center=True,
                    dense_interface=False,
                    resnet_mode=True,
                    skip_student_training=False,
                    fine_tune_student=False,
                    snellen=True)
    return parser

