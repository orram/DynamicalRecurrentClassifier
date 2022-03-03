

import os 
import sys
import gc
sys.path.insert(1, os.getcwd()+'/..')
sys.path.insert(1, os.getcwd()[:-15])
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import cifar10, cifar100

import time
import argparse
from utils.feature_learning_utils import  student3, traject_learning_dataframe_update, default_parameters
from dataset.dataset_utils import DRC_dataset_generator, test_num_of_trajectories, dataset_preprocessing


from utils.vanvalenlab_convolutional_recurrent import ConvGRU2D

print(os.getcwd() + '/')
#%%
default_parser = default_parameters()
parser = argparse.ArgumentParser(parents=[default_parser])

parser.add_argument('--n_classes', default=10, type=int, help='number of classes') #According to CIFAR10 or 100
parser.add_argument('--n_samples', default=5, type=int, help='number of samples the DRC recieves as inputs')
parser.add_argument('--student_block_size', default=2, type=int, help='number of repetition of each convrnn block')
parser.add_argument('--style', default='spiral_2dir2', type=str, help='choose DRC style of motion')
parser.add_argument('--teacher_net', default='pretrained_teachers/resnet50_on_imagenet_cifar10_teacher.hdf', type=str, help='path to pretrained teacher net')


config = parser.parse_args()
config = vars(config)
config['position_dim'] = (config['n_samples'],config['res'],config['res'],2) if  config['broadcast']==1 else (config['n_samples'],2)
config['movie_dim'] = (config['n_samples'], config['res'], config['res'], 3)
print('config  ',config)

config = config
TESTMODE = config['testmode']


lsbjob = os.getenv('LSB_JOBID')
lsbjob = str(np.random.randint(100,999)) if lsbjob is None else lsbjob

# load dataset
if config['n_classes']==10:
    (trainX, trainY), (testX, testY)= cifar10.load_data()
elif config['n_classes']==100:
    (trainX, trainY), (testX, testY) = cifar100.load_data()
else:
    error

this_run_name = config['run_name_prefix'] + '_j' + lsbjob + '_t' + str(int(time.time()))
config['this_run_name'] = this_run_name
path = os.getcwd() + '/'
save_model_path = path + 'saved_models/{}'.format(this_run_name)
config['save_model_path'] = save_model_path
config['path'] = path
print(config)
# prepare data
trainX, testX = dataset_preprocessing(trainX, testX, 
                                      dataset_norm = config['dataset_norm'], 
                                      resnet_mode=config['resnet_mode'])

config_memory = config.copy()

#%%
#########################   Get Trained Teacher      #########################

def get_teacher():
    if os.path.exists(config['teacher_net']):
        teacher = keras.models.load_model(config['teacher_net'])
    else:
        from pretrained_teachers.cifar_style import train_teacher
        teacher = train_teacher(n_classes = config['n_classes'],network_topology=config['network_topology'])
        
        
    teacher.summary()
    teacher.evaluate(trainX[45000:], trainY[45000:], verbose=2)
    #Divide the teacher to front-end (fe_model) and back-end (be_model) parts:
        #Front-end - the early layers of the teacher to be replaced with the 
        #            recurrent DRC
        #Back-end  - the layers to be used as the DRC-Backend, they will be 
        #            retrained with the DRC-backend outputs.
    fe_model = teacher.layers[0]
    be_model = teacher.layers[1]
    
    return fe_model, be_model, config

def get_student():
    if config['student_version']==3:
        student_fun = student3
    else:
        error
    
    ########################   Initializing student (DRC-FE) #####################
    print('initializing student')
    
    student = student_fun(sample = config['max_length'],
                       res = config['res'],
                        activation = config['student_nl'],
                        dropout = config['dropout'],
                        rnn_dropout = config['rnn_dropout'],
                        num_feature = config['num_feature'],
                       rnn_layer1 = config['rnn_layer1'],
                       rnn_layer2 = config['rnn_layer2'],
                       layer_norm = config['layer_norm_student'],
                       batch_norm = config['batch_norm_student'],
                       conv_rnn_type = config['conv_rnn_type'],
                       block_size = config['student_block_size'],
                       add_coordinates = config['broadcast'],
                       time_pool = config['time_pool'],
                       dense_interface=config['dense_interface'],
                        loss=config['loss'],
                          upsample=config['upsample'])
    student.summary()
    
    return student


########################   Initializing data generator #######################
# generator parameters:
def initialize_data_generator(fe_model):
    BATCH_SIZE=32
    def args_to_dict(**kwargs):
        return kwargs
    generator_params = args_to_dict(batch_size=BATCH_SIZE, movie_dim=config['movie_dim'] , position_dim=config['position_dim'] , n_classes=None, shuffle=True,
                     prep_data_per_batch=True,one_hot_labels=False, one_random_sample=False,
                                        res = config['res'],
                                        n_samples = config['n_samples'],
                                        mixed_state = True,
                                        n_trajectories = config['trajectories_num'],
                                        trajectory_list = 0,
                                        broadcast=config['broadcast'],
                                        style = config['style'],
                                        max_length=config['max_length'],
                                        noise = config['noise'],
                                    time_sec=config['time_sec'], traj_out_scale=config['traj_out_scale'],  snellen=config['snellen'],vm_kappa=config['vm_kappa'])
    print('Preparing generators')
    # generator 1
    train_generator_features = DRC_dataset_generator(trainX[:-5000], None, teacher=fe_model, **generator_params)
    val_generator_features = DRC_dataset_generator(trainX[-5000:].repeat(config['val_set_mult'],axis=0), None, teacher=fe_model, validation_mode=True, **generator_params)
    # generator 2
    train_generator_classifier = DRC_dataset_generator(trainX[:-5000], trainY[:-5000], **generator_params)
    val_generator_classifier = DRC_dataset_generator(trainX[-5000:].repeat(config['val_set_mult'],axis=0), trainY[-5000:].repeat(config['val_set_mult'],axis=0), validation_mode=True, **generator_params)
    
    if  config['broadcast']==1:
        print('-------- total trajectories {}, out of tries: {}'.format( *test_num_of_trajectories(val_generator_classifier)))
    
    gc.collect()

    return train_generator_features, val_generator_features, train_generator_classifier, val_generator_classifier, generator_params

def train_student(student, val_generator_features, train_generator_features, ):
    #Define DRC-fe student training callbacks
    lr_reducer = keras.callbacks.ReduceLROnPlateau(factor=np.sqrt(0.1),
                                                   cooldown=0,
                                                   patience=5,
                                                   min_lr=0.5e-6)
    early_stopper = keras.callbacks.EarlyStopping(
                                                  min_delta=5e-5,
                                                  patience=10,
                                                  verbose=0,
                                                  mode='auto',
                                                  baseline=None,
                                                  restore_best_weights=True
                                                  )
    
    if True:
        student.evaluate(val_generator_features, verbose = 2)
        # print('{}/{}'.format(epoch+1,epochs))
    
        if config['pretrained_student_path'] is not None:
            student = tf.keras.models.load_model(config['pretrained_student_path'],custom_objects={'ConvGRU2D':ConvGRU2D})
    
        if not config['skip_student_training']:
            student_history = student.fit(train_generator_features,
                            epochs = config['epochs'],
                            validation_data=val_generator_features,
                            verbose = config['verbose'],
                            callbacks=[lr_reducer,early_stopper],
                            use_multiprocessing=False) #checkpoints won't really work
    
            train_accur = np.array(student_history.history['mean_squared_error']).flatten()
            test_accur = np.array(student_history.history['val_mean_squared_error']).flatten()
            student.save(save_model_path)
        student.evaluate(val_generator_features, verbose = 2)
        
        return student, train_accur, test_accur

###############      Evaluate and retrain the DRC-be         ##################
def combine_fe_be(student, be_model):
    #Define a Student_Decoder Network that will take the Teacher weights of the last layers:
    
    student.trainable = config['fine_tune_student']
    
    input0 = keras.layers.Input(shape=config['movie_dim'] )
    input1 = keras.layers.Input(shape=config['position_dim'] )
    x = student((input0,input1))
    x = be_model(x)
    fro_student_and_decoder = keras.models.Model(inputs=[input0,input1], outputs=x, name='frontend')
    if config['decoder_optimizer'] == 'Adam':
        opt=tf.keras.optimizers.Adam(lr=2.5e-4)
    elif config['decoder_optimizer'] == 'SGD':
        opt=tf.keras.optimizers.SGD(lr=2.5e-3)
    else:
        error
    
    fro_student_and_decoder.compile(
            optimizer=opt,
            loss="sparse_categorical_crossentropy",
            metrics=["sparse_categorical_accuracy"],
        )
    
    return fro_student_and_decoder


def fine_tune_DRC(fro_student_and_decoder,train_generator_classifier, val_generator_classifier):
    ################# Evaluate full DRC before be retraining #####################
    print('DRC accuracy before retraining the back-end')
    pre_training_accur = fro_student_and_decoder.evaluate(val_generator_classifier, verbose=2)
    print('')
    
    #Define callbacks for the DRC-be retraining
    lr_reducer = keras.callbacks.ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
    early_stopper = keras.callbacks.EarlyStopping(
        monitor='val_sparse_categorical_accuracy', min_delta=1e-4, patience=10, verbose=0,
        mode='auto', baseline=None, restore_best_weights=True
    )
    
    config['pre_training_decoder_accur'] = pre_training_accur[1]
    
    print('\nTraining the decoder')
    decoder_history = fro_student_and_decoder.fit(train_generator_classifier,
                           epochs = config['decoder_epochs'] if not TESTMODE else 1,
                           validation_data = val_generator_classifier,
                           verbose = 2,
                           callbacks=[lr_reducer,early_stopper],
                workers=8, use_multiprocessing=True)
    return decoder_history
    
def eval_and_save(be_model, student, fro_student_and_decoder,
                  DRC_dataset_generator, testX, testY, home_folder, generator_params, 
                  train_accur, test_accur, decoder_history):
    be_model.save(home_folder +'backend_trained_model') 
    if config['fine_tune_student']:
        student.save(home_folder +'student_fine_tuned_model')
    
    print('running 5 times on test data')
    test_generator_classifier = DRC_dataset_generator(testX, testY, **generator_params)
    for ii in range(5):
        fro_student_and_decoder.evaluate(test_generator_classifier, verbose=2)
    fro_student_and_decoder.save(home_folder + 'fro_student_and_decoder_trained')
    print('saved fro_student_and_decoder_trained!')
    traject_learning_dataframe_update(train_accur,test_accur, decoder_history, student,config, name = 'DRC_training_data')
    
def main():
    home_folder = save_model_path + '{}_saved_models/'.format(config['this_run_name'])
    fe_model, be_model = get_teacher()
    student = get_student()
    train_generator_features, val_generator_features, \
        train_generator_classifier, val_generator_classifier,\
            generator_params= initialize_data_generator(fe_model)
        
    student, train_accur, test_accur = \
        train_student(student, val_generator_features, train_generator_features, )
    fro_student_and_decoder = combine_fe_be(student, be_model)
    decoder_history = fine_tune_DRC(fro_student_and_decoder,train_generator_classifier, val_generator_classifier)
    eval_and_save(be_model, student, fro_student_and_decoder,
                  DRC_dataset_generator, testX, testY, home_folder, generator_params, 
                  train_accur, test_accur, decoder_history)
    
if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    