

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
import pickle
import argparse
from feature_learning_utils import  student3, traject_learning_dataset_update, default_parameters
from dataset_utils import Syclopic_dataset_generator, test_num_of_trajectories
import cifar10_resnet50_lowResBaseline as cifar10_resnet50
print(os.getcwd() + '/')
#%%
default_parser = default_parameters()
parser = argparse.ArgumentParser(parents=[default_parser])

parser.add_argument('--n_classes', default=10, type=int, help='number of classes') #According to CIFAR10 or 100
parser.add_argument('--n_samples', default=5, type=int, help='number of samples the DRC recieves as inputs')
parser.add_argument('--student_block_size', default=2, type=int, help='number of repetition of each convrnn block')
parser.add_argument('--style', default='spiral_2dir2', type=str, help='choose DRC style of motion')
parser.add_argument('--teacher_net', default='pretrained_teachers/default_cifar_10_teacher.hdf', type=str, help='path to pretrained teacher net')

config = parser.parse_args()
config = vars(config)
print('config  ',config)

parameters = config
TESTMODE = parameters['testmode']


lsbjob = os.getenv('LSB_JOBID')
lsbjob = '' if lsbjob is None else lsbjob

# load dataset
if config['n_classes']==10:
    (trainX, trainY), (testX, testY)= cifar10.load_data()
elif config['n_classes']==100:
    (trainX, trainY), (testX, testY) = cifar100.load_data()
else:
    error

images, labels = trainX, trainY

# layer_name = parameters['layer_name']
num_feature = parameters['num_feature']
trajectory_index = parameters['trajectory_index']
n_samples = parameters['n_samples']
res = parameters['res']
trajectories_num = parameters['trajectories_num']
run_index = parameters['run_index']
dropout = parameters['dropout']
rnn_dropout = parameters['rnn_dropout']
this_run_name = parameters['run_name_prefix'] + '_j' + lsbjob + '_t' + str(int(time.time()))
parameters['this_run_name'] = this_run_name
epochs = parameters['epochs']
int_epochs = parameters['int_epochs']
student_block_size = parameters['student_block_size']
print(parameters)
# scale pixels
def prep_pixels(train, test,resnet_mode=False):
    # convert from integers to floats
    if resnet_mode:
        train_norm = cifar10_resnet50.preprocess_image_input(train)
        test_norm = cifar10_resnet50.preprocess_image_input(test)
        print('preprocessing in resnet mode')
    else:
        train_norm = train.astype('float32')
        test_norm = test.astype('float32')
        #center
        if parameters['dataset_center']:
            mean_image = np.mean(train_norm, axis=0)
            train_norm -= mean_image
            test_norm -= mean_image
        # normalize to range 0-1
        train_norm = train_norm / parameters['dataset_norm']
        test_norm = test_norm /  parameters['dataset_norm']
        # return normalized images
    return train_norm, test_norm

# prepare pixel data
trainX, testX = prep_pixels(trainX, testX, resnet_mode=parameters['resnet_mode'])


#%%
############################### Get Trained Teacher ##########################3

path = os.getcwd() + '/'
if True:
    teacher = keras.models.load_model(parameters['teacher_net'])
    teacher.summary()
    teacher.evaluate(trainX[45000:], trainY[45000:], verbose=2)

    fe_model = teacher.layers[0]
    be_model = teacher.layers[1]


    #%%
    #################### Get Layer features as a dataset ##########################
    print('making feature data')
    intermediate_layer_model = fe_model
    decoder = be_model
    batch_size = 32
    start = 0
    end = batch_size
    train_data = []
    validation_data = []
    upsample_factor = parameters['upsample'] if parameters['upsample'] !=0 else 1
    # train_data = np.zeros([50000,upsample_factor*res,upsample_factor*res,num_feature])
    count = 0

    feature_space = 64
    feature_list = 'all'

    # for batch in range(len(trainX)//batch_size + 1):
    #     count+=1
    #     intermediate_output = intermediate_layer_model(trainX[start:end]).numpy()
    #     train_data[start:end,:,:] = intermediate_output[:,:,:,:]
    #     start += batch_size
    #     end += batch_size


    print('\nLoaded feature data from teacher')

    #%%
    # feature_val_data = train_data[45000:]
    # feature_train_data = train_data[:45000]

    #%%
    ##################### Define Student #########################################
    verbose =parameters['verbose']
    evaluate_prediction_size = 150
    prediction_data_path = path +'predictions/'
    # shape = feature_val_data.shape
    # teacher_mean = np.mean(feature_val_data.reshape(shape[0]*shape[1]*shape[2], shape[3]),axis = 0)
    # teacher_var = np.var(feature_val_data.reshape(shape[0]*shape[1]*shape[2], shape[3]),axis = 0)
    # #print('teacher mean = ', teacher_mean, 'var =', teacher_var)
    # parameters['teacher_mean'] = teacher_mean
    # parameters['teacher_var'] = teacher_var
    parameters['feature_list'] = feature_list
    save_model_path = path + 'saved_models/{}_feature/'.format(this_run_name)
    checkpoint_filepath = save_model_path + '/{}_feature_net_ckpt'.format(this_run_name)

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        mode='min',
        save_best_only=True)
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


    def save_model(net,path,parameters,checkpoint = True):
        home_folder = path + '{}_saved_models/'.format(this_run_name)
        if not os.path.exists(home_folder):
            os.mkdir(home_folder)
        if checkpoint:
            child_folder = home_folder + 'checkpoint/'
        else:
            child_folder = home_folder + 'end_of_run_model/'
        if not os.path.exists(child_folder):
            os.mkdir(child_folder)

        #Saving weights as numpy array
        numpy_weights_path = child_folder + '{}_numpy_weights/'.format(this_run_name)
        if not os.path.exists(numpy_weights_path):
            os.mkdir(numpy_weights_path)
        all_weights = net.get_weights()
        with open(numpy_weights_path + 'numpy_weights_{}'.format(this_run_name), 'wb') as file_pi:
            pickle.dump(all_weights, file_pi)
        #LOAD WITH - pickle.load - and load manualy to model.get_layer.set_weights()

        #save weights with keras
        keras_weights_path = child_folder + '{}_keras_weights/'.format(this_run_name)
        if not os.path.exists(keras_weights_path):
            os.mkdir(keras_weights_path)
        net.save_weights(keras_weights_path + 'keras_weights_{}'.format(this_run_name))
        #LOADING WITH - load_status = sequential_model.load_weights("ckpt")

    #%%

if parameters['student_version']==3:
    student_fun = student3
else:
    error

print('initializing student')


student = student_fun(sample = parameters['max_length'],
                   res = res,
                    activation = parameters['student_nl'],
                    dropout = dropout,
                    rnn_dropout = rnn_dropout,
                    num_feature = num_feature,
                   rnn_layer1 = parameters['rnn_layer1'],
                   rnn_layer2 = parameters['rnn_layer2'],
                   layer_norm = parameters['layer_norm_student'],
                   batch_norm = parameters['batch_norm_student'],
                   conv_rnn_type = parameters['conv_rnn_type'],
                   block_size = parameters['student_block_size'],
                   add_coordinates = parameters['broadcast'],
                   time_pool = parameters['time_pool'],
                   dense_interface=parameters['dense_interface'],
                    loss=parameters['loss'],
                      upsample=parameters['upsample'])
student.summary()

train_accur = []
test_accur = []
# generator parameters:

BATCH_SIZE=32
position_dim = (parameters['n_samples'],parameters['res'],parameters['res'],2) if  parameters['broadcast']==1 else (parameters['n_samples'],2)
movie_dim = (parameters['n_samples'], parameters['res'], parameters['res'], 3)
def args_to_dict(**kwargs):
    return kwargs
generator_params = args_to_dict(batch_size=BATCH_SIZE, movie_dim=movie_dim, position_dim=position_dim, n_classes=None, shuffle=True,
                 prep_data_per_batch=True,one_hot_labels=False, one_random_sample=False,
                                    res = parameters['res'],
                                    n_samples = parameters['n_samples'],
                                    mixed_state = True,
                                    n_trajectories = parameters['trajectories_num'],
                                    trajectory_list = 0,
                                    broadcast=parameters['broadcast'],
                                    style = parameters['style'],
                                    max_length=parameters['max_length'],
                                    noise = parameters['noise'],
                                time_sec=parameters['time_sec'], traj_out_scale=parameters['traj_out_scale'],  snellen=parameters['snellen'],vm_kappa=parameters['vm_kappa'])
print('preparing generators')
# generator 1
train_generator_features = Syclopic_dataset_generator(trainX[:-5000], None, teacher=fe_model, **generator_params)
val_generator_features = Syclopic_dataset_generator(trainX[-5000:].repeat(parameters['val_set_mult'],axis=0), None, teacher=fe_model, validation_mode=True, **generator_params)
# generator 2
train_generator_classifier = Syclopic_dataset_generator(trainX[:-5000], labels[:-5000], **generator_params)
val_generator_classifier = Syclopic_dataset_generator(trainX[-5000:].repeat(parameters['val_set_mult'],axis=0), labels[-5000:].repeat(parameters['val_set_mult'],axis=0), validation_mode=True, **generator_params)

if  parameters['broadcast']==1:
    print('-------- total trajectories {}, out of tries: {}'.format( *test_num_of_trajectories(val_generator_classifier)))

gc.collect()
if True:
    student.evaluate(val_generator_features, verbose = 2)
    # print('{}/{}'.format(epoch+1,epochs))

    if parameters['pretrained_student_path'] is not None:
        # student_path, student_run_name = parameters['pretrained_student_path'].split('saved_models/')
        # student_run_name = student_run_name.split('_feature')[0]
        # student, _  = load_student(path = student_path,  run_name = student_run_name , student=student)
        load_status = student.load_weights(parameters['pretrained_student_path'])

    if not parameters['skip_student_training']:
        student_history = student.fit(train_generator_features,
                        epochs = epochs,
                        validation_data=val_generator_features,
                        verbose = verbose,
                        callbacks=[model_checkpoint_callback,lr_reducer,early_stopper],
                        use_multiprocessing=False) #checkpoints won't really work

        train_accur = np.array(student_history.history['mean_squared_error']).flatten()
        test_accur = np.array(student_history.history['val_mean_squared_error']).flatten()
        save_model(student, save_model_path, parameters, checkpoint = False)
        #student.load_weights(checkpoint_filepath) # todo! works @ orram
        save_model(student, save_model_path, parameters, checkpoint = True)
    student.evaluate(val_generator_features, verbose = 2)


############################# The Student learnt the Features!! #################################################
####################### Now Let's see how good it is in classification ##########################################

#Define a Student_Decoder Network that will take the Teacher weights of the last layers:

student.trainable = parameters['fine_tune_student']
# config = student.get_config() # Returns pretty much every information about your model
# print('debu--------------------------',config)
# print('debu--------------------------',config["layers"][0]["config"])
# print('debu--------------------------',config["layers"][1]["config"])
# student.summary()
input0 = keras.layers.Input(shape=movie_dim)
input1 = keras.layers.Input(shape=position_dim)
x = student((input0,input1))
x = decoder(x)
fro_student_and_decoder = keras.models.Model(inputs=[input0,input1], outputs=x, name='frontend')
if parameters['decoder_optimizer'] == 'Adam':
    opt=tf.keras.optimizers.Adam(lr=2.5e-4)
elif parameters['decoder_optimizer'] == 'SGD':
    opt=tf.keras.optimizers.SGD(lr=2.5e-3)
else:
    error

fro_student_and_decoder.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )



################################## Sanity Check with Teachers Features ###########################################
pre_training_accur = fro_student_and_decoder.evaluate(val_generator_classifier, verbose=2)

################################## Evaluate with Student Features ###################################
print('\nEvaluating students features witout more training')
lr_reducer = keras.callbacks.ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
early_stopper = keras.callbacks.EarlyStopping(
    monitor='val_sparse_categorical_accuracy', min_delta=1e-4, patience=10, verbose=0,
    mode='auto', baseline=None, restore_best_weights=True
)

parameters['pre_training_decoder_accur'] = pre_training_accur[1]
############################ Re-train the half_net with the student training features ###########################
print('\nTraining the base newtwork with the student features')


#generator 2
print('\nTraining the decoder')
decoder_history = fro_student_and_decoder.fit(train_generator_classifier,
                       epochs = parameters['decoder_epochs'] if not TESTMODE else 1,
                       validation_data = val_generator_classifier,
                       verbose = 2,
                       callbacks=[lr_reducer,early_stopper],
            workers=8, use_multiprocessing=True)

home_folder = save_model_path + '{}_saved_models/'.format(this_run_name)
decoder.save(home_folder +'decoder_trained_model') #todo - ensure that weights were updated
if parameters['fine_tune_student']:
    student.save(home_folder +'student_fine_tuned_model')

print('running 5 times on test data')
test_generator_classifier = Syclopic_dataset_generator(testX, testY, **generator_params)
for ii in range(5):
    fro_student_and_decoder.evaluate(test_generator_classifier, verbose=2)

traject_learning_dataset_update(train_accur,test_accur, decoder_history, student,parameters, name = 'full_train_103')