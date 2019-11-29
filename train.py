from utils.train_data_handler import TrainDataGenerator, get_speaker_list
from utils.test_data_handler import TestDataGenerator
from utils.sim_data_handler import SimDataGenerator

from utils.path_handler import get_sweep_config_path, get_config_path
from utils.config_handler import Config
from utils.preprocessing import setup_datasets 

from model.Similarity_MLP import build_similarity_MLP
from model.WaveNet import build_WaveNet, make_trainable, change_output_dense

from keras.engine import Input, Model
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam
from keras.metrics import categorical_accuracy
from keras import backend as K

from wandb.keras import WandbCallback

from tqdm import tqdm

from clustering import ClusterCallback, calculate_eer

import numpy as np

import wandb
import json
import os
import argparse

import mlflow
import mlflow.keras


def setup_optimizer(config):
    lr = config.get('OPTIMIZER.lr')
    optimizers = {
        'sgd': SGD(learning_rate=lr, clipnorm=1.),
        'rmsprop': RMSprop(learning_rate=lr, clipnorm=1.),
        'adagrad': Adagrad(learning_rate=lr, clipnorm=1.),
        'adadelta': Adadelta(learning_rate=lr, clipnorm=1.),
        'adam': Adam(learning_rate=lr, clipnorm=1.),
        'adamax': Adamax(learning_rate=lr, clipnorm=1.),
        'nadam': Nadam(learning_rate=lr, clipnorm=1.)
    }
    return optimizers[config.get('OPTIMIZER.type')]


def train(config_name=None, project_name=None):
    K.clear_session()
    
    config = Config(config_name)

    num_epochs = config.get('TRAINING.num_epochs')
    setting = config.get('TRAINING.setting')
    batch_type = config.get('DATASET.batch_type')
    val_active = config.get('DATASET.val_active')
    
    config.store_required_variables()
    run_name = config.update_run_name()


    # Start MlFlow log
    path = get_config_path() + 'global.json'
    global_settings = json.load(open(path, 'r'))

    if global_settings['mlflow']:
        mlflow.set_tracking_uri("databricks")
        mlflow.set_experiment("/Users/neurudan@students.zhaw.ch/" + project_name)
        
        mlflow.start_run(run_name=run_name)
        config.set_mlflow_params()
        mlflow.keras.autolog()


    # Setup Optimizer
    optimizer = setup_optimizer(config)


    # Setup Model
    model, loss = build_WaveNet(config)
    model.summary()


    # Setup Test Data-Generator
    test_data_generator = TestDataGenerator(config)


    if setting == 'normal':
        # Setup Train Data-Generator
        train_data_generator = TrainDataGenerator(config, val_active)
        
        train_generator = train_data_generator.get_generator('train')
        train_steps = train_data_generator.steps_per_epoch

        val_generator = None
        val_steps = None
        if batch_type == 'real' and val_active:
            val_set = config.get('DATASET.val_set')
            val_generator = train_data_generator.get_generator('val')
            val_steps = int(train_steps * val_set / (1 - val_set))


        # Setup Callback
        wandb_cb = WandbCallback()
        cb = ClusterCallback(config, model, test_data_generator)


        # Compile model
        model.compile(optimizer=optimizer,
                    loss=loss,
                    metrics=['accuracy'])


        # Train Model
        model.fit_generator(train_generator,
                            steps_per_epoch=train_steps,
                            validation_data=val_generator,
                            validation_steps=val_steps,
                            epochs=num_epochs,
                            callbacks=[cb, wandb_cb])


        # Terminate enqueueing process
        train_data_generator.terminate_enqueuer()
        if global_settings['mlflow']:
            mlflow.end_run(status='FINISHED')

    elif setting == 'hyperepochs':
        dataset = config.get('DATASET.base')
        speaker_list = config.get('DATASET.speaker_list')

        num_speakers = config.get('HYPEREPOCH.num_speakers')
        pretrain_epochs = config.get('HYPEREPOCH.pretrain_epochs')
        num_hyperepochs = config.get('HYPEREPOCH.num_hyperepochs')
        

        full_speaker_list = get_speaker_list(dataset, speaker_list)

        initial_epoch = True
        current_speaker_id = 0
        current_epoch = 0
        current_sim_epoch = 0

        for hyperepoch in range(num_hyperepochs):

            speakers = []
            for _ in range(num_speakers):
                if len(full_speaker_list) == current_speaker_id:
                    current_speaker_id = 0
                speakers.append(full_speaker_list[current_speaker_id])
                current_speaker_id += 1


            # Setup Train Data-Generator
            train_data_generator = TrainDataGenerator(config, val_active, train_speakers=speakers)
            
            train_generator = train_data_generator.get_generator('train')
            train_steps = train_data_generator.steps_per_epoch

            val_generator = None
            val_steps = None
            if batch_type == 'real' and val_active:
                val_set = config.get('DATASET.val_set')
                val_generator = train_data_generator.get_generator('val')
                val_steps = int(train_steps * val_set / (1 - val_set))


            # Setup Callback
            wandb_cb = WandbCallback()

            if not initial_epoch:
                model = change_output_dense(model, config)
                

                # Compile model
                optimizer = setup_optimizer(config)
                model.compile(optimizer=optimizer,
                              loss=loss,
                              metrics=['accuracy'])


                # Train Model
                model.fit_generator(train_generator,
                                    steps_per_epoch=train_steps,
                                    validation_data=val_generator,
                                    validation_steps=val_steps,
                                    epochs=current_epoch + pretrain_epochs,
                                    callbacks=[wandb_cb],
                                    initial_epoch=current_epoch)

                wandb.save('model_hyperepoch_%d_pretrain.h5' % hyperepoch)
                
                model = make_trainable(model)
                current_epoch += pretrain_epochs


            # Compile model
            optimizer = setup_optimizer(config)
            model.compile(optimizer=optimizer,
                          loss=loss,
                          metrics=['accuracy'])


            # Train Model
            model.fit_generator(train_generator,
                                steps_per_epoch=train_steps,
                                validation_data=val_generator,
                                validation_steps=val_steps,
                                epochs=current_epoch + num_epochs,
                                callbacks=[wandb_cb],
                                initial_epoch=current_epoch)

            current_epoch += num_epochs
            initial_epoch = False


            # Terminate enqueueing process
            train_data_generator.terminate_enqueuer()
            

            # Save model
            wandb.save('model_hyperepoch_%d.h5' % hyperepoch)


            # Setup Embedding model
            embedding_model = Model(inputs=model.input,
                                    outputs=model.layers[-2].output)


            # Setup Train Data-Generator
            sim_data_generator = SimDataGenerator(config, embedding_model)
            
            train_generator = sim_data_generator.get_generator('train')
            train_steps = sim_data_generator.steps_per_epoch

            val_set = config.get('DATASET.val_set')
            val_generator = sim_data_generator.get_generator('val')
            val_steps = int(train_steps * val_set / (1 - val_set))

            
            # Build and train similarity model
            sim_model_epochs = config.get('SIM_MODEL.num_epochs')
            sim_model = build_similarity_MLP(config)
            optimizer = setup_optimizer(config)
            sim_model.compile(optimizer=optimizer,
                              loss='binary_crossentropy',
                              metrics=['accuracy'])

            print('Training Similiarity model...')
            for epoch in range(sim_model_epochs):
                t_a, t_l, v_a, v_l = [], [], [], []
                for _ in tqdm(range(train_steps), ncols=100, ascii=True, desc='train epoch %d'%epoch):
                    x, y = train_generator.__next__()
                    a, l = sim_model.train_on_batch(x, y)
                    t_a.append(a)
                    t_l.append(l)
                print('accuracy:     %.5f    loss:     %.5f'%(np.mean(t_a), np.mean(t_l)))
                for _ in tqdm(range(val_steps), ncols=100, ascii=True, desc='val epoch %d'%epoch):
                    x, y = val_generator.__next__()
                    a, l = sim_model.test_on_batch(x, y)
                    v_a.append(a)
                    v_l.append(l)
                print('val_accuracy: %.5f    val_loss: %.5f\n'%(np.mean(v_a), np.mean(v_l)))
                current_sim_epoch += 1
                current_epoch += 1
                log = {'accuracy': np.mean(t_a), 'loss': np.mean(t_l), 'val_accuracy': np.mean(v_a), 'val_loss': np.mean(v_l), 'sim_ep': current_sim_epoch}
                wandb.log(log, step=current_epoch)

            sim_data_generator.terminate_generator(train_generator)
            sim_data_generator.terminate_generator(val_generator)
            

            # Test Model (calculate EER)
            current_epoch += 1
            eers = calculate_eer(model, test_data_generator, sim_model=sim_model)
            eers['Hyperepoch'] = hyperepoch
            wandb.log(eers, step=current_epoch)

            if global_settings['mlflow']:
                mlflow.end_run(status='FINISHED')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training for the WaveVoice Project')
    parser.add_argument('--project', '-p', dest='project', default=None,
                        help='The name of the project on wandb. (define first using "wandb init")')
    parser.add_argument('--sweep', '-s', dest='sweep', action='store_true',
                        help='If passed, uses the wandb sweep function for gridsearch.')
    parser.add_argument('--config', '-c', dest='config', default='default.json',
                        help='The config file to use - is only used, if --sweep has not been passed. (default is "default.json")')
    parser.add_argument('--sweep-config', '-sc', dest='sweep_config', default='sweep_default.json',
                        help='The sweep config file to use - is only used, if --sweep has been passed aswell. (default is "sweep_default.json")')
    parser.add_argument('--setup', dest='setup', action='store_true',
                        help='If passed, the datasets are being set up.')

    args = parser.parse_args()
    if args.setup:
        setup_datasets()
    else:
        project = args.project
        if project == None:
            project = args.sweep_config.split('.')[0]

        os.environ['WANDB_ENTITY'] = 'bratwolf'
        os.environ['WANDB_AGENT_REPORT_INTERVAL'] = '0'
        os.environ['WANDB_PROJECT'] = project

        if args.sweep:
            path = get_sweep_config_path(args.sweep_config)
            sweep_config = json.load(open(path, 'r'))
            sweep_id = wandb.sweep(sweep_config)
            wandb.agent(sweep_id, function=train)
        else:
            train(config_name=args.config)
