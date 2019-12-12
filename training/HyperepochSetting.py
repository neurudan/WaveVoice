from data_handlers.train_data_handler import TrainDataGenerator, get_speaker_list
from data_handlers.test_data_handler import TestDataGenerator
from model.WaveNet import build_WaveNet, make_trainable, change_output_dense
from training.evaluation import calculate_eer

from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam
from wandb.keras import WandbCallback

import wandb
import numpy as np


def setup_optimizer(config):
    lr = config.get('OPTIMIZER.lr')
    optimizers = {
        'sgd': SGD(learning_rate=lr),
        'rmsprop': RMSprop(learning_rate=lr),
        'adagrad': Adagrad(learning_rate=lr),
        'adadelta': Adadelta(learning_rate=lr),
        'adam': Adam(learning_rate=lr),
        'adamax': Adamax(learning_rate=lr),
        'nadam': Nadam(learning_rate=lr)
    }
    return optimizers[config.get('OPTIMIZER.type')]


def train_hyperepochs(config):
    # Setup Test Data-Generator
    test_data_generator = TestDataGenerator(config)
    
    # Setup wandb Callback
    wandb_cb = WandbCallback()

    # Initialize required Variables
    speaker_list = get_speaker_list(config)
    initial_epoch = True
    current_epoch = 0
    
    # Setup Model
    model, loss = build_WaveNet(config)
    model.summary()

    # Run hyperepoch setting
    for hyperepoch in range(config.get('HYPEREPOCH.num_hyperepochs')):
        # Randomly sample 
        speakers = []
        for _ in range(config.get('HYPEREPOCH.num_speakers')):
            speaker = speaker_list[np.random.randint(len(speaker_list))]
            while speaker in speakers:
                speaker = speaker_list[np.random.randint(len(speaker_list))]
            speakers.append(speaker)

        # Setup Train Data-Generator
        train_data_generator = TrainDataGenerator(config, config.get('DATASET.val_active'), train_speakers=speakers)

        train_generator, train_steps = train_data_generator.get_generator('train')
        val_generator, val_steps = train_data_generator.get_generator('val')

        # Pretrain Output Dense Layer
        if not initial_epoch:
            model = change_output_dense(model, config)
            
            # Compile model
            optimizer = setup_optimizer(config)
            model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

            # Train Model
            model.fit_generator(train_generator, steps_per_epoch=train_steps,
                                validation_data=val_generator, validation_steps=val_steps,
                                epochs=current_epoch + config.get('HYPEREPOCH.pretrain_epochs'), initial_epoch=current_epoch,
                                callbacks=[wandb_cb])

            wandb.save('model_hyperepoch_%d_pretrain.h5' % hyperepoch)
                
            model = make_trainable(model)
            current_epoch += config.get('HYPEREPOCH.pretrain_epochs')

        # Compile model
        optimizer = setup_optimizer(config)
        model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

        # Train Model
        model.fit_generator(train_generator, steps_per_epoch=train_steps,
                            validation_data=val_generator, validation_steps=val_steps,
                            epochs=current_epoch + config.get('TRAINING.num_epochs'), initial_epoch=current_epoch,
                            callbacks=[wandb_cb])

        current_epoch += config.get('TRAINING.num_epochs')
        initial_epoch = False

        # Terminate enqueueing process
        train_data_generator.terminate_enqueuer()
        
        # Save model
        wandb.save('model_hyperepoch_%d.h5' % hyperepoch)

        # Test Model (calculate EER)
        current_epoch += 1
        eers = calculate_eer(model, test_data_generator)
        wandb.log(eers, step=current_epoch)



"""

from tqdm import tqdm
from data_handlers.sim_data_handler import SimDataGenerator
from model.Similarity_MLP import build_similarity_MLP

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
                    l, a = sim_model.train_on_batch(x, y)
                    t_a.append(a)
                    t_l.append(l)
                time.sleep(1)
                print('accuracy:     %.5f    loss:     %.5f'%(np.mean(t_a), np.mean(t_l)))
                for _ in tqdm(range(val_steps), ncols=100, ascii=True, desc='val epoch %d'%epoch):
                    x, y = val_generator.__next__()
                    l, a = sim_model.test_on_batch(x, y)
                    v_a.append(a)
                    v_l.append(l)
                time.sleep(1)
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
"""