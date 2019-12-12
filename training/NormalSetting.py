from data_handlers.train_data_handler import TrainDataGenerator
from data_handlers.test_data_handler import TestDataGenerator
from model.WaveNet import build_WaveNet
from training.evaluation import EvalCallback

from wandb.keras import WandbCallback
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam


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


def train_normal(config):
    # Setup Optimizer
    optimizer = setup_optimizer(config)

    # Setup Model
    model, loss = build_WaveNet(config)
    model.summary()

    # Setup Test Data-Generator
    test_data_generator = TestDataGenerator(config)

    # Setup Train Data-Generator
    train_data_generator = TrainDataGenerator(config, config.get('DATASET.val_active'))
        
    train_generator, train_steps = train_data_generator.get_generator('train')
    val_generator, val_steps = train_data_generator.get_generator('val')
    
    # Setup Callback
    wandb_cb = WandbCallback()
    eval_cb = EvalCallback(config, model, test_data_generator)
    
    # Compile model
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=['accuracy'])
                  
    # Train Model
    model.fit_generator(train_generator,
                        steps_per_epoch=train_steps,
                        validation_data=val_generator,
                        validation_steps=val_steps,
                        epochs=config.get('TRAINING.num_epochs'),
                        callbacks=[eval_cb, wandb_cb])

    # Terminate enqueueing process
    train_data_generator.terminate_enqueuer()