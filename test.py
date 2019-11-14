import wandb
import time
import os

os.environ['WANDB_ENTITY'] = "bratwolf"
os.environ['WANDB_PROJECT'] = "sweep-test"

sweep_config = {
    'method': 'grid',
    'parameters': {
        'layers': {
            'values': [32, 64, 96, 128, 256]
        }
    }
}

# %load train_lib.py

def train():
    import numpy as np
    import tensorflow as tf
    import wandb
    config_defaults = {
        'pre.val': 128
    }
    wandb.init(config=config_defaults, magic=True)

    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    train_images.shape
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    wandb.config.update({"epochs": 4, "batch_size": 32})


    key = 'pre.val'

    val = wandb.config.get(key)
    if val is None:
        keys = key.split('.')
        val = wandb.config.get(keys[0])
        for key in keys[1:]:
            val = val[key]

    print(wandb.config.get('epochs'))
    print(val)
    print(wandb.config.get('nonexisting'))
    print(wandb.config.get('layers'))

    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(wandb.config.layers, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.fit(train_images, train_labels, epochs=5,
                  validation_data=(test_images, test_labels))


sweep_id = wandb.sweep(sweep_config)
wandb.agent(sweep_id, function=train)