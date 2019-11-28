from keras.layers import Dense, Activation, Add, Multiply, Subtract, Concatenate
from keras.engine import Model, Input


def build_similarity_MLP(config):
    embedding_size = config.get('MODEL.embedding_size')

    method = config.get('SIM_MODEL.method')
    filter_size = config.get('SIM_MODEL.filter_size')
    
    embedding_1 = Input(shape=(embedding_size,), name='embedding_1')
    embedding_2 = Input(shape=(embedding_size,), name='embedding_2')
    
    embeddings = [embedding_1, embedding_2]

    output = None
    if method == 'add':
        output = Add()(embeddings)
    elif method == 'multiply':
        output = Multiply()(embeddings)
    elif method == 'subtract':
        output = Subtract()(embeddings)
    elif method == 'concat':
        output = Concatenate()(embeddings)
    
    output = Dense(filter_size, activation='relu')(output)
    output = Dense(1, activation='sigmoid')(output)

    model = Model(inputs=embeddings, outputs=output)
    return model