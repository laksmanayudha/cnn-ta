from keras.layers import Input, Dense, Dropout, Conv1D, MaxPool1D, GlobalMaxPool1D, Embedding, Activation, Flatten, Concatenate, MaxPooling1D
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras import Model

def gensim_to_keras_embedding(model, input_length, train_embeddings=False):
    """Get a Keras 'Embedding' layer with weights set from Word2Vec model's learned word embeddings.

    Parameters
    ----------
    train_embeddings : bool
        If False, the returned weights are frozen and stopped from being updated.
        If True, the weights can / will be further updated in Keras.

    Returns
    -------
    `keras.layers.Embedding`
        Embedding layer, to be used as input to deeper network layers.

    """
    keyed_vectors = model.wv  # structure holding the result of training
    weights = keyed_vectors.vectors  # vectors themselves, a 2D numpy array    
    index_to_key = keyed_vectors.index_to_key  # which row in `weights` corresponds to which word?

    layer = Embedding(
        weights.shape[0],
        weights.shape[1],
        weights=[weights],
        input_length=input_length,
        trainable=train_embeddings,
    )
    
    return layer


def get_model(feature_maps, filter_sizes, input_len, embedding, summary=False):
    # Kim Yoon CNN
    sequence_input = Input(shape=(input_len,), dtype='int32')
    
    embedding_layer = gensim_to_keras_embedding(embedding, input_len, True)
    
    embedded_sequences = embedding_layer(sequence_input)

    convs = []
    filter_sizes = filter_sizes

    for fsz in filter_sizes:
        x = Conv1D(feature_maps, fsz, activation='relu',padding='same')(embedded_sequences)
        x = MaxPool1D()(x)
        convs.append(x)

    x = Concatenate(axis=-1)(convs)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(6, activation='softmax')(x)

    model = Model(sequence_input, output)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])
    if summary:
        model.summary()
    
    return model

def get_model_base(feature_maps, filter_sizes, input_len, embedding, summary=False):
    # Kim Yoon CNN
    sequence_input = Input(shape=(input_len,), dtype='int32')
    
    embedding_layer = gensim_to_keras_embedding(embedding, input_len, True)
    
    embedded_sequences = embedding_layer(sequence_input)

    convs = []
    filter_sizes = filter_sizes

    for fsz in filter_sizes:
        x = Conv1D(feature_maps, fsz, activation='relu')(embedded_sequences)
        x = GlobalMaxPool1D()(x)
        convs.append(x)

    x = Concatenate(axis=-1)(convs)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(6, activation='softmax')(x)

    model = Model(sequence_input, output)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])
    if summary:
        model.summary()
    
    return model

def get_model_base_1d(feature_maps, filter_sizes, input_len, summary=False):
    # Kim Yoon CNN
    sequence_input = Input(shape=(1,input_len), dtype='float32')

    convs = []
    filter_sizes = filter_sizes

    for fsz in filter_sizes:
        x = Conv1D(feature_maps, fsz, activation='relu')(sequence_input)
        x = GlobalMaxPool1D()(x)
        convs.append(x)

    x = Concatenate(axis=-1)(convs)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(6, activation='softmax')(x)

    model = Model(sequence_input, output)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])
    if summary:
        model.summary()
    
    return model


def get_model_base_tfidf(feature_maps, filter_sizes, input_len, summary=False):
    # Kim Yoon CNN
    input = Input(shape=(1,input_len), dtype='float32')
    x = Conv1D(feature_maps, filter_sizes, activation='relu')(input)
    x = GlobalMaxPool1D()(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(6, activation='softmax')(x)

    model = Model(input, output)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])
    if summary:
        model.summary()
    
    return model