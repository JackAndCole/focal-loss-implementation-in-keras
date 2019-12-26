"""A demo shows how to use binary focal loss."""
import numpy as np
from keras.datasets import imdb
from keras.layers import Dense
from keras.models import Input, Model
from keras.regularizers import l2
from sklearn.utils import compute_class_weight

from losses.focal_loss import binary_focal_loss


def create_model(l=0.01):
    inputs = Input(shape=(8000,))
    x = inputs

    x = Dense(32, activation="relu", kernel_regularizer=l2(l), bias_regularizer=l2(l))(x)
    x = Dense(16, activation="relu", kernel_regularizer=l2(l), bias_regularizer=l2(l))(x)

    outputs = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss=binary_focal_loss(gamma=2), optimizer="adam", metrics=["accuracy"])

    return model


def vectorizer_sequences(sequences, dimension=8000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=8000)

    x_train = vectorizer_sequences(x_train)
    x_test = vectorizer_sequences(x_test)

    class_weight = compute_class_weight("balanced", np.unique(y_train), y_train)
    print("class weight:", class_weight)

    model = create_model(l=0.1)
    model.summary()

    # the class weight is the alpha of focal loss, so in focal loss function, we needn't define the alpha variable.
    model.fit(x_train, y_train, batch_size=512, epochs=20, verbose=1, validation_data=(x_test, y_test),
              class_weight=dict(enumerate(class_weight)))

    loss, accuracy = model.evaluate(x_test, y_test)
    print("loss:", loss)
    print("accuracy:", accuracy)
