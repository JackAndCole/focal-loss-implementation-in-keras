import keras.backend as K


def binary_focal_loss(gamma=2):
    """
        Binary form of focal loss.
            FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
        References:
            https://arxiv.org/pdf/1708.02002.pdf
        Usage:
            model.compile(loss=binary_focal_loss(gamma=2), optimizer="adam", metrics=["accuracy"])
            model.fit(class_weight={0:alpha0, 1:alpha1, ...}, ...)
        Notes:
           1. The alpha variable is the class_weight of keras.fit, so in implementation of the focal loss function
           we needn't define this variable.
           2. (important!!!) The output of the loss is the loss value of each training sample, not the total or average
            loss of each batch.
    """

    def focal_loss(y_true, y_pred):
        y_pred = K.constant(y_pred) if not K.is_tensor(y_pred) else y_pred
        y_true = K.cast(y_true, y_pred.dtype)

        return -y_true * K.pow(1 - y_pred, gamma) * K.log((y_pred + K.epsilon())) - \
               (1 - y_true) * K.pow(y_pred, gamma) * K.log((1 - y_pred + K.epsilon()))

    return focal_loss


def categorical_focal_loss(gamma=2):
    """
        Categorical form of focal loss.
            FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
        References:
            https://arxiv.org/pdf/1708.02002.pdf
        Usage:
            model.compile(loss=categorical_focal_loss(gamma=2), optimizer="adam", metrics=["accuracy"])
            model.fit(class_weight={0:alpha0, 1:alpha1, ...}, ...)
        Notes:
           1. The alpha variable is the class_weight of keras.fit, so in implementation of the focal loss function
           we needn't define this variable.
           2. (important!!!) The output of the loss is the loss value of each training sample, not the total or average
            loss of each batch.
    """

    def focal_loss(y_true, y_pred):
        y_pred = K.constant(y_pred) if not K.is_tensor(y_pred) else y_pred
        y_true = K.cast(y_true, y_pred.dtype)

        return K.max(
            -y_true * K.pow(1 - y_pred, gamma) * K.log(y_pred + K.epsilon()), axis=-1)

    return focal_loss


if __name__ == "__main__":
    import tensorflow as tf

    tf.enable_eager_execution()

    # binary focal loss test
    y_true1 = [0, 1, 0, 1]
    y_pred1 = [0.2, 0.6, 0.4, 0.7]
    print(binary_focal_loss()(y_true1, y_pred1))

    # categorical focal loss test
    y_true2 = [[1, 0], [0, 1], [1, 0], [0, 1]]
    y_pred2 = [[0.8, 0.2], [0.4, 0.6], [0.6, 0.4], [0.3, 0.7]]
    print(categorical_focal_loss()(y_true2, y_pred2))
