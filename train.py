import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as tf_keras
from segmentation_models import Unet
from segmentation_models.losses import CategoricalCELoss, DiceLoss
from segmentation_models.metrics import Precision, Recall, IOUScore
from generator import ParallelArrayReaderThread


def entropy_loss(labels, predictions):
    """
    Функция ошибки для минимизации эетропии.

    :param labels: метки классов
    :param predictions: предсказанные метки классов

    :return: энтропия (одномерный тензор)
    """
    loss = -tf.reduce_sum(predictions * tf.math.log(predictions + 1e-30) / tf.math.log(2.0))
    loss /= tf.reduce_prod(tf.cast(tf.shape(predictions), tf.float32))
    return loss * 0.001


def prob_to_entropy_tf(inputs):
    """Функция для формирования карты энтропии.

    :param inputs: тензор изображения

    :return: тензор со значениями энтропии в каждом пикселе изображения
    """
    inputs = -inputs * tf.math.log(inputs + 1e-30) / tf.math.log(2.0)
    return inputs


def discriminator(input_shape, name, ndf=32):
    """
    Дискриминатор для adversarial learning.

    :param input_shape: размерность входного тензора
    :param name: имя модели
    :param ndf: количество фильтров

    :return: модель дискриминатор
    """
    input_tensor = tf_keras.layers.Input(shape=input_shape)

    # На вход подается карта энтропии для изображения.
    x = tf_keras.layers.Lambda(prob_to_entropy_tf, output_shape=input_shape)(input_tensor)

    x = tf_keras.layers.Conv2D(ndf, kernel_size=4, strides=2, padding='same')(x)
    x = tf_keras.layers.LeakyReLU(0.2)(x)
    x = tf_keras.layers.Conv2D(ndf * 2, kernel_size=4, strides=2, padding='same')(x)
    x = tf_keras.layers.LeakyReLU(0.2)(x)
    x = tf_keras.layers.Conv2D(ndf * 4, kernel_size=4, strides=2, padding='same')(x)
    x = tf_keras.layers.LeakyReLU(0.2)(x)
    x = tf_keras.layers.Conv2D(ndf * 8, kernel_size=4, strides=2, padding='same')(x)
    x = tf_keras.layers.LeakyReLU(0.2)(x)
    x = tf_keras.layers.Conv2D(1, kernel_size=4, strides=2, padding='same')(x)
    x = tf_keras.layers.GlobalAveragePooling2D()(x)
    x = tf_keras.layers.Activation('sigmoid')(x)
    return tf_keras.models.Model(inputs=input_tensor, outputs=x, name=name)


def train_min_entropy(weights=None):
    """
    Подход 1.
    Обучение модели сегментации и минимизации энтропии.

    :param weights: путь к файлу с весами модели
    """
    # Создание модели Unet (backbone - resnet18).
    source_model = Unet('resnet18', input_shape=(192, 192, 1), classes=2, activation='softmax', encoder_weights=None,
                        weights=weights)

    # Подробное описание модели по слоям.
    source_model.summary()

    # Создание объекта модели для использования другой функции ошибки.
    target_model = tf_keras.models.Model(inputs=source_model.inputs, outputs=source_model.outputs,
                                         name='target_' + source_model.name)

    # Чтение данных из файла по батчам.
    data_generator = ParallelArrayReaderThread(os.path.join('data', 'dataset.hdf5'), 16)

    # Установка параметров моделей.
    source_model.compile(loss=CategoricalCELoss() + DiceLoss(), optimizer=tf_keras.optimizers.Adam(0.001),
                         metrics=[Precision(), Recall(), IOUScore()])
    target_model.compile(loss=entropy_loss, optimizer=tf_keras.optimizers.Adam(0.001))

    # Обучение.
    for iter_number in range(10000):
        train_batch, target_batch, label_batch = next(data_generator)
        res = source_model.train_on_batch(train_batch, label_batch)
        res_target = target_model.train_on_batch(target_batch, label_batch)
        print('{:05d})'.format(iter_number), res, 'target_loss:', res_target)
        if iter_number > 0 and iter_number % 1000 == 0:
            source_model.save(os.path.join('data', 'model_{}.h5'.format(iter_number)))


def train_advent(weights=None):
    """
    Подход 2.
    Обучение модели сегментации и минимизации энтропии с дискриминатором.

    :param weights: путь к файлу с весами модели
    """

    # Размерность входного тензора.
    input_shape = (192, 192, 1)

    # Размерность тензора на выходе из генератора.
    disc_input_shape = (192, 192, 2)

    # Создание и компиляция дискриминатора.
    discriminator_model = discriminator(disc_input_shape, 'discriminator_trainable')
    discriminator_model.compile(optimizer=tf_keras.optimizers.Adam(0.001),
                                metrics=['accuracy'],
                                loss=tf_keras.losses.binary_crossentropy)

    # Создание и компиляция модели Unet (backbone - resnet18).
    generator = Unet('resnet18', input_shape=input_shape, classes=2, activation='softmax', encoder_weights=None,
                     weights=weights)
    generator.compile(optimizer=tf_keras.optimizers.Adam(0.001),
                      loss=CategoricalCELoss() + DiceLoss(),
                      metrics=[Precision(), Recall(), IOUScore()])

    # Входной тензор.
    gan_input = tf_keras.layers.Input(shape=input_shape)

    # Сегментационная карта для входного изображения, полученная с помощью Unet.
    generator_output = generator(gan_input)

    # Заморозка весов дискриминатора.
    for layer in discriminator_model.layers:
        layer.trainable = False

    # Получение выхода из дискримиатора (метка домена).
    disc_gan_output = discriminator_model(generator_output)

    # Создание и компиляция общей модели с двумя выходами(в которой заморожены веса дискриминатора).
    combined_model = tf_keras.models.Model(inputs=gan_input, outputs=[generator_output, disc_gan_output])
    combined_model.compile(
        optimizer=tf_keras.optimizers.Adam(0.001),
        loss={combined_model.output_names[0]: CategoricalCELoss() + DiceLoss(),
              combined_model.output_names[1]: tf_keras.losses.binary_crossentropy},
        metrics={combined_model.output_names[0]: [Precision(), Recall(), IOUScore()],
                 combined_model.output_names[1]: 'accuracy'},
        loss_weights=[0., 0.001]
    )

    # Чтение данных из файла по батчам.
    data_generator = ParallelArrayReaderThread(os.path.join('data', 'dataset.hdf5'), 16)

    # Метки батчей.
    source_labels = np.zeros((data_generator.batch_size, 1))
    target_labels = np.zeros((data_generator.batch_size, 1)) + 1

    # Обучение.
    for iter_number in range(10000):

        train_batch, target_batch, label_batch = next(data_generator)

        # 0. Получение предиктов в генераторе для дальнейшего обучения дискриминатора.
        source_predict = generator.predict(train_batch)
        target_predict = generator.predict(target_batch)

        # 1. Обучение генератора на синтезированных данных (source).
        gen_result = generator.train_on_batch(train_batch, label_batch)

        # 2. Обучение генератора на реальных данных для обмана дискриминатора (замена target_labels на source_labels).
        target_result = combined_model.train_on_batch(target_batch, [label_batch, source_labels])

        # 3. Обучение дискриминатора на синтезированных данных.
        source_disc_res = discriminator_model.train_on_batch(source_predict, source_labels)

        # 4. Обучение дискриминатора на реальных данных.
        target_disc_res = discriminator_model.train_on_batch(target_predict, target_labels)

        print('{:05d})'.format(iter_number))
        print('Generator source: ', gen_result)
        print('Generator target: ', target_result)
        print('Discriminator source: ', source_disc_res)
        print('Discriminator target: ', target_disc_res)
        print()

        if iter_number > 0 and iter_number % 100 == 0:
            generator.save(os.path.join('data', 'model_{}.h5'.format(iter_number)))


if __name__ == '__main__':
    # train_min_entropy()
    train_advent()
