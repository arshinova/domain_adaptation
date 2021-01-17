import os
import matplotlib.pyplot as plt
from segmentation_models import Unet
from generator import ParallelArrayReaderThread


def predict(weights=None):
    """
    Визуализация данных, разметки и предсказаний.

    :param weights: путь к файлу с весами модели
    """
    # Обученная модель (использует сохраненные веса).
    model = Unet('resnet18', input_shape=(192, 192, 1), classes=2, activation='softmax', encoder_weights=None,
                 weights=weights)

    with ParallelArrayReaderThread('data/dataset.hdf5', 1) as train_gen:
        for i in range(10):
            train, target, label = next(train_gen)
            fig, ax = plt.subplots(nrows=2, ncols=3, constrained_layout=True)

            # Предсказание для синтезированного изображения.
            pred_1 = model.predict(train)

            # Предсказание для реального изображения.
            pred_2 = model.predict(target)

            ax[0, 0].set_title('Синтезированное\n изображение (source)', fontdict={'fontsize': 10, 'fontweight': 'medium'})
            ax[0, 0].imshow(train[0, :, :, 0].T.astype(float), cmap='binary')

            ax[1, 0].set_title('Реальное\n изображение (target)', fontdict={'fontsize': 10, 'fontweight': 'medium'})
            ax[1, 0].imshow(target[0, :, :, 0].T.astype(float), cmap='binary')

            ax[0, 1].set_title('Разметка', fontdict={'fontsize': 10, 'fontweight': 'medium'})
            ax[0, 1].imshow(label[0, :, :, 0].T.astype(float), cmap='binary')

            ax[0, 2].set_title('Предсказание\n для source', fontdict={'fontsize': 10, 'fontweight': 'medium'})
            ax[0, 2].imshow(pred_1[0, :, :, 0].T.astype(float), cmap='binary')

            ax[1, 2].set_title('Предсказание\n для target', fontdict={'fontsize': 10, 'fontweight': 'medium'})
            ax[1, 2].imshow(pred_2[0, :, :, 0].T.astype(float), cmap='binary')
            plt.show()


if __name__ == '__main__':
    # weights_path = os.path.join('data', 'trained_model_minent.h5')
    weights_path = os.path.join('data', 'trained_model_advent.h5')
    predict(weights_path)
