"""
Генераторы для использования в Model.fit_generator(train_gen, ...)
"""
import os
import threading
import h5py
import numpy as np
import matplotlib.pyplot as plt


class ParallelArrayReaderThread(object):
    """
    Класс чтения данных в параллельном потоке
    """
    def __init__(self, hdf5_filename, batch_size):
        self.hf = h5py.File(hdf5_filename, 'r')
        self.x_train = self.hf['x_source']
        self.flt_train = self.hf['y_source']
        self.x_target = self.hf['x_target']
        self.lockr = threading.Lock()
        self.lockr.acquire()
        self.lockw = threading.Lock()
        self.lockw.acquire()
        self.batch_size = batch_size
        self.thread_exit = False
        self.p = threading.Thread(target=self.write_to_queue)
        self.p.start()

    def __iter__(self):
        return self

    def write_to_queue(self):
        while True:
            source_i = np.random.randint(len(self.x_train) - self.batch_size)
            target_i = np.random.randint(len(self.x_target) - self.batch_size)
            rgx = self.x_target[target_i:target_i + self.batch_size].astype(np.float32)
            rgx = rgx - rgx.min(axis=(1, 2, 3), keepdims=True)
            rgx = rgx / rgx.max(axis=(1, 2, 3), keepdims=True)
            cgx = self.x_train[source_i:source_i + self.batch_size].astype(np.float32)
            cgx = cgx - cgx.min(axis=(1, 2, 3), keepdims=True)
            cgx = cgx / cgx.max(axis=(1, 2, 3), keepdims=True)
            self.seis = cgx, rgx  # !
            self.seg = np.copy(self.flt_train[source_i:source_i + self.batch_size].astype(np.float32))
            self.seg = np.concatenate([1 - self.seg, self.seg], axis=-1)  # !

            self.lockr.release()
            self.lockw.acquire()
            if self.thread_exit:
                return

    def next(self):
        self.lockr.acquire()
        ret = self.seis + (self.seg,)  # !
        self.lockw.release()
        return ret

    def terminate(self):
        self.thread_exit = True
        self.p.join(1)

    def __next__(self):
        return self.next()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.terminate()


if __name__ == '__main__':
    with ParallelArrayReaderThread(os.path.join('data', 'dataset.hdf5'), 1) as train_gen:
        for i in range(100):
            x1, x2, y = next(train_gen)
            fig, ax = plt.subplots(nrows=1, ncols=3)
            ax[0].imshow(x1[0, :, :, 0].T.astype(float), cmap='binary')
            ax[1].imshow(x2[0, :, :, 0].T.astype(float), cmap='binary')
            ax[2].imshow(y[0, :, :, 0].T.astype(float), cmap='jet')
            plt.show()
