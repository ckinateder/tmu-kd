from typing import Dict
import numpy as np
from tmu.data import TMUDataset
from keras.datasets import cifar10
import cv2

class CIFAR10(TMUDataset):
    def _retrieve_dataset(self) -> Dict[str, np.ndarray]:
        (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
        X_train = np.copy(X_train)
        X_test = np.copy(X_test)

        for i in range(X_train.shape[0]):
            for j in range(X_train.shape[3]):
                    X_train[i,:,:,j] = cv2.adaptiveThreshold(X_train[i,:,:,j], 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2) 

        for i in range(X_test.shape[0]):
            for j in range(X_test.shape[3]):
                X_test[i,:,:,j] = cv2.adaptiveThreshold(X_test[i,:,:,j], 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)

        Y_train = Y_train.reshape(Y_train.shape[0])
        Y_test = Y_test.reshape(Y_test.shape[0])

        return dict(
            x_train=X_train,
            y_train=Y_train,
            x_test=X_test,
            y_test=Y_test
        )

    def _transform(self, name, dataset):
        return dataset


if __name__ == "__main__":
    cifar_ds = CIFAR10()
    cifar_ds.get()

    print(cifar_ds.get())

