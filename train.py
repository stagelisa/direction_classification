import os
import numpy as np
import matplotlib.pyplot as plt
import argparse

from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay

from sklearn.utils import shuffle

from time import time

from dataset import *
from model import *
from util import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', help='Model output name')
    args = parser.parse_args()

    x_gyro = []
    x_acc = []

    y = []

    imu_data_filenames = []
    gt_data = []


    for i in range(9):
        data_imu_path = f'/home/huydung/devel/intern/data/1ere/{i}/data_deep/imu/'
        for j in range(len([name for name in os.listdir(data_imu_path) if os.path.isfile(os.path.join(data_imu_path, name))])):
            imu_data_filenames.append(data_imu_path + f'{j}.csv')
            gt_data.append(np.array([0., 1.]))
    for i in range(9):
        data_imu_path = f'/home/huydung/devel/intern/data/2eme/{i}/data_deep/imu/'
        for j in range(len([name for name in os.listdir(data_imu_path) if os.path.isfile(os.path.join(data_imu_path, name))])):
            imu_data_filenames.append(data_imu_path + f'{j}.csv')
            gt_data.append(np.array([1., 0.]))

    for i, (cur_imu_data_filename, cur_gt_data) in enumerate(zip(imu_data_filenames, gt_data)):
        cur_x_gyro, cur_x_acc, cur_gt = load_cea_dataset(cur_imu_data_filename, cur_gt_data)
        x_gyro.append(cur_x_gyro)
        x_acc.append(cur_x_acc)
        y.append(cur_gt)
    
    x_gyro = np.reshape(x_gyro, (len(x_gyro), x_gyro[0].shape[0], x_gyro[0].shape[1]))
    x_acc = np.reshape(x_acc, (len(x_acc), x_acc[0].shape[0], x_acc[0].shape[1]))

    y = np.vstack(y)

    x_gyro, x_acc, y = shuffle(x_gyro, x_acc, y)

    initial_learning_rate = 3e-4
    lr_schedule = ExponentialDecay(
        initial_learning_rate,
        decay_steps=100000,
        decay_rate=0.97,
        staircase=True)
    pred_model = create_pred_model_6d_quat()
    # train_model = create_train_model_6d_quat(pred_model)
    pred_model.compile(optimizer=Adam(initial_learning_rate), loss='categorical_crossentropy')

    filepath = "model_checkpoint.hdf5"
    model_checkpoint = ModelCheckpoint('model_checkpoint.hdf5', monitor='val_loss', save_best_only=True, verbose=1)
    tensorboard = TensorBoard(log_dir="logs/{}".format(time()), profile_batch=0)

    try:
        history = pred_model.fit([x_gyro, x_acc], y, epochs=20, batch_size=1, verbose=1, callbacks=[model_checkpoint, tensorboard], validation_split=0.1)
        pred_model.load_weights(filepath)
        pred_model.save('last_best_model_with_custom_layer.hdf5')
        # pred_model = create_pred_model_6d_quat(window_size)
        pred_model.set_weights(pred_model.get_weights())
        pred_model.save('%s.hdf5' % args.output)

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        plt.show()

    except KeyboardInterrupt:
        pred_model.load_weights(filepath)
        pred_model.save('last_best_model_with_custom_layer.hdf5')
        # pred_model = create_pred_model_6d_quat(window_size)
        pred_model.set_weights(pred_model.get_weights())
        pred_model.save('%s.hdf5' % args.output)
        print('Early terminate')

    print('Training complete')

if __name__ == '__main__':
    main()