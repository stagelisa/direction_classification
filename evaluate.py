import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score
import joblib

from dataset import *
from util import *
from model import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help='Model path')
    args = parser.parse_args()

    model = load_model(args.model)

    imu_data_filenames = []
    gt_data = []
    x_gyro, x_acc, y = [], [], []

    for i in range(9, 10):
        data_imu_path = f'/home/huydung/devel/intern/data/1ere/{i}/data_deep/imu/'
        for j in range(len([name for name in os.listdir(data_imu_path) if os.path.isfile(os.path.join(data_imu_path, name))])):
            imu_data_filenames.append(data_imu_path + f'{j}.csv')
            gt_data.append(np.array([0., 1.]))
    for i in range(9, 10):
        data_imu_path = f'/home/huydung/devel/intern/data/2eme/{i}/data_deep/imu/'
        for j in range(len([name for name in os.listdir(data_imu_path) if os.path.isfile(os.path.join(data_imu_path, name))])):
            imu_data_filenames.append(data_imu_path + f'{j}.csv')
            gt_data.append(np.array([1., 0.]))
    
    
    for i, (cur_imu_data_filename, cur_gt_data) in enumerate(zip(imu_data_filenames, gt_data)):
        cur_x_gyro, cur_x_acc, cur_gt = load_cea_dataset(cur_imu_data_filename, cur_gt_data)
        x_gyro.append(cur_x_gyro)
        x_acc.append(cur_x_acc)
        y.append(cur_gt)

    y_pred = model.predict([x_gyro, x_acc], batch_size=1, verbose=0)
    y_pred = np.argmax(y_pred, axis=-1)
        

    
    print(confusion_matrix(np.argmax(y, axis=-1), y_pred))
    print(accuracy_score(np.argmax(y, axis=-1), y_pred))

if __name__ == '__main__':
    main()