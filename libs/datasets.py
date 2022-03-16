import numpy as np
import cv2

from datetime import datetime as dt
from datetime import timedelta
import pandas as pd

# get date of train datas


def get_train_days_list():

    start_date = dt(2017, 1, 1, 0, 0)

    days_list = []

    for i in range(672):

        date = start_date + timedelta(days=i)
        days_list.append(date)

    return days_list


# get test list from csv

def get_test_days_list():

    test_terms = pd.read_csv('./libs/inference_terms.csv')
    days_list = pd.to_datetime(test_terms["test_date_1"])

    return days_list


# get data from images
def get_sat_data(date):

    year = date.year
    month = date.month
    day = date.day
    hour = date.hour
    minute = date.minute

    if year == 2019:
        phase = 'test'
    else:
        phase = 'train'

    file_name = f"./datasets/{phase}/{year}/NC_H08_{year}{month:02}{day:02}_{hour:02}{minute:02}_r14_FLDK.02701_02601_tbb_14_4326.tif"

    sat_data = cv2.imread(file_name, -1)

    if sat_data is not None:
        sat_data = (sat_data/256).astype('uint8')

    return sat_data


def make_sat_dataset(resize, phase):

    if phase == 'train':
        start_date_list = get_train_days_list()
    elif phase == 'test':
        start_date_list = get_test_days_list()

    dataset = []

    for start_date in start_date_list:

        data_in_1day = []

        for i in range(48):
            date = start_date + timedelta(minutes=(i * 30))
            sat_data = get_sat_data(date)
            if sat_data is None:
                data_in_1day.append(resized_data)
            else:
                resized_data = cv2.resize(sat_data, (int(994 / resize), int(745 / resize)),
                                          interpolation=cv2.INTER_AREA)
                data_in_1day.append(resized_data)

        data_in_1day = np.array(data_in_1day, dtype='uint8').reshape(
            48, int(745 / resize), int(994 / resize), 1)
        dataset.append(data_in_1day)

    dataset = np.array(dataset, dtype='uint8').reshape(
        len(start_date_list), 48, int(745 / resize), int(994 / resize), 1)
    save_name = f'./datasets/{phase}_sat.npy'

    np.save(save_name, dataset)
