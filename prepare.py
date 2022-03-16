from libs.datasets import make_sat_dataset

# prepare train and test datasets
make_sat_dataset(resize=10, phase='train')
make_sat_dataset(resize=10, phase='test')
