# 1、导入包
import pandas as pd
from urllib.request import urlretrieve


# 2、下载读取数据，观察数据结构
def load_data(download=True):
    if download:
        data_path, msg = urlretrieve("http://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data", "car.csv")
        print("Download to car.csv")
    col_names = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]
    data = pd.read_csv("car.csv", names=col_names)
    return data


# 3、转换为onehot格式，
def covert2onehot(data):
    return pd.get_dummies(data, prefix=data.columns)


if __name__ == '__main__':
    car_data = load_data(download=False)
    print(car_data)
    new_data = covert2onehot(car_data)
    new_data.to_csv("car_onehot.csv", index=False)
    print(new_data)
