import os
from datetime import datetime


def mark_attendance(name_in):
    now = datetime.now()
    filename = now.strftime("%Y_%m_%d") + ".csv"
    name_list = []
    if os.path.exists(filename):
        with open(filename, 'r+') as f:
            my_data_list = f.readlines()
            for line in my_data_list:
                entry = line.split(',')
                name_list.append(entry[0])
            if name_in not in name_list:
                f.writelines("\n" + name_in + "," + now.strftime('%H:%M:%S'))
    else:
        with open(filename, 'a+') as f:
            f.writelines("Name,Time")
            my_data_list = f.readlines()
            for line in my_data_list:
                entry = line.split(',')
                name_list.append(entry[0])
            if name_in not in name_list:
                f.writelines("\n" + name_in + "," + now.strftime('%H:%M:%S'))
