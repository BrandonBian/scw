from difflib import SequenceMatcher

import struct

import modbus_tk.defines as cst
from modbus_tk import modbus_tcp
import time
import csv
import pymongo
#
db = pymongo.MongoClient("localhost", 27017).energy

# A_Power_List = [0]
# B_Power_List = [0]
# E_Power_List = [0]
# F_Power_List = [0]
# J_Power_List = [0]
# N_Power_List = [0]


def start():
    try:
        master = modbus_tcp.TcpMaster(host='192.168.1.3', port=502)  # 你电表的ip和端口

        ### Extract Information of each module

        A_power = master.execute(1, cst.READ_HOLDING_REGISTERS, 1167, 2)

        B_power = master.execute(2, cst.READ_HOLDING_REGISTERS, 1169, 2)

        E_power = master.execute(5, cst.READ_HOLDING_REGISTERS, 1171, 2)

        F_power = master.execute(6, cst.READ_HOLDING_REGISTERS, 1169, 2)

        J_power = master.execute(10, cst.READ_HOLDING_REGISTERS, 1171, 2)

        N_power = master.execute(14, cst.READ_HOLDING_REGISTERS, 1173, 2)

        return A_power, B_power, E_power, F_power, J_power, N_power

    except:
        print("Smart Meter Connection Error.")
        return 0, 0, 0, 0, 0, 0


def ReadFloat(*args, reverse=True):
    for n, m in args:
        n, m = '%04X' % n, '%04X' % m
    if reverse:
        v = n + m
    else:
        v = m + n
    y_bytes = bytes.fromhex(v)
    y = struct.unpack('!f', y_bytes)[0]
    y = round(y, 6)
    return y


def get_readings():
    # time_list.append(time.strftime("%H:%M:%S", time.localtime(time.time())))

    A_power, B_power, E_power, F_power, J_power, N_power = start()  # power = (,)

    A_Power = ReadFloat(A_power)
    B_Power = ReadFloat(B_power)
    E_Power = ReadFloat(E_power)
    F_Power = ReadFloat(F_power)
    J_Power = ReadFloat(J_power)
    N_Power = ReadFloat(N_power)

    main_Power = A_Power + B_Power + E_Power + F_Power + J_Power + N_Power # Total power

    timestamp = int(time.time() * 1000.0)

    print("A Power", "{:.5f}".format(A_Power))
    print("B Power", "{:.5f}".format(B_Power))
    print("E Power", "{:.5f}".format(E_Power))
    print("F Power", "{:.5f}".format(F_Power))
    print("J Power", "{:.5f}".format(J_Power))
    print("N Power", "{:.5f}".format(N_Power))
    print("")

    db.may_18.insert_one({"Time": timestamp,
                         "Total_Power": main_Power,
                         "A_Power": A_Power,
                         "B_Power": B_Power,
                         "E_Power": E_Power,
                         "F_Power": F_Power,
                         "J_Power": J_Power,
                         "N_Power": N_Power
                         })

    # March 5th Printing large model (12hours)
    # March 6th J from freezer to water bath
    # March 20th systematic framework data (printing model, with skeleton, object, finger detections)
    # April 7th B-freezer, E-incubator, J-refrigerator
    # April 23rd ?
    # May 18th Circuit 1 - 3D printer,


if __name__ == '__main__':

    while True:
        get_readings()
        time.sleep(0.5)
