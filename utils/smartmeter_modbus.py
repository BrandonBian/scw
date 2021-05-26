


from difflib import SequenceMatcher

import struct

import modbus_tk.defines as cst
from modbus_tk import modbus_tcp
import csv




A_Accumulated_List = [0]

B_Accumulated_List = [0]

E_Accumulated_List = [0]

F_Accumulated_List = [0]

J_Accumulated_List = [0]

N_Accumulated_List = [0]

A_Power_List = [0]
B_Power_List = [0]
E_Power_List = [0]
F_Power_List = [0]
J_Power_List = [0]
N_Power_List = [0]

A_Current_List = [0]
B_Current_List = [0]
E_Current_List = [0]
F_Current_List = [0]
J_Current_List = [0]
N_Current_List = [0]


def similar(a, b):  # The similarity between two strings
    return SequenceMatcher(None, a, b).ratio()


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

        A_current = master.execute(1, cst.READ_HOLDING_REGISTERS, 1141, 2)
        B_current = master.execute(2, cst.READ_HOLDING_REGISTERS, 1143, 2)
        E_current = master.execute(5, cst.READ_HOLDING_REGISTERS, 1145, 2)
        F_current = master.execute(6, cst.READ_HOLDING_REGISTERS, 1143, 2)
        J_current = master.execute(10, cst.READ_HOLDING_REGISTERS, 1145, 2)
        N_current = master.execute(14, cst.READ_HOLDING_REGISTERS, 1147, 2)

        return A_power, B_power, E_power, F_power, J_power, N_power, A_current, B_current, E_current, F_current, J_current, N_current

    except:
        print("Smart Meter Connection Error.")
        return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0


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


def generateData():
    # time_list.append(time.strftime("%H:%M:%S", time.localtime(time.time())))

    n=1

    A_power, B_power, E_power, F_power, J_power, N_power, A_current, B_current, E_current, F_current, J_current, N_current = start()  # power = (,)

    A_Accumulated_List.append(A_Accumulated_List[len(A_Accumulated_List) - 1] + ReadFloat(A_power) / 3600)

    B_Accumulated_List.append(B_Accumulated_List[len(B_Accumulated_List) - 1] + ReadFloat(B_power) / 3600)

    E_Accumulated_List.append(E_Accumulated_List[len(E_Accumulated_List) - 1] + ReadFloat(E_power) / 3600)

    F_Accumulated_List.append(F_Accumulated_List[len(F_Accumulated_List) - 1] + ReadFloat(F_power) / 3600)

    J_Accumulated_List.append(J_Accumulated_List[len(J_Accumulated_List) - 1] + ReadFloat(J_power) / 3600)

    N_Accumulated_List.append(N_Accumulated_List[len(N_Accumulated_List) - 1] + ReadFloat(N_power) / 3600)

    A_Power_List.append(ReadFloat(A_power))
    B_Power_List.append(ReadFloat(B_power))
    E_Power_List.append(ReadFloat(E_power))
    F_Power_List.append(ReadFloat(F_power))
    J_Power_List.append(ReadFloat(J_power))
    N_Power_List.append(ReadFloat(N_power))

    A_Current_List.append(ReadFloat(A_current))
    B_Current_List.append(ReadFloat(B_current))
    E_Current_List.append(ReadFloat(E_current))
    F_Current_List.append(ReadFloat(F_current))
    J_Current_List.append(ReadFloat(J_current))
    N_Current_List.append(ReadFloat(N_current))

    # All lists prepared
    A_Accumulated_List2 = A_Accumulated_List[-n:]
    B_Accumulated_List2 = B_Accumulated_List[-n:]
    E_Accumulated_List2 = E_Accumulated_List[-n:]
    F_Accumulated_List2 = F_Accumulated_List[-n:]
    J_Accumulated_List2 = J_Accumulated_List[-n:]
    N_Accumulated_List2 = N_Accumulated_List[-n:]

    A_Power_List2 = A_Power_List[-n:]
    B_Power_List2 = B_Power_List[-n:]
    E_Power_List2 = E_Power_List[-n:]
    F_Power_List2 = F_Power_List[-n:]
    J_Power_List2 = J_Power_List[-n:]
    N_Power_List2 = N_Power_List[-n:]

    A_Current_List2 = A_Current_List[-n:]
    B_Current_List2 = B_Current_List[-n:]
    E_Current_List2 = E_Current_List[-n:]
    F_Current_List2 = F_Current_List[-n:]
    J_Current_List2 = J_Current_List[-n:]
    N_Current_List2 = N_Current_List[-n:]

    return_list = [round(A_Accumulated_List2[0], 5),
                   round(B_Accumulated_List2[0], 5),
                   round(E_Accumulated_List2[0], 5),
                   round(F_Accumulated_List2[0], 5),
                   round(J_Accumulated_List2[0], 5),
                   round(N_Accumulated_List2[0], 5),
                   A_Power_List2[0],
                   B_Power_List2[0],
                   E_Power_List2[0],
                   F_Power_List2[0],
                   J_Power_List2[0],
                   N_Power_List2[0],
                   A_Current_List2[0],
                   B_Current_List2[0],
                   E_Current_List2[0],
                   F_Current_List2[0],
                   J_Current_List2[0],
                   N_Current_List2[0]
                   ]



    # print("Modbus Smartmeter: ", return_list)

    return return_list
