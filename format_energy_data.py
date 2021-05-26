import pymongo

db = pymongo.MongoClient("localhost", 27017).energy

if __name__ == '__main__':

    num = 0

    for file in db.may_18.find():
        time = file['Time']
        A_power = file['A_Power']
        B_power = file['B_Power']
        E_power = file['E_Power']
        F_power = file['F_Power']
        J_power = file['J_Power']
        N_power = file['N_Power']
        Total_power = file['Total_Power']

        with open('may_18\channel_1.dat', 'a') as handle1:
            handle1.write(str(time))
            handle1.write(' ')
            handle1.write(str(round(Total_power, 4)))
            handle1.write('\n')
            print("Processing: ", num)

            num += 1

        with open('may_18\channel_2.dat', 'a') as handle2:
            handle2.write(str(time))
            handle2.write(' ')
            handle2.write(str(round(A_power, 4)))
            handle2.write('\n')

        with open('may_18\channel_3.dat', 'a') as handle3:
            handle3.write(str(time))
            handle3.write(' ')
            handle3.write(str(round(B_power, 4)))
            handle3.write('\n')

        with open('may_18\channel_4.dat', 'a') as handle4:
            handle4.write(str(time))
            handle4.write(' ')
            handle4.write(str(round(E_power, 4)))
            handle4.write('\n')

        with open('may_18\channel_5.dat', 'a') as handle5:
            handle5.write(str(time))
            handle5.write(' ')
            handle5.write(str(round(F_power, 4)))
            handle5.write('\n')

        with open('may_18\channel_6.dat', 'a') as handle6:
            handle6.write(str(time))
            handle6.write(' ')
            handle6.write(str(round(J_power, 4)))
            handle6.write('\n')

        with open('may_18\channel_7.dat', 'a') as handle7:
            handle7.write(str(time))
            handle7.write(' ')
            handle7.write(str(round(N_power, 4)))
            handle7.write('\n')
