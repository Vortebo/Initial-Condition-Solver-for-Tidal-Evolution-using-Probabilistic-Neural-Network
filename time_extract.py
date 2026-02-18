import datetime, os
import numpy as np
import matplotlib.pyplot as plt

def extract_datetime(line):
    date = line[1].split('-')#' '.join(splitline[1],splitline[2])
    time = line[2].split(':')
    year = int(date[0])
    month = int(date[1])
    day = int(date[2])
    hour = int(time[0])
    minute = int(time[1])
    second = int(time[2].split(',')[0])
    microsecond = int(time[2].split(',')[1])*1000
    return datetime.datetime(year,month,day,hour,minute,second,microsecond)

def main():
    path = '/home/jas180011/work/code/Initial-Condition-Solver-for-Tidal-Evolution-using-Probabilistic-Neural-Network/training_output/calculate_new/'
    #path = '/home/vortebo/ctime/Initial-Condition-Solver-for-Tidal-Evolution-using-Probabilistic-Neural-Network/training_output/calculate_new/'
    alllogs = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path,f))]
    alllogs = [f for f in alllogs if f.split('.')[-1]=='log']
    alllogs = [f for f in alllogs if f.split('_')[0]=='ztbd']

    ###TESTING
    alllogs = [f for f in alllogs if len(f.split('_'))==4]
    ###TESTING

    systems = [f.split('_')[1] for f in alllogs]
    systems = list(set(systems))
    # systems=['3348093','9665503']
    # systems=['3348093','4346875']
    
    systemkeys = dict([[systems[i],i] for i in range(len(systems))])
    systemstats= np.zeros((len(systems),2,120))
    # print(systemkeys)
    # print(systemstats)

    alldiffs = []

    for system in systems:
        relevant_logs = [f for f in alllogs if f.split('_')[1]==system]
        relevant_logs.sort()
        # print(relevant_logs)
        length = int(len(relevant_logs)/2)

        for j in [0,1]:
            for log in relevant_logs[length*j:length*(j+1)]:
                with open(os.path.join(path,log)) as f:
                    test_start = None
                    test_end = None
                    test = 0
                    for line in f:
                        if 'Running test' in line:
                            splitline = line.split(' ')
                            # print(splitline)
                            test = int(splitline[6].split('.')[0]) - 1
                            test_start = extract_datetime(splitline)
                            # print(test_start)
                        elif 'Finished test' in line:
                            splitline = line.split(' ')
                            # print(splitline)
                            test_end = extract_datetime(splitline)
                            # print(test_end)
                            # print(test_end == test_start)
                            test_time = (test_end - test_start).total_seconds()/3600
                            systemstats[systemkeys[system]][j][test] = test_time
                            test_start = None
                            test_end = None

        if np.sum(systemstats[systemkeys[system]][0][:]==0)==0 and np.sum(systemstats[systemkeys[system]][1][:]==0)==0:
            systemdiffs = systemstats[systemkeys[system]][1][:] - systemstats[systemkeys[system]][0][:]
            alldiffs.append(systemdiffs)
            plt.hist(systemdiffs,bins=20)
            plt.ylabel('Number of Tests')
            plt.xlabel('Time Saved (hours)')
            plt.title(f'{system} ML Performance')
            plt.savefig(f'{system}_mlperf.pdf')
            plt.close()
    
            times = {
                'With ML': systemstats[systemkeys[system]][0][:],
                'Without ML': systemstats[systemkeys[system]][1][:],
            }
            # fig, ax = plt.subplots(layout='constrained')
            width=0.4
            # multiplier=0
            # for attr,meas in times.items():
            #     offset = width*multiplier
            #     rects=ax.bar(np.arange(120)+offset,meas,width,label=attr)
            #     #print(np.arange(120)+offset)
            #     multiplier+=1
            #     # ax.bar_label(rects, padding=3)
            # # ax.set_xticks(np.arange(120)+width)
            # ax.legend()
            # ax.set_ylabel('Time (hours)')
            # ax.set_xlabel('Test Number')
            # ax.set_title(f'{system} ML Performance')
            plt.bar(np.arange(120),systemdiffs,width*2)
            plt.ylabel('Time Saved (hours)')
            plt.xlabel('Test Number')
            plt.title(f'{system} ML Performance')
            plt.savefig(f'{system}_mllots.pdf')
            plt.close()
    
            #print(f'{system}')
            #print('Total ML time: ',np.sum(systemstats[systemkeys[system]][0][:]))
            #print('Total not ML time: ',np.sum(systemstats[systemkeys[system]][1][:]))
            
        else:
            print(f'deleting {system}')
            systemstats = np.delete(systemstats,systemkeys[system],axis=0)
            current_i = systemkeys[system]
            for key in systemkeys:
                if systemkeys[key]>current_i:
                    systemkeys[key] -= 1

    #print(systemstats)
    biggest_ml = 0
    biggest_no = 0
    for i in range(8):
        time_ml = 0
        time_notml = 0
        for j in range(15):
            time_ml += systemstats[0][0][i+j]
            time_notml += systemstats[0][1][i+j]
        biggest_ml = max(biggest_ml,time_ml)
        biggest_no = max(biggest_no,time_notml)
    #     print('ml',time_ml)
    #     print('no',time_notml)
    # print('ml time: ',time_ml/3600)
    # print('not ml time: ',time_notml/3600)
    print(f'{system}')
    print('Average ML time: ',np.sum(systemstats[systemkeys[system]][0][:])/120)
    print('Average not ML time: ',np.sum(systemstats[systemkeys[system]][1][:])/120)
    print('Average time saved: ',np.sum(systemstats[systemkeys[system]][1][:])/120 - np.sum(systemstats[systemkeys[system]][0][:])/120)

    alldiffs = np.array(alldiffs)
    plt.hist(alldiffs.flatten(),bins=20,range=(-1,1))
    plt.ylabel('Number of Tests')
    plt.xlabel('Time Saved (hours)')
    plt.title('All-System ML Performance')
    plt.savefig('all_mlperf.pdf')
    plt.close()

if __name__ == '__main__':
    main()
