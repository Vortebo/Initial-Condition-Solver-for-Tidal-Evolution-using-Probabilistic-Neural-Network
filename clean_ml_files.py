import sys
import os
from multiprocessing_util import setup_process
import pandas as pd
import logging

def clean_file(systemname,systempath):

    logger = logging.getLogger(__name__)

    versions = ['1d_period','2d_period','2d_eccentricity']

    for version in versions:

        logger.info(f'Cleaning {version}_{systemname}')
        print(f'Cleaning {version}_{systemname}')

        listofdirs = [x for x in os.listdir(f'/{systempath}/') if x.startswith('2025')]
        previousversionpath = sorted(listofdirs)[-2]

        data_path = f'/{systempath}/poet_output/{version}_{systemname}/datasets/data.csv'
        label_path = f'/{systempath}/poet_output/{version}_{systemname}/datasets/label.csv'
        
        if not os.path.exists(data_path) or not os.path.exists(label_path):
            logger.warning(f'No data or label file for {version}_{systemname}')
            print(f'No data or label file for {version}_{systemname}')
            continue
        current_data = pd.read_csv(data_path, float_precision='round_trip')
        current_label = pd.read_csv(label_path, float_precision='round_trip')

        data_length = len(current_data.iloc[:])
        label_length = len(current_label.iloc[:])

        if data_length != label_length:
            new_length = min(data_length,label_length)
            current_data = current_data.iloc[:new_length]
            current_label = current_label.iloc[:new_length]
            data_length = new_length

        new_data = pd.DataFrame()
        new_label = pd.DataFrame()

        paired_matchs = 0
        unpaired_data = 0
        unpaired_label = 0
        paired_data = list()
        paired_label = list()
        for i in range(data_length):
            if current_data['time'][i] == current_label['time'][i]:
                new_data = new_data._append(current_data.iloc[i])
                new_label = new_label._append(current_label.iloc[i])
                continue
            found_data_match = i in paired_data
            found_label_match = i in paired_label
            for j in range(i+1,data_length):
                if not found_label_match:
                    if current_data['time'][j] == current_label['time'][i]:
                        logger.debug(f'Data at {j} matches label at {i}: ')
                        logger.debug(f'{current_data.iloc[j]}')
                        logger.debug(f'{current_label.iloc[i]}')
                        new_data = new_data._append(current_data.iloc[j])
                        new_label = new_label._append(current_label.iloc[i])
                        found_label_match = True
                        paired_data.append(j)
                        paired_matchs += 1
                if not found_data_match:
                    if current_data['time'][i] == current_label['time'][j]:
                        logger.debug(f'Data at {i} matches label at {j}: ')
                        logger.debug(f'{current_data.iloc[i]}')
                        logger.debug(f'{current_label.iloc[j]}')
                        new_data = new_data._append(current_data.iloc[i])
                        new_label = new_label._append(current_label.iloc[j])
                        found_data_match = True
                        paired_label.append(j)
                        paired_matchs += 1
                if found_data_match and found_label_match:
                    break
            if not found_data_match:
                logger.debug(f'No match found for data at {i}')
                logger.debug(f'{current_data.iloc[i]}')
                unpaired_data += 1
            if not found_label_match:
                logger.debug(f'No match found for label at {i}')
                logger.debug(f'{current_label.iloc[i]}')
                unpaired_label += 1
        paired_matchs = paired_matchs/2
        logger.info(f'Paired matches: {paired_matchs}')
        logger.info(f'Unpaired data: {unpaired_data}')
        logger.info(f'Unpaired label: {unpaired_label}')
        print(f'Paired matches: {paired_matchs}')
        print(f'Unpaired data: {unpaired_data}')
        print(f'Unpaired label: {unpaired_label}')
        
        new_data.to_csv(data_path, index=False, mode='w')
        new_label.to_csv(label_path, index=False, mode='w')

        data_check = pd.read_csv(data_path, float_precision='round_trip')
        label_check = pd.read_csv(label_path, float_precision='round_trip')
        data_length = len(data_check.iloc[:])
        label_length = len(label_check.iloc[:])
        if data_length != label_length:
            logger.error(f'Error: updated data and label lengths do not match for {version}_{systemname}')
            print(f'Error: updated data and label lengths do not match for {version}_{systemname}')

if __name__ == "__main__":
    systemname = str(sys.argv[1])
    systempath = str(sys.argv[2])

    setup_process(
                    fname_datetime_format='%Y%m%d%H%M%S',
                    system=systemname,
                    std_out_err_fname='cleaning_output/{task}/{system}_{now}_{pid:d}.outerr',
                    logging_fname='cleaning_output/{task}/{system}_{now}_{pid:d}.log',
                    logging_verbosity='debug',
                    logging_message_format='%(levelname)s %(asctime)s %(name)s: %(message)s | %(pathname)s.%(funcName)s:%(lineno)d'#,
                    #logging_datetime_format=config.logging_datetime_format
                  )

    clean_file(systemname,systempath)