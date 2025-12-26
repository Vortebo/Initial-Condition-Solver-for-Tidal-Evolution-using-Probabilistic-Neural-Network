import logging

import numpy as np
import pandas as pd

from multiprocessing_util import setup_process

from multiprocessing import Pool
from functools import partial

from bayesian.parse_command_line import parse_command_line
from bayesian.sample_windemuth_et_al import prepare_sampling

def load_tests(system):
    
    alltests = pd.read_csv(
        '/home/vortebo/ctime/Initial-Condition-Solver-for-Tidal-Evolution-using-Probabilistic-Neural-Network/'+str(system)+'_tests.csv'
    )
    keys=alltests.keys()
    # alltests=alltests.to_dict(orient='records')

    # Make a list of dicts for each test
    tests=[]
    for i in range(120):#(8):
        newdict = {}
        for key in keys:
            newdict[key]=float(alltests[key][i])
        tests.append(newdict)

    # Divide all 120 tests into 8 groups of 15 tests each
    tests=[tests[i*15:(i+1)*15] for i in range(8)]
    # tests=[tests[i*1:(i+1)*1] for i in range(8)]
    # for i in range(8):
    #     test_group = alltests[i*15:(i+1)*15]
    #     tests.append(test_group)

    return tests

def run_tests(log_likelihood, system, tests):

    setup_process(
        fname_datetime_format='%Y%m%d%H%M%S',
        system='ztbd_' + str(system),
        std_out_err_fname=logpath+'{task}_new/{system}_{now}_{pid:d}.outerr',
        logging_fname=logpath+'{task}_new/{system}_{now}_{pid:d}.log',
        logging_verbosity='debug',
        logging_message_format='%(levelname)s %(asctime)s %(name)s: %(message)s | %(pathname)s.%(funcName)s:%(lineno)d'#,
        #logging_datetime_format=config.logging_datetime_format
    )

    _logger = logging.getLogger(__name__)

    sample_weights = np.zeros(len(tests[0]))

    _logger.info('Beginning this process\' tests.')
    for test in tests:
        testnum = test.pop('testnum')
        test = np.array(list(test.items()),dtype=float)[:,1]
        testvalues = {'encoded_parameters': test, 'sample_weights_envelope': sample_weights}
        _logger.info(f'Running test {testnum}')
        log_likelihood.calculate_log_likelihood(**testvalues)
        _logger.info(f'Finished test {testnum}')
    _logger.info('This process has concluded.')

def main(config):

    _logger = logging.getLogger(__name__)

    tests = load_tests(config.system)

    # First, with ML
    log_likelihood, _ = prepare_sampling(config)
    thefunction = partial(run_tests, log_likelihood, config.system)
    _logger.info('Starting ML tests for system %s', repr(config.system))
    with Pool(processes=8) as pool:
        pool.map(thefunction, tests)
    # for testgroup in tests:
    #     run_tests(log_likelihood, testgroup)
    _logger.info('Finished ML tests for system %s', repr(config.system))

    # Next, without ML
    config.nn_data_dir = 'no'
    log_likelihood, _ = prepare_sampling(config)
    thefunction = partial(run_tests, log_likelihood, config.system)
    _logger.info('Starting non-ML tests for system %s', repr(config.system))
    with Pool(processes=8) as pool:
        pool.map(thefunction, tests)
    _logger.info('Finished non-ML tests for system %s', repr(config.system))

if __name__ == '__main__':
    logpath='/home/vortebo/ctime/Initial-Condition-Solver-for-Tidal-Evolution-using-Probabilistic-Neural-Network/training_output/'
    setup_process(
        fname_datetime_format='%Y%m%d%H%M%S',
        system='tbd',
        std_out_err_fname=logpath+'{task}_new/{system}_{now}_{pid:d}.outerr',
        logging_fname=logpath+'{task}_new/{system}_{now}_{pid:d}.log',
        logging_verbosity='debug',
        logging_message_format='%(levelname)s %(asctime)s %(name)s: %(message)s | %(pathname)s.%(funcName)s:%(lineno)d'#,
        #logging_datetime_format=config.logging_datetime_format
    )
    main(
        parse_command_line(
                'Properly test ML performance.',
                'test.cfg',
                dissipation=True,
                cluster=False,
                choose_binary='w19',
                spindown=2
            )
    )