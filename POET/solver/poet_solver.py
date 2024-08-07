#
# Import all the dependencies
#
import os
import random
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
#from POET.solver import poet_logging
import logging
import fcntl

#
# Set all the parameters
#
random.seed(0)
tfpl = tfp.layers
tfd = tfp.distributions


class POET_IC_Solver(object):
    """
    path_to_store: string
        path or directory to store the output
    retrain: boolean
        retraining of the NN model
    type: string
        type of initial condition (orbital period or eccentricity)
    epochs: int (default = 500)
        number of epochs to train the model
    batch_size: int (default = 100)
        number of samples per gradient update
    threshold: int (default = 1000)
        minimum number of training data samples
    verbose: int (default =2)
        'auto', 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.
    version: string
        version of the "type" data used to train the NN model
    features: list of booleans (default = "all features")
        list of booleans for each features used to train the NN model
    """

    def __init__(self, type=None, path_to_store=None, epochs=500, batch_size=100,
                 verbose=2, threshold=1000, version=None, retrain=False, features=None):
        self.epochs = epochs
        self.verbose = verbose
        self.type = type
        self.path = path_to_store
        self.version = version
        self.batch_size = batch_size
        self.threshold = threshold
        self.model = None
        self.y_hat = None
        self.y_sd = None
        self.retrain = retrain
        self.features = features
        #
        # start logging the output
        #
        #poet_logging.start(path=self.path)
        #
        # Check if '/self.path/' file or directory exist
        #
        if not os.path.exists(f'/{self.path}/'):
            raise FileNotFoundError(f"\nFileNotFoundError: '{self.path}/' file or directory "
                                    f"doesn't exist.\n")
        #
        # Create '/self.path/poet_output/{type}' folder if it does not exist already
        #
        if not os.path.exists(f'/{self.path}/poet_output/{self.type}_{self.version}'):
            os.makedirs(f'/{self.path}/poet_output/{self.type}_{self.version}')

    def store_data(self, X_train=None, y_train=None):
        """
        Parameters
        ----------
        X_train: numpy ndarray
            training data set
        y_train: numpy ndarray
            training labels

        Returns
        -------
        the results are stored as CSV files in folder - /{self.path}/poet_output/{self.type}_{self.version}/datasets
        """
        logger = logging.getLogger(__name__)

        #
        # Declare all variables
        #
        file_list = list()
        X_train, y_train = np.array(X_train), np.array(y_train)
        #
        # Create '{self.path}/poet_output/{self.type}_{self.version}/datasets/' folder if it does not exist already
        # Else store all the files from the /datasets/ folder
        #
        if not os.path.exists(f'/{self.path}/poet_output/{self.type}_{self.version}/datasets/'):
            os.makedirs(f'/{self.path}/poet_output/{self.type}_{self.version}/datasets/')
        else:
            for file in os.listdir(f'/{self.path}/poet_output/{self.type}_{self.version}/datasets/'):
                file_list.append(file)
        #
        # Check if the data is a numpy nd-array
        #
        if isinstance(X_train, (np.ndarray, np.generic, list)):
            X_train = np.array(X_train)
        elif isinstance(X_train, pd.DataFrame):
            X_train = X_train.to_numpy()
        else:
            raise TypeError(f"\nTypeError: Unable to load the data for X_train. Expected a format as "
                            f"a numpy array or a pandas dataframe.\n")

        if isinstance(y_train, (np.ndarray, np.generic, list)):
            y_train = np.array(y_train)
        elif isinstance(y_train, pd.DataFrame):
            y_train = y_train.to_numpy()
        else:
            raise TypeError(f"\nTypeError: Unable to load the data for y_train. Expected a format as "
                            f"a numpy array or a pandas dataframe.\n")
        #
        # Change the dimension of the data
        #
        if X_train.ndim < 2:
            X_train = X_train.reshape((1, X_train.shape[0]))
        if y_train.ndim < 2:
            if y_train.ndim == 0:
                y_train = np.array([y_train])
            y_train = y_train.reshape((1, y_train.shape[0]))
        #
        # Store the data in a CSV file
        #
        column_names = [f'{i}' for i in range(X_train.shape[1])]
        column_names = pd.Index(column_names)
        new_data_df = pd.DataFrame(X_train,columns=column_names)
        new_data_df = new_data_df.astype('float64')
        new_labels_df = pd.DataFrame(y_train,columns=pd.Index(['0']))
        new_labels_df = new_labels_df.astype('float64')
        #
        # Append data to the dataframe or create a new dataframe
        #
        skipping = False
        logger.debug('file_list is: %s', repr(file_list))
        for part in ['data', 'label']:
            if skipping:
                continue
            new_df,mode = (new_data_df,'r+') if part == 'data' else (new_labels_df,'a+')
            if f"{part}.csv" in file_list:
                logger.debug('Attempting to update %s.csv', part)
                with open(f"/{self.path}/poet_output/{self.type}_{self.version}/datasets/{part}.csv", mode) as f:
                    logger.debug('Size of file is %s. Attempting to lock file.',repr(os.path.getsize(f.name)))
                    fcntl.lockf(f, fcntl.LOCK_EX)
                    logger.debug('File locked. Data being added: %s', repr(new_df))
                    if part == 'data': # We don't need to do this for label because it's allowed to have duplicates
                        logger.debug('Attempting to check for duplicates.')
                        try:
                            old_df = pd.read_csv(f, float_precision='round_trip')
                            matching_rows = pd.merge(old_df, new_df, how='inner')
                            logger.debug('Matching rows: %s', matching_rows)
                            if not matching_rows.empty:
                                skipping = True
                                logger.debug('Data already exists in %s.csv. Attempting to unlock.', part)
                                fcntl.lockf(f, fcntl.LOCK_UN)
                                logger.debug('File unlocked.')
                                continue
                        except:
                            logger.error('Could not check the file for duplicates. Attempting to unlock.')
                            fcntl.lockf(f, fcntl.LOCK_UN)
                            logger.warning('File unlocked. Raising error.')
                            raise
                    new_df.to_csv(f, header=False, index=False, mode = 'a')
                    logger.debug('File updated. Attempting to unlock.')
                    fcntl.lockf(f, fcntl.LOCK_UN)
                    logger.debug('File unlocked. New size of file is %s.', repr(os.path.getsize(f.name)))
            else:
                logger.debug('Attempting to create %s.csv', part)
                with open(f"/{self.path}/poet_output/{self.type}_{self.version}/datasets/{part}.csv", 'x') as f:
                    logger.debug('Attempting to lock the file')
                    fcntl.lockf(f, fcntl.LOCK_EX)
                    logger.debug('File locked. Attempting to write %s.csv', part)
                    new_df.to_csv(f, index=False)
                    logger.debug('File written successfully. Attempting to unlock.')
                    fcntl.lockf(f, fcntl.LOCK_UN)
                    logger.debug('File unlocked.')
        data = pd.read_csv(f"/{self.path}/poet_output/{self.type}_{self.version}/datasets/data.csv", float_precision='round_trip')
        label = pd.read_csv(f"/{self.path}/poet_output/{self.type}_{self.version}/datasets/label.csv", float_precision='round_trip')
        assert len(data.iloc[:]) == len(label.iloc[:])

        print(f"\nThe data is stored in --{self.path}/poet_output/{self.type}_{self.version}/datasets/ folder!\n")
        logger.debug(f"\nThe data is stored in --{self.path}/poet_output/{self.type}_{self.version}/datasets/ folder!\n")

    def load_data(self):
        logger = logging.getLogger(__name__)
        #
        # Declare all variables
        #
        file_list = list()
        #
        # Check if '{self.path}/poet_output/{self.type}_{self.version}/datasets/' folder exist already
        # Else store all the files from the /datasets/ folder
        #
        if os.path.exists(f'/{self.path}/poet_output/{self.type}_{self.version}/datasets/'):
            for file in os.listdir(f'/{self.path}/poet_output/{self.type}_{self.version}/datasets/'):
                file_list.append(file)
            if "data.csv" not in file_list:
                raise FileNotFoundError(f"\nFileNotFoundError: 'data.csv' file "
                                        f"doesn't exists in the folder - "
                                        f"/{self.path}/poet_output/{self.type}_{self.version}/datasets/.\n")
            if "label.csv" not in file_list:
                raise FileNotFoundError(f"\nFileNotFoundError: 'label.csv' file "
                                        f"doesn't exists in the folder - "
                                        f"/{self.path}/poet_output/{self.type}_{self.version}/datasets/.\n")
        #
        # Load the data
        #
        data_df = pd.read_csv(f"/{self.path}/poet_output/{self.type}_{self.version}/datasets/data.csv")
        labels_df = pd.read_csv(f"/{self.path}/poet_output/{self.type}_{self.version}/datasets/label.csv")
        #
        # Train using specified list of features
        #
        if self.features:
            data_df = data_df.loc[:, self.features]
        #
        # Convert pandas dataframe to numpy ndarray
        #
        data_df = data_df.to_numpy()
        labels_df = labels_df.to_numpy()

        #
        # Remove NaN values
        #
        to_remove = []
        for i in range(len(data_df)):
            if np.isnan(data_df[i]).any():
                logger.debug(f'NaN found in data_df at index {i} '
                                f'data_df[i] is {data_df[i]} '
                                f'labels_df[i] is {labels_df[i]}')
                to_remove.append(i)
        for i in range(len(labels_df)):
            if np.isnan(labels_df[i]).any():
                logger.debug(f'NaN found in labels_df at index {i} '
                                f'data_df[i] is {data_df[i]} '
                                f'labels_df[i] is {labels_df[i]}')
                if i not in to_remove:
                    to_remove.append(i)
        logger.debug(f'Indices to remove: {to_remove}')
        data_df = np.delete(data_df,to_remove,0)
        labels_df = np.delete(labels_df,to_remove,0)

        return data_df, labels_df

    def log_loss(self, y_true, y_pred):
        return -y_pred.log_prob(y_true)
    
    def data_length(self):
        logger = logging.getLogger(__name__)
        y_train = self.load_data()[1]
        length = len(y_train)
        print(length)
        return length
    
    def just_fit(self, X_train = None, y_train = None):
        logger = logging.getLogger(__name__)
        
        if X_train is None or y_train is None:
            X_train, y_train = self.load_data()
        #
        # Fit the model using the X_train and y_train
        # Custom loss function used - log loss
        # Optimizer - Adam
        #
        #
        # event shape: integer vector Tensor representing the shape
        # of single draw from this distribution
        event_shape = 1
        #
        # features: number of features from the training sample
        #
        features = X_train.shape[1]
        #
        # Initialize the Adam optimizer
        #
        opt = tf.keras.optimizers.Adam(learning_rate=0.0003)
        #
        # Store the training loss for each epoch
        #
        csv_logger = tf.keras.callbacks.CSVLogger(f"/{self.path}/poet_output/"
                                                    f"{self.type}_{self.version}/model_training_log.csv")
        #
        # build the model using a independent normal distribution
        #
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(units=tfpl.IndependentNormal.params_size(event_shape),
                                    input_shape=(features,),
                                    kernel_initializer=tf.keras.initializers.Ones()),
                                    #kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.1, stddev=0.05)),
            tfpl.IndependentNormal(event_shape=event_shape)])
        print('weights: ',self.model.weights)
        self.model.compile(loss=self.log_loss, optimizer=opt)
        self.model.fit(X_train, y_train,
                        epochs=self.epochs,
                        batch_size=self.batch_size,
                        verbose=self.verbose,
                        callbacks=[csv_logger])
        #
        # Save the model (model.h5) under the folder - {self.path}/poet_output/{self.type}_{self.version}/
        #
        self.model.save(f"/{self.path}/poet_output/{self.type}_{self.version}/model.h5")
        logger.debug("The POET_IC_Solver model is fitted!")
        logger.debug("The model is stored in -- {self.path}/poet_output/{self.type}_{self.version}/model.h5 directory!")
        #print(f"\nThe POET_IC_Solver model is fitted!\n"
        #        f"\nThe model is stored in -- {self.path}/poet_output/{self.type}_{self.version}/model.h5 "
        #        f"directory!\n")

    def fit_evaluate(self, X_test=None, y_test=None):
        """

        Parameters
        ----------
        X_test: numpy nd-array
            test data sample
        y_test: numpy nd-array (ignored)
            test labels

        Returns
        -------
        y_hat_lower: numpy nd-array
            lower bound of the actual estimate
        y_hat_upper: numpy nd-array
            upper bound of the actual estimate

        Notes: the results are stored as dictionary in folder - /{self.path}/poet_output/{self.type}_{self.version}/
                - as results.pickle
        """
        logger = logging.getLogger(__name__)
        #
        # Declare all variables
        #
        file_list = list()
        count = 0
        # Verify if the NN model already exist or retrain is True
        #
        if os.path.exists(f'/{self.path}/poet_output/{self.type}_{self.version}'):
            for file in os.listdir(f'/{self.path}/poet_output/{self.type}_{self.version}'):
                file_list.append(file)
        #
        # Create new NN model if "model.h5" doesn't exist or retrain = True
        #
        if "model.h5" not in file_list or self.retrain:
            #
            # Load the training data set
            #
            X_train, y_train = self.load_data()
            #
            # Check if the training data set reached the threshold value
            #
            if len(y_train) < self.threshold:
                raise ValueError(f"\nValueError: the training data size (current size - {len(y_train)}) should be greater "
                                f"than equals to the threshold value--{self.threshold} to begin training!\n")
            #
            self.just_fit(X_train, y_train)
        #
        # If "model.h5" exists and retrain = False then load the existing NN model
        #
        else:
            #
            # Load the stored model from the folder - /{self.path}/poet_output/{self.type}_{self.version}/
            #
            self.model = tf.keras.models.load_model(f"/{self.path}/poet_output/{self.type}_{self.version}/model.h5",
                                                    custom_objects={'log_loss': self.log_loss})

        #
        # Check if the data is a numpy nd-array
        #
        if isinstance(X_test, (np.ndarray, np.generic)):
            X_test = np.array(X_test)
        elif isinstance(X_test, pd.DataFrame):
            X_test = X_test.to_numpy()
        else:
            raise TypeError(f"\nTypeError: Unable to load the data for X_test. Expected a format as "
                            f"a numpy array or a pandas dataframe.\n")

        if y_test:
            if isinstance(y_test, (np.ndarray, np.generic)):
                y_test = np.array(y_test)
            elif isinstance(y_test, pd.DataFrame):
                y_test = y_test.to_numpy()
            else:
                raise TypeError(f"\nTypeError: Unable to load the data for y_test. Expected a format as "
                                f"a numpy array or a pandas dataframe.\n")
        #
        # Change the dimension of the data
        #
        if X_test.ndim < 2:
            X_test = X_test.reshape((1, X_test.shape[0]))
        if y_test:
            if y_test.ndim < 2:
                y_test = y_test.reshape((1, y_test.shape[0]))
        
        #
        # Evaluate the model using the expected variables
        #
        if self.features:
            X_test = X_test[:, self.features]

        #
        # Calculate the mean and the std. deviation
        #
        self.y_hat = self.model(X_test).mean()
        self.y_sd = self.model(X_test).stddev()
        #
        # Calculate the lower and the upper bound of the original estimate
        #
        y_hat_lower = self.y_hat - 2 * self.y_sd
        y_hat_upper = self.y_hat + 2 * self.y_sd
        #
        # Calculate the accuracy of the model if y_test is provided
        #
        if y_test:
            for i in range(len(y_test)):
                if (y_test[i] >= y_hat_lower[i]) and (y_test[i] <= y_hat_upper[i]):
                    count += 1
            #
            #
            #
            accuracy = (count/len(y_test))*100
        else:
            accuracy = None
        #
        # Calculate the log ratio - (y_hat_upper/y_hat_lower)
        #
        #with np.errstate(invalid='ignore'):
        #    log_ratio = np.nanmean(np.log(y_hat_upper/y_hat_lower))
        #
        # Store the results as a dictionary
        #
        #data = {"y_hat_lower": y_hat_lower, "y_hat_upper": y_hat_upper,
        #        "accuracy": accuracy, "log_ratio": log_ratio,
        #        "y_hat": self.y_hat, "y_sd": self.y_sd}
        #
        #
        #
        #logger.debug("The POET_IC_Solver model is evaluated!")
        #print(f"\nThe POET_IC_Solver model is evaluated!\n")
        #
        # Store the result in -- '/{self.path}/poet_output/{self.type}/' folder
        #
        #with open(f"/{self.path}/poet_output/{self.type}_{self.version}/results.pickle", 'wb') as file:
        #    pickle.dump(data, file)
        #
        #
        #
        #logger.debug(f"The results are stored in --{self.path}/poet_output/{self.type}_{self.version}/results.pickle "
        #                f"directory!")
        logger.debug(f"\nLower bound of the estimate: {y_hat_lower}"
                        f"\nMean of the estimate: {self.y_hat}"
                        f"\nUpper bound of the estimate: {y_hat_upper}")
        #print(f"The results are stored in --{self.path}/poet_output/{self.type}_{self.version}/results.pickle "
        #        f"directory!\n")
        #print(f"\nLower bound of the estimate: {y_hat_lower}"
        #        f"\nMean of the estimate: {self.y_hat}"
        #        f"\nUpper bound of the estimate: {y_hat_upper}")

        return y_hat_lower, y_hat_upper

        #
        # End of logging
        #
        #poet_logging.stop()


if __name__ == '__main__':

    X_train, y_train = None, None
    X_test, y_test = None, None
    path, version = None, None
    params = {
        "type": "eccentricity",
        "epochs": 30,
        "batch_size": 100,
        "verbose": 2,
        "retrain": False,
        "threshold": 2000,
        "path_to_store": path,
        "version": version,
        "features": [True, True, True, True, True, True]
    }
    poet = POET_IC_Solver(**params)
    poet.store_data(X_train=X_train, y_train=y_train)
    poet.fit_evaluate(X_test=X_test)






