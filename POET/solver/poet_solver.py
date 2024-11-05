#
# Import all the dependencies
#
import os
import time
import random
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
#from POET.solver import poet_logging
import logging
import fcntl

from sklearn.preprocessing import MinMaxScaler
import joblib

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
                 verbose=2, threshold=1000, version=None, retrain=False, features=None, bin=None, quantile=0.5):
        logger = logging.getLogger(__name__)
        logger.debug('Beginning init')
        self.epochs = epochs
        self.verbose = verbose
        self.type = type
        self.path = path_to_store
        self.version = version
        self.batch_size = batch_size
        self.threshold = threshold
        self.model_auto = None
        self.model_prob = None
        self.y_hat = None
        self.y_sd = None
        self.retrain = retrain
        self.features = features
        self.bin = bin
        self.quantile = quantile

        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print('The following RuntimeError exception is coming from set_memory_growth.')
                print(e)
                pass
            except ValueError as e:
                print('The following ValueError exception is coming from set_memory_growth.')
                print(e)
                pass
        print('IMPORTANT ML THREADS')
        print(tf.config.threading.get_inter_op_parallelism_threads())
        print(tf.config.threading.get_intra_op_parallelism_threads())
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)
        print(tf.config.threading.get_inter_op_parallelism_threads())
        print(tf.config.threading.get_intra_op_parallelism_threads())

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
        problem_counter = 0
        retry = True
        while retry:
            logger.debug('Attempt %s to read data.csv and label.csv', problem_counter+1)
            try:
                data = pd.read_csv(f"/{self.path}/poet_output/{self.type}_{self.version}/datasets/data.csv", float_precision='round_trip')
                label = pd.read_csv(f"/{self.path}/poet_output/{self.type}_{self.version}/datasets/label.csv", float_precision='round_trip')
                retry = False
            except:
                problem_counter += 1
                if problem_counter == 9:
                    raise
                time.sleep(60)
        logger.debug('Data and label read successfully.')
        assert len(data.iloc[:]) == len(label.iloc[:])

        #
        # Find and save the splitpoint
        #
        data.columns = column_names
        read_column = '10' if self.type == '2d_eccentricity' else '5'
        splitpoint = np.quantile(data[read_column].to_numpy(), self.quantile)
        splitpoint=np.array([splitpoint])
        with open(f"/{self.path}/poet_output/{self.type}_{self.version}/datasets/splitpoint.npy", 'wb') as f:
            np.save(f, splitpoint)

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
        def scalar(df_data):
            mean = np.nanmean(df_data)
            std = np.nanmax(df_data) - np.nanmin(df_data)
            scaled_data = (df_data - mean)/std
            return scaled_data
        
        def scale_in(thedataset,col,scalername):
            scaler_1 = MinMaxScaler(feature_range=(0, 1))
            result = scaler_1.fit_transform(thedataset[col].to_numpy().reshape(-1,1))
            joblib.dump(scaler_1, scalername+'.gz')
            return result
        
        logger = logging.getLogger(__name__)
        
        if X_train is None or y_train is None:
            X_train, y_train = self.load_data()
        
        columns = ['lgQ_min','lgQ_break_period','lgQ_powerlaw','final_age','feh','final_orbital_period','primary_mass','secondary_mass','Rprimary','Rsecondary']
        if self.type != '1d_period':
            columns.append('final_eccentricity')
            if self.type == '2d_eccentricity':
                columns.append('initial_eccentricity')
            else:
                columns.append('initial_orbital_period')
        else:
            columns.append('initial_orbital_period')
        new_data_df = pd.DataFrame(X_train)
        new_data_df = new_data_df.astype('float64')
        new_labels_df = pd.DataFrame(y_train)
        new_labels_df = new_labels_df.astype('float64')
        dataset_df = pd.concat([new_data_df, new_labels_df], axis=1)
        dataset_df.columns = columns

        final_column= 'final_orbital_period' if (self.type == '1d_period' or self.type == '2d_period') else 'final_eccentricity'
        initial_column = [columns[-1]]
        logger.debug('Fit setup done.')

        for i in range(2):
            bin_name = '1' if i == 0 else '2'
            splitpoint = np.quantile(dataset_df[final_column].to_numpy(), self.quantile)
            RANGE_START = 0 if i == 0 else splitpoint
            RANGE_END = splitpoint if i == 0 else 0.8 if self.type == '2d_eccentricity' else 100

            above_low = (dataset_df[final_column]>=RANGE_START)
            if RANGE_END == 0.8 or RANGE_END == 100:
                above_high = (dataset_df[final_column]<=RANGE_END)
            else:
                above_high = (dataset_df[final_column]<RANGE_END)
            dataset_df= dataset_df.loc[above_low & above_high]
            
            if len(dataset_df[initial_column]) < 1: # Specific column doesn't matter, just needed to choose one
                raise ValueError('Not enough data in the bin to train the model.')

            """# Normalize the data set - (y - y_mean)/(y_max - y_min)"""
            for col in dataset_df.columns:
                if col not in initial_column:
                    dataset_df[col] = scalar(dataset_df[col].to_numpy())
                elif col == initial_column[0]:
                    dataset_df[col] = scale_in(dataset_df,col,
                                               f"/{self.path}/poet_output/{self.type}_{self.version}/scaler{bin_name}")

            logger.debug('About to prepare.')
            """# Prepare data and label for the auto-encoder"""
            y = dataset_df[initial_column[0]]
            X = dataset_df.drop(initial_column[0], axis=1)
            logger.debug('Prepared.')
            """# Auto-encoder model"""
            #
            # Store the training loss for each epoch
            #
            csv_logger = tf.keras.callbacks.CSVLogger(f"/{self.path}/poet_output/"
                                                        f"{self.type}_{self.version}/auto{bin_name}_training_log.csv")
            logger.debug('logger made')
            n_inputs = X.shape[1]
            logger.debug('inputs shaped')
            n_bottleneck = n_inputs
            logger.debug('bottleneck bottled')
            # define encoder
            visible = tf.keras.layers.Input(shape=(n_inputs,))
            logger.debug('inputs visified')
            e = tf.keras.layers.Dense(n_inputs*2)(visible)
            logger.debug('dense')
            e = tf.keras.layers.BatchNormalization()(e)
            logger.debug('normal')
            e = tf.keras.layers.ReLU()(e)
            logger.debug('reluded')
            # define bottleneck
            bottleneck = tf.keras.layers.Dense(n_bottleneck)(e)
            logger.debug('bottleneck defined')
            # define decoder
            d = tf.keras.layers.Dense(n_inputs*2)(bottleneck)
            logger.debug('dense bottleneck decoder')
            d = tf.keras.layers.BatchNormalization()(d)
            logger.debug('ur such a batch')
            d = tf.keras.layers.ReLU()(d)
            logger.debug('relulu')
            # output layer
            output = tf.keras.layers.Dense(n_inputs, activation='linear')(d)
            logger.debug('output layer')
            # define autoencoder model
            model = tf.keras.Model(inputs=visible, outputs=output)
            logger.debug('AE model made.')
            # compile autoencoder model
            model.compile(optimizer='adam', loss='mse')
            logger.debug('Compiled.')
            # fit the autoencoder model to reconstruct input
            model.fit(X, y, epochs=500, batch_size=50, verbose=2, validation_split=0.2, callbacks=[csv_logger])
            logger.debug('Fit.')
            """# Extract and train the encoder from the auto-encoder
            ### The embedded space from the encoder is same as the original dimension of the data. No compression in the feature space.
            """
            # define an encoder model (without the decoder)
            self.model_auto = tf.keras.Model(inputs=visible, outputs=bottleneck)
            logger.debug('E model made.')
            self.model_auto.compile(optimizer='adam', loss='mse')
            logger.debug('Compiled.')
            # encode the train data
            X_train = self.model_auto.predict(X)
            logger.debug('Encoded.')
            #
            # Save autoencoder model (model.h5) under the folder - {self.path}/poet_output/{self.type}_{self.version}/
            #
            self.model_auto.save(f"/{self.path}/poet_output/{self.type}_{self.version}/auto{bin_name}.h5")
            logger.debug(f"auto_model {bin_name} is fitted!")
            logger.debug(f"The model is stored in -- {self.path}/poet_output/{self.type}_{self.version}/auto{bin_name}.h5 directory!")

            """# Define the probabilistic model"""
            #
            # Store the training loss for each epoch
            #
            csv_logger = tf.keras.callbacks.CSVLogger(f"/{self.path}/poet_output/"
                                                        f"{self.type}_{self.version}/model{bin_name}_training_log.csv")
            event_shape = 1
            input_shape = X_train.shape[1]
            self.model_prob = tf.keras.Sequential([
                    tf.keras.layers.Dense(units=64, input_shape=(input_shape, ), activation='sigmoid'),
                    tf.keras.layers.Dense(units=32, activation='sigmoid'),
                    tf.keras.layers.Dense(units=8, input_shape=(input_shape, ), activation='sigmoid'),
                    tf.keras.layers.Dense(tfpl.IndependentNormal.params_size(event_shape)),
                    tfpl.IndependentNormal(event_shape)
            ])
            logger.debug('Prob mod made.')
            self.model_prob.compile(loss=self.log_loss, optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.005))
            logger.debug('Compiled.')
            self.model_prob.fit(X_train, y,
                                epochs=self.epochs,
                                batch_size=self.batch_size,
                                verbose=self.verbose,
                                callbacks=[csv_logger],
                                validation_split=0.2)
            logger.debug('Fit.')
            #
            # Save the model (model.h5) under the folder - {self.path}/poet_output/{self.type}_{self.version}/
            #
            self.model_prob.save(f"/{self.path}/poet_output/{self.type}_{self.version}/model{bin_name}.h5")
            logger.debug(f"prob_model {bin_name} is fitted!")
            logger.debug(f"The model is stored in -- {self.path}/poet_output/{self.type}_{self.version}/model{bin_name}.h5 directory!")

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
        logger.debug('Beginning fit_evaluate')
        def scale_out(model_mean, y_l, y_u,scalername):
            scaler_1 = joblib.load(scalername+'.gz')
            model_mean = scaler_1.inverse_transform(model_mean.numpy()).reshape(-1)
            y_l = scaler_1.inverse_transform(y_l.numpy()).reshape(-1)
            y_u = scaler_1.inverse_transform(y_u.numpy()).reshape(-1)
            return model_mean, y_l, y_u
        #
        # Declare all variables
        #
        file_list = list()
        count = 0
        # Verify if the NN model already exists or retrain is True
        #
        if os.path.exists(f'/{self.path}/poet_output/{self.type}_{self.version}'):
            for file in os.listdir(f'/{self.path}/poet_output/{self.type}_{self.version}'):
                file_list.append(file)
                
        model_to_load = "model"+self.bin+".h5" if self.bin is not None else "model.h5"
        #
        # Create new NN model if "model.h5" doesn't exist or retrain = True
        #
        if model_to_load not in file_list or self.retrain:
            logger.debug('Model to load is not in file list or we are retraining.')
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
            logger.debug('Starting fit.')
            self.just_fit(X_train, y_train)
        #
        # If "model.h5" exists and retrain = False then load the existing NN model
        #
        else:
            logger.debug('We are just loading an existing model.')
            #
            # Load the stored models from the folder - /{self.path}/poet_output/{self.type}_{self.version}/
            #
            self.model_auto = tf.keras.models.load_model(f"/{self.path}/poet_output/{self.type}_{self.version}/auto{self.bin}.h5")
            logger.debug('Loaded a model.')
            self.model_auto.compile(optimizer='adam', loss='mse')
            logger.debug('Compiled the model.')
            self.model_prob = tf.keras.models.load_model(f"/{self.path}/poet_output/{self.type}_{self.version}/model{self.bin}.h5",
                                                    custom_objects={'log_loss': self.log_loss},
                                                    safe_mode=False)
            logger.debug('Loaded another model.')

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

        # encode the data
        X_test = self.model_auto.predict(X_test)
        logger.debug('Data encoded.')

        #
        # Calculate the mean and the std. deviation
        #
        self.y_hat = self.model_prob(X_test).mean()
        logger.debug('Got mean.')
        self.y_sd = self.model_prob(X_test).stddev()
        logger.debug('Got stddev.')
        #
        # Calculate the lower and the upper bound of the original estimate
        #
        y_hat_lower = self.y_hat - 2 * self.y_sd
        y_hat_upper = self.y_hat + 2 * self.y_sd

        self.y_hat,y_hat_lower,y_hat_upper = scale_out(self.y_hat,y_hat_lower,y_hat_upper,
                                                       f"/{self.path}/poet_output/{self.type}_{self.version}/scaler{self.bin}")
        logger.debug('Scaled out.')
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






