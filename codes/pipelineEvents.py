from pathlib import Path
from gc import collect
from tqdm import tqdm
import time
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from scikit_learn.sklearn_svdd.svm import SVDD
import warnings
import tensorflow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Input, concatenate, multiply, average, subtract, add, LayerNormalization
from tensorflow.keras.models import Model
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
import sys
import os


def interest_outlier(df):
    df_int = df[df['class'] == 'interest']
    df_out = df[df['class'] == 'outlier']

    return df_int, df_out


def train_test_split_one_class(df_int, num_train):
    df_train = df_int[:num_train]
    df_test = df_int[num_train:]

    return df_train, df_test


def load_dataset(all_one_dataset):
    datasets_dictionary = {}
    if all_one_dataset == 'All':
        basepath = Path('../Events/')
        files_in_basepath = basepath.iterdir()
        for item in files_in_basepath:
            if item.is_file():
                df = pd.read_pickle('../Events/' + item.name)
                datasets_dictionary[item.name.replace('.plk', '')] = df
    else:
        df = pd.read_pickle('../Events/' + all_one_dataset + '.plk')
        datasets_dictionary[all_one_dataset] = df

    return datasets_dictionary


def make_density_information(cluster_list, df_train, df_test, df_outlier):
    l_x_train = []
    l_x_test = []
    l_x_outlier = []

    for cluster in cluster_list:
        kmeans = KMeans(n_clusters=cluster, random_state=0).fit(df_train)

        x_train_temp = silhouette_samples(df_train, kmeans.labels_).reshape(len(df_train), 1)
        l_x_train.append(x_train_temp)

        x_test_temp = silhouette_samples(df_test, kmeans.predict(df_test)).reshape(len(df_test), 1)
        l_x_test.append(x_test_temp)

        x_outlier_temp = silhouette_samples(df_outlier, kmeans.predict(df_outlier)).reshape(len(df_outlier), 1)
        l_x_outlier.append(x_outlier_temp)

    return np.concatenate(l_x_train, axis=1), np.concatenate(l_x_test, axis=1), np.concatenate(l_x_outlier, axis=1)


def make_representation(df_train, df_test, df_outlier, representation_type, cluster_list, parameter_list):
    df_train_embedding = np.array(df_train['DBERTML'].to_list())
    df_train_latlong = np.array(df_train['lat_long'].to_list())
    df_test_embedding = np.array(df_test['DBERTML'].to_list())
    df_test_latlong = np.array(df_test['lat_long'].to_list())
    df_outlier_embedding = np.array(df_outlier['DBERTML'].to_list())
    df_outlier_latlong = np.array(df_outlier['lat_long'].to_list())
    density_train, density_test, density_outlier = make_density_information(cluster_list, df_train_embedding,
                                                                            df_test_embedding,
                                                                            df_outlier_embedding)
    if representation_type == 'TripleVAE':
        epoch = parameter_list[0]
        arq = parameter_list[1]
        operator = parameter_list[2]

        tf.random.set_seed(1)

        tvae, encoder, decoder = triplevae(arq, len(df_train_embedding[0]), len(cluster_list),
                                           len(df_train_latlong[0]), operator)

        tvae.fit([df_train_embedding, density_train, df_train_latlong],
                 [df_train_embedding, density_train, df_train_latlong], epochs=epoch, batch_size=32, verbose=0)

        x_train, _, _ = encoder.predict([df_train_embedding, density_train, df_train_latlong])
        x_test, _, _ = encoder.predict([df_test_embedding, density_test, df_test_latlong])
        x_outlier, _, _ = encoder.predict([df_outlier_embedding, density_outlier, df_outlier_latlong])

    elif representation_type == 'Concatenate' or representation_type == 'AE' or representation_type == 'VAE':
        x_train = np.concatenate([df_train_embedding, density_train, df_train_latlong], axis=1)
        x_test = np.concatenate([df_test_embedding, density_test, df_test_latlong], axis=1)
        x_outlier = np.concatenate([df_outlier_embedding, density_outlier, df_outlier_latlong], axis=1)

        if representation_type == 'AE':
            epoch = parameter_list[0]
            arq = parameter_list[1]

            tf.random.set_seed(1)

            ae, encoder = autoencoder(arq, len(x_train[0]))

            ae.fit(x_train, x_train, epochs=epoch, batch_size=32, verbose=0)

            x_train = encoder.predict(np.array(x_train))
            x_test = encoder.predict(np.array(x_test))
            x_outlier = encoder.predict(np.array(x_outlier))

        elif representation_type == 'VAE':

            epoch = parameter_list[0]
            arq = parameter_list[1]

            tf.random.set_seed(1)

            vae, encoder, decoder = variationalautoencoder(arq, len(x_train[0]))

            vae.fit(x_train, x_train, epochs=epoch, batch_size=32, verbose=0)

            x_train, _, _ = encoder.predict(np.array(x_train))
            x_test, _, _ = encoder.predict(np.array(x_test))
            x_outlier, _, _ = encoder.predict(np.array(x_outlier))

    return x_train, x_test, x_outlier


class VAE(keras.Model):
    def __init__(self, encoder, decoder, factor_multiply, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.factor_multiply = factor_multiply

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                keras.losses.mean_squared_error(data, reconstruction)
            )
            reconstruction_loss *= self.factor_multiply
            kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            kl_loss = tf.reduce_mean(kl_loss)
            kl_loss *= -0.5
            total_loss = reconstruction_loss + kl_loss

            grads = tape.gradient(total_loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

            return {
                "loss": total_loss,
                "reconstruction_loss": reconstruction_loss,
                "kl_loss": kl_loss,
            }


def encoder_vae(arq, input_dim):
    encoder_inputs_old = keras.Input(shape=(input_dim,), name='encoder_input')

    encoder_inputs = LayerNormalization(axis=[0, 1], center=False, scale=False)(encoder_inputs_old)

    if len(arq) == 3:
        first_dense = Dense(arq[0], activation="linear")(encoder_inputs)

        second_dense = Dense(arq[1], activation="linear")(first_dense)

        z_mean = layers.Dense(arq[2], name="Z_mean")(second_dense)
        z_log_var = layers.Dense(arq[2], name="Z_log_var")(second_dense)
        z = Sampling()([z_mean, z_log_var])

    elif len(arq) == 2:
        first_dense = Dense(arq[0], activation="linear")(encoder_inputs)

        z_mean = layers.Dense(arq[1], name="Z_mean")(first_dense)
        z_log_var = layers.Dense(arq[1], name="Z_log_var")(first_dense)
        z = Sampling()([z_mean, z_log_var])

    else:  # len(arq) == 1
        z_mean = layers.Dense(arq[0], name="Z_mean")(encoder_inputs)
        z_log_var = layers.Dense(arq[0], name="Z_log_var")(encoder_inputs)
        z = Sampling()([z_mean, z_log_var])

    encoder = keras.Model([encoder_inputs_old], [z_mean, z_log_var, z], name="Encoder")

    return encoder


def decoder_vae(arq, output_dim):
    latent_inputs = keras.Input(shape=(arq[(len(arq) - 1)],), name='decoder_input')

    if len(arq) == 3:
        first_dense = Dense(arq[1], activation="linear")(latent_inputs)

        second_dense = Dense(arq[0], activation="linear")(first_dense)

        decoder_outputs = Dense(output_dim, activation="linear")(second_dense)

    elif len(arq) == 2:
        first_dense = Dense(arq[0], activation="linear")(latent_inputs)

        decoder_outputs = Dense(output_dim, activation="linear")(first_dense)

    else:  # len(arq) == 1
        decoder_outputs = Dense(output_dim, activation="linear")(latent_inputs)

    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

    return decoder


def variationalautoencoder(arq, input_dim):
    encoder = encoder_vae(arq, input_dim)

    decoder = decoder_vae(arq, input_dim)

    vae = VAE(encoder, decoder, input_dim)

    vae.compile(optimizer=keras.optimizers.Adam())

    return vae, encoder, decoder


def init_metrics():
    metrics = {
        'precision': 0,
        'recall': 0,
        'f1-score': 0,
        'auc_roc': 0,
        'accuracy': 0,
        'time': 0
    }
    return metrics


def save_values(metricas, values):
    for key in metricas.keys():
        metricas[key] = values[key]


def evaluation_one_class(preds_interest, preds_outliers):
    y_true = [1] * len(preds_interest) + [-1] * len(preds_outliers)
    y_pred = list(preds_interest) + list(preds_outliers)
    return classification_report(y_true, y_pred, output_dict=True)


def evaluate_model(x_train, x_test, x_outlier, model):
    one_class_classifier = model.fit(x_train)

    y_pred_interest = one_class_classifier.predict(x_test)

    y_pred_outlier = one_class_classifier.predict(x_outlier)

    score_interest = one_class_classifier.decision_function(x_test)

    score_outlier = one_class_classifier.decision_function(x_outlier)

    y_true = np.array([1] * len(x_test) + [-1] * len(x_outlier))

    fpr, tpr, _ = roc_curve(y_true, np.concatenate([score_interest, score_outlier]))

    dic = evaluation_one_class(y_pred_interest, y_pred_outlier)
    metrics = {'precision': dic['1']['precision'],
               'recall': dic['1']['recall'],
               'f1-score': dic['1']['f1-score'],
               'auc_roc': roc_auc_score(y_true, np.concatenate([score_interest, score_outlier])),
               'accuracy': dic['accuracy']}

    return metrics, fpr, tpr


def evaluate_models(models, reps, file_name, line_parameters, path):
    for model in tqdm(models):
        lp = model + '_' + line_parameters
        fn = file_name + '_' + model.split('_')[0] + '.csv'
        metrics = init_metrics()

        start = time.time()
        values, fpr, tpr = evaluate_model(reps[0], reps[1], reps[2], models[model])
        end = time.time()
        time_ = end - start
        values['time'] = time_

        save_values(metrics, values)

        write_results(metrics, fn, lp, path)


def write_results(metrics, file_name, line_parameters, path):
    if not Path(path + file_name).is_file():
        file_ = open(path + file_name, 'w')
        string = 'Parameters'

        for metric in metrics.keys():
            string += ';' + metric
        string += '\n'

        file_.write(string)
        file_.close()

    file_ = open(path + file_name, 'a')
    string = line_parameters

    for metric in metrics.keys():
        string += ';' + str(metrics[metric])

    string += '\n'
    file_.write(string)
    file_.close()


def autoencoder(arq, input_length):
    encoder_inputs_old = Input(shape=(input_length,), name='encoder_input')

    encoder_inputs = LayerNormalization(axis=[0, 1], center=False, scale=False)(encoder_inputs_old)

    if len(arq) == 3:
        first_dense_encoder = Dense(arq[0], activation="linear")(encoder_inputs)

        second_dense_encoder = Dense(arq[1], activation="linear")(first_dense_encoder)

        encoded = Dense(arq[2], activation="linear")(second_dense_encoder)

        first_dense_decoder = Dense(arq[1], activation="linear")(encoded)

        second_dense_decoder = Dense(arq[0], activation="linear")(first_dense_decoder)

        decoder_output = Dense(input_length, activation="linear")(second_dense_decoder)

    elif len(arq) == 2:
        first_dense_encoder = Dense(arq[0], activation="linear")(encoder_inputs)

        encoded = Dense(arq[1], activation="linear")(first_dense_encoder)

        first_dense_decoder = Dense(arq[0], activation="linear")(encoded)

        decoder_output = Dense(input_length, activation="linear")(first_dense_decoder)

    else:  # len(arq) == 1
        encoded = Dense(arq[0], activation="linear")(encoder_inputs)

        decoder_output = Dense(input_length, activation="linear")(encoded)

    encoder = Model(encoder_inputs_old, encoded)

    autoencoder_model = Model(encoder_inputs_old, decoder_output)

    autoencoder_model.compile(optimizer=tensorflow.keras.optimizers.Adam(), loss='mse')

    return autoencoder_model, encoder


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim), seed=1)
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class TVAE(keras.Model):
    def __init__(self, encoder, decoder, factor_multiply_embedding, factor_multiply_density, factor_multiply_latlong,
                 **kwargs):
        super(TVAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.factor_multiply_embedding = factor_multiply_embedding
        self.factor_multiply_density = factor_multiply_density
        self.factor_multiply_latlong = factor_multiply_latlong

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder((data[0], data[1], data[2]))

            reconstruction = self.decoder(z)

            embedding_loss = tf.reduce_mean(
                keras.losses.mean_squared_error(data[0], reconstruction[0])
            )

            embedding_loss *= self.factor_multiply_embedding

            density_loss = tf.reduce_mean(
                keras.losses.mean_squared_error(data[1], reconstruction[1])
            )

            density_loss *= self.factor_multiply_density

            latlong_loss = tf.reduce_mean(
                keras.losses.mean_squared_error(data[2], reconstruction[2])
            )

            latlong_loss *= self.factor_multiply_latlong

            kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            kl_loss = tf.reduce_mean(kl_loss)
            kl_loss *= -0.5
            total_loss = embedding_loss + density_loss + latlong_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        return {
            "total loss": total_loss,
            "embedding loss": embedding_loss,
            "density loss": density_loss,
            "Lat-Long loss": latlong_loss,
            "kl loss": kl_loss
        }


def encoder_tvae(arq, embedding_dim, density_dim, latlong_dim, operator):
    embedding_inputs = keras.Input(shape=(embedding_dim,), name='first_input_encoder')
    density_inputs = keras.Input(shape=(density_dim,), name='second_input_encoder')
    latlong_inputs = keras.Input(shape=(latlong_dim,), name='third_input_encoder')

    lat_long_norm = LayerNormalization(axis=[0, 1], center=False, scale=False)(latlong_inputs)

    l1 = Dense(np.max([embedding_dim, density_dim, latlong_dim]), activation='linear')(embedding_inputs)
    l2 = Dense(np.max([embedding_dim, density_dim, latlong_dim]), activation='linear')(density_inputs)
    l3 = Dense(np.max([embedding_dim, density_dim, latlong_dim]), activation='linear')(lat_long_norm)

    fusion = None
    if operator == 'concatenate':
        fusion = concatenate([l1, l2, l3])
    if operator == 'multiply':
        fusion = multiply([l1, l2, l3])
    if operator == 'average':
        fusion = average([l1, l2, l3])
    if operator == 'subtract':
        fusion = subtract([l1, l2])
        fusion = subtract([fusion, l3])
    if operator == 'add':
        fusion = add([l1, l2, l3])

    if len(arq) == 3:
        first_dense = Dense(arq[0], activation="linear")(fusion)

        second_dense = Dense(arq[1], activation="linear")(first_dense)

        z_mean = layers.Dense(arq[2], name="Z_mean")(second_dense)
        z_log_var = layers.Dense(arq[2], name="Z_log_var")(second_dense)
        z = Sampling()([z_mean, z_log_var])

    elif len(arq) == 2:
        first_dense = Dense(arq[0], activation="linear")(fusion)

        z_mean = layers.Dense(arq[1], name="Z_mean")(first_dense)
        z_log_var = layers.Dense(arq[1], name="Z_log_var")(first_dense)
        z = Sampling()([z_mean, z_log_var])

    else:  # len(arq) == 1
        z_mean = layers.Dense(arq[0], name="Z_mean")(fusion)
        z_log_var = layers.Dense(arq[0], name="Z_log_var")(fusion)
        z = Sampling()([z_mean, z_log_var])

    encoder = keras.Model([embedding_inputs, density_inputs, latlong_inputs], [z_mean, z_log_var, z], name="encoder")

    return encoder


def decoder_tvae(arq, embedding_dim, density_dim, latlong_dim):
    latent_inputs = keras.Input(shape=(arq[(len(arq) - 1)],), name='input_decoder')

    if len(arq) == 3:
        first_dense = Dense(arq[1], activation="linear")(latent_inputs)

        second_dense = Dense(arq[0], activation="linear")(first_dense)

        embedding_outputs = Dense(embedding_dim, activation="linear")(second_dense)

        density_outputs = Dense(density_dim, activation="linear")(second_dense)

        latlong_outputs = Dense(latlong_dim, activation="linear")(second_dense)

    elif len(arq) == 2:
        first_dense = Dense(arq[0], activation="linear")(latent_inputs)

        embedding_outputs = Dense(embedding_dim, activation="linear")(first_dense)

        density_outputs = Dense(density_dim, activation="linear")(first_dense)

        latlong_outputs = Dense(latlong_dim, activation="linear")(first_dense)

    else:  # len(arq) == 1
        embedding_outputs = Dense(embedding_dim, activation="linear")(latent_inputs)

        density_outputs = Dense(density_dim, activation="linear")(latent_inputs)

        latlong_outputs = Dense(latlong_dim, activation="linear")(latent_inputs)

    decoder = keras.Model(latent_inputs, [embedding_outputs, density_outputs, latlong_outputs], name="decoder")

    return decoder


def triplevae(arq, embedding_dim, density_dim, latlong_dim, operator):
    encoder = encoder_tvae(arq, embedding_dim, density_dim, latlong_dim, operator)

    decoder = decoder_tvae(arq, embedding_dim, density_dim, latlong_dim)

    tvae = TVAE(encoder, decoder, embedding_dim, density_dim, latlong_dim)

    tvae.compile(optimizer=keras.optimizers.Adam())

    return tvae, encoder, decoder


def make_prepro_evaluate(df_train, df_test, df_out, preprocessing, line_parameters, file_name, path_results, models,
                         cluster_list=(), parameter_list=()):
    representations = make_representation(df_train, df_test, df_out, preprocessing, cluster_list=cluster_list,
                                          parameter_list=parameter_list)

    evaluate_models(models, representations, file_name, line_parameters, path_results)

    del representations
    collect()


def preprocessing_evaluate(datasets_dictionary, dataset, preprocessing, models):
    path_results = '../results/'
    num_train = 2000
    cluster_matrix = [[3, 6, 9, 12], [2, 4, 6, 8, 10], [3, 5, 7, 9, 11], [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]
    epochs = [5, 10, 25, 50]
    arqs = [[384, 128], [256, 64], [64, 2], [128]]
    operators = ['multiply', 'concatenate', 'subtract', 'add', 'average']

    df_int, df_out = interest_outlier(datasets_dictionary[dataset])

    df_train, df_test = train_test_split_one_class(df_int, num_train)

    file_name = dataset + '_' + preprocessing

    line_parameters = ''

    for cluster_list in cluster_matrix:
        if preprocessing == 'Concatenate':
            line_parameters = str(cluster_list)
            make_prepro_evaluate(df_train, df_test, df_out, preprocessing, line_parameters, file_name, path_results,
                                 models, cluster_list=cluster_list)
        else:
            for epoch in epochs:
                for arq in arqs:
                    if preprocessing == 'AE' or preprocessing == 'VAE':
                        line_parameters = str(cluster_list) + '_' + str(epoch) + '_' + str(arq)

                        parameter_list = (epoch, arq)

                        make_prepro_evaluate(df_train, df_test, df_out, preprocessing, line_parameters,
                                             file_name, path_results, models, cluster_list=cluster_list,
                                             parameter_list=parameter_list)

                    else:
                        for operator in operators:
                            line_parameters = str(cluster_list) + '_' + str(epoch) + '_' + str(arq) + '_' + str(
                                operator)

                            parameter_list = (epoch, arq, operator)

                            make_prepro_evaluate(df_train, df_test, df_out, preprocessing, line_parameters,
                                                 file_name, path_results, models, cluster_list=cluster_list,
                                                 parameter_list=parameter_list)

    del df_train
    del df_test
    del df_int
    del df_out
    collect()


def run(datasets_dictionary, models, all_one_dataset, all_one_preprocessing):
    prepros = ['Concatenate', 'AE', 'VAE', 'TripleVAE']

    if all_one_dataset != 'All':
        if all_one_preprocessing != 'All':
            preprocessing_evaluate(datasets_dictionary, all_one_dataset, all_one_preprocessing, models)
        else:
            for prepro in prepros:
                preprocessing_evaluate(datasets_dictionary, all_one_dataset, prepro, models)
    else:
        for dataset in tqdm(datasets_dictionary.keys()):
            if all_one_preprocessing != 'All':
                preprocessing_evaluate(datasets_dictionary, dataset, all_one_preprocessing, models)
            else:
                for prepro in prepros:
                    preprocessing_evaluate(datasets_dictionary, dataset, prepro, models)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    print('Start')

    os.system("../download_datasets.sh")

    models_svdd = {
        'SVDD_Linear_scale': SVDD(kernel='linear', gamma='scale'),
        'SVDD_Linear_auto': SVDD(kernel='linear', gamma='auto'),
        'SVDD_Sigmoid_scale': SVDD(kernel='sigmoid', gamma='scale'),
        'SVDD_Sigmoid_auto': SVDD(kernel='sigmoid', gamma='auto'),
        'SVDD_Poly_2_scale': SVDD(kernel='poly', degree=2, gamma='scale'),
        'SVDD_Poly_2_auto': SVDD(kernel='poly', degree=2, gamma='auto'),
        'SVDD_Poly_3_scale': SVDD(kernel='poly', degree=3, gamma='scale'),
        'SVDD_Poly_3_auto': SVDD(kernel='poly', degree=3, gamma='auto'),
        'SVDD_Poly_4_scale': SVDD(kernel='poly', degree=4, gamma='scale'),
        'SVDD_Poly_4_auto': SVDD(kernel='poly', degree=4, gamma='auto'),
        'SVDD_RBF_0.001_scale': SVDD(kernel='rbf', nu=0.001, gamma='scale'),
        'SVDD_RBF_0.01_scale': SVDD(kernel='rbf', nu=0.01, gamma='scale'),
        'SVDD_RBF_0.05_scale': SVDD(kernel='rbf', nu=0.05, gamma='scale'),
        'SVDD_RBF_0.1_scale': SVDD(kernel='rbf', nu=0.1, gamma='scale'),
        'SVDD_RBF_0.2_scale': SVDD(kernel='rbf', nu=0.2, gamma='scale'),
        'SVDD_RBF_0.3_scale': SVDD(kernel='rbf', nu=0.3, gamma='scale'),
        'SVDD_RBF_0.4_scale': SVDD(kernel='rbf', nu=0.4, gamma='scale'),
        'SVDD_RBF_0.5_scale': SVDD(kernel='rbf', nu=0.5, gamma='scale'),
        'SVDD_RBF_0.6_scale': SVDD(kernel='rbf', nu=0.6, gamma='scale'),
        'SVDD_RBF_0.7_scale': SVDD(kernel='rbf', nu=0.7, gamma='scale'),
        'SVDD_RBF_0.8_scale': SVDD(kernel='rbf', nu=0.8, gamma='scale'),
        'SVDD_RBF_0.9_scale': SVDD(kernel='rbf', nu=0.9, gamma='scale'),
        'SVDD_RBF_0.001_auto': SVDD(kernel='rbf', nu=0.001, gamma='auto'),
        'SVDD_RBF_0.01_auto': SVDD(kernel='rbf', nu=0.01, gamma='auto'),
        'SVDD_RBF_0.05_auto': SVDD(kernel='rbf', nu=0.05, gamma='auto'),
        'SVDD_RBF_0.1_auto': SVDD(kernel='rbf', nu=0.1, gamma='auto'),
        'SVDD_RBF_0.2_auto': SVDD(kernel='rbf', nu=0.2, gamma='auto'),
        'SVDD_RBF_0.3_auto': SVDD(kernel='rbf', nu=0.3, gamma='auto'),
        'SVDD_RBF_0.4_auto': SVDD(kernel='rbf', nu=0.4, gamma='auto'),
        'SVDD_RBF_0.5_auto': SVDD(kernel='rbf', nu=0.5, gamma='auto'),
        'SVDD_RBF_0.6_auto': SVDD(kernel='rbf', nu=0.6, gamma='auto'),
        'SVDD_RBF_0.7_auto': SVDD(kernel='rbf', nu=0.7, gamma='auto'),
        'SVDD_RBF_0.8_auto': SVDD(kernel='rbf', nu=0.8, gamma='auto'),
        'SVDD_RBF_0.9_auto': SVDD(kernel='rbf', nu=0.9, gamma='auto')
    }

    all_one_dataset = sys.argv[1]

    all_one_preprocessing = sys.argv[2]

    datasets_dic = load_dataset(all_one_dataset)

    run(datasets_dic, models_svdd, all_one_dataset, all_one_preprocessing)

    print('Done!')
