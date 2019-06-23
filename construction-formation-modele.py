import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from utils import INPUT_SHAPE, batch_generator
import argparse
import os
np.random.seed(0)


def load_data(args):
    """
    lire le fichier csv   et preciser les entrees  sorties
    """
    data_df = pd.read_csv(os.path.join(os.getcwd(), args.data_dir, 'driving_log.csv'), names=['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed'])
    X = data_df[['center', 'left', 'right']].values
    y = data_df['steering'].values
    X_t, X_v, y_t, y_v = train_test_split(X, y, test_size=args.test_size, random_state=0)
    return X_train, X_v, y_t, y_v


def build_model(args):
    """
    CNN 

    """
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5-1.0, input_shape=INPUT_SHAPE))
    model.add(Conv2D(24, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Conv2D(36, 4, 4, activation='elu', subsample=(2, 2)))
    model.add(Conv2D(48, 3, 3, activation='elu', subsample=(2, 2)))
    model.add(Conv2D(64, 3, 3, activation='elu'))
    model.add(Conv2D(72, 3, 3, activation='elu'))
    model.add(Dropout(args.keep_prob))
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))
    model.summary()

    return model


def train_model(model, args, X_t, X_v, y_t, y_v):
    """
    entrainer le model
    """

    checkpoint = ModelCheckpoint('model-{epoch:03d}.h5',monitor='val_loss',verbose=0,save_best_only=args.save_best_only,mode='auto')

    model.compile(loss='mean_squared_error', optimizer=Adam(lr=args.learning_rate))


    model.fit_generator(batch_generator(args.data_dir, X_t, y_t, args.batch_size, True),args.samples_per_epoch,args.nb_epoch,max_q_size=1,validation_data=batch_generator(args.data_dir, X_v, y_v, args.batch_size, False),nb_val_samples=len(X_v),callbacks=[checkpoint],verbose=1)

def s2b(s):
    """
    Convertir la chaine en booleen
    """
    s = s.lower()
    return s == 'true' or s == 'yes' or s == 'y' or s == '1'


def main():
    """
    validation et entrainement du modele
    """
    parser = argparse.ArgumentParser(description='le programme du clonage du comportement ')
    parser.add_argument('-d', help='donnees controle ',   type=str,   default='data')
    parser.add_argument('-t', help='test size fraction',       type=float, default=0.2)
    parser.add_argument('-k', help='drop out probability',        type=float, default=0.5)
    parser.add_argument('-n', help='nombre des  epochs',            type=int,   default=10)
    parser.add_argument('-s', help='echantiollinage ',   type=int,   default=20000)
    parser.add_argument('-b', help='taille',       type=int,   default=40)
    parser.add_argument('-o', help='enregistrer le modele ',type=s2b,   default='true')
    parser.add_argument('-l', help='apprendre ',   type=float, default=1.0e-4)
    args = parser.parse_args()
    print('-' * 30)
    print('Parameters')
    print('-' * 30)
    for key, value in vars(args).items():
        print('{:<20} := {}'.format(key, value))
    print('-' * 30)
    data = load_data(args)
    model = build_model(args)
    train_model(model, args, *data)
if __name__ == '__main__':
    main()
