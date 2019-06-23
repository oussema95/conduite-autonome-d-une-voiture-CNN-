from keras.callbacks import ModelCheckpoint
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.optimizers import Adam

from utils import INPUT_SHAPE, batch_generator
import argparse
import os
np.random.seed(0)


def chargement_data(args):
    """
    lire le fichier csv   et preciser les entrees  sorties
    """
    data_df = pd.read_csv(os.path.join(os.getcwd(), args.data_dir, 'driving_log.csv'), names=['center', 'gauche', 'droite', 'direction', 'throttle', 'reverse', 'vitesse'])
    X = data_df[['center', 'gauche', 'droite']].values
    y = data_df['steering'].values
    X_t, X_v, y_t, y_v = train_test_split(X, y, test_size=args.test_size, random_state=0)
    return X_t, X_v, y_t, y_v


def constuction_modele(args):
    """
    le modele  NVIDIA  a ete utilisé  ici on trouve la construction du modele

    """
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5-1.0, input_shape=INPUT_SHAPE))
    model.add(Conv2D(12, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Conv2D(24, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Conv2D(36 5, 5, activation='elu', subsample=(2, 2)))
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


def trainer_modele(model, args, X_t, X_v, y_t, y_v):
    """
    Trainer le modele
    """

    checkpoint = ModelCheckpoint('model.h5',monitor='val_loss',verbose=0,meilleur_enregistre=args.meilleur_enregistre,mode='auto')

    model.compile(loss='mean_squared_error', optimizer=Adam(lr=args.learning_rate))


    
 model.fit_generator(batch_generator(args.data_dir, X_t, y_t, args.batch_size, True),args.samples_per_epoch,args.nb_epoch,max_q_size=1,validation_data=batch_generator(args.data_dir, X_v y_v, args.batch_size, False),nb_val_samples=len(X_valid),callbacks=[checkpoint],verbose=1) 
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
    parser = argparse.ArgumentParser(description='programme CNN')
    parser.add_argument('-d', help='data control',  ,   default='data')
    parser.add_argument('-t', help='test taille ',    default=0.2)
    parser.add_argument('-n', help='nombre of epochs',    default=10)
    parser.add_argument('-s', help='echantillonage per epoch',     default=20000)
    parser.add_argument('-b', help='batch taille',         default=40)
    parser.add_argument('-o', help='que le meilleur modele',   default='true')
    parser.add_argument('-l', help='learning rate',        default=1.0e-8)
    args = parser.parse_args()
    for key, value in vars(args).items():
    data = load_data(args)
    model = constuction_modele(args)
    trainer_modele(model, args, *data)
if __name__ == '__main__':
    main()
