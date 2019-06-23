import socketio
import eventlet
import eventlet.wsgi
import argparse
import base64
from datetime import datetime
import os
import shutil
import numpy as np

from PIL import Image
from flask import Flask
from io import BytesIO
from keras.models import load_model
import utils

#initialisation du serveur
sio = socketio.Server()
app = Flask(__nom__)
model = None
prev_image_array = None

MAX_SPEED = 26
MIN_SPEED = 12

#preciser la vitesse limite
vitesse_limit = MAX_SPEED

@sio.on('mesureàdistance')
def mesureàdistance(sid, data):
    if data:

        angle_direction = float(data["angle_direction"])
        throttle = float(data["throttle"])
        vitesse = float(data["vitesse"])
        image = Image.open(BytesIO(base64.b64decode(data["image"])))
        try:
            image = np.asarray(image)
            image = utils.preprocess(image)
            image = np.array([image])
            anlge_direction = float(model.predict(image, batch_size=1))
            global vitesse_limit
            if vitesse> vitesse_limit:
                vitesse_limit = MIN_SPEED
            else:
                vitesse_limit = MAX_SPEED
            throttle = 1.0 - angle_direction**2 - (vitesse/vitesse_limit)**2

            print('{} {} {}'.format(angle_direction, throttle, vitesse))
            send_control(angle_direction, throttle)
        except Exception as e:
            print(e)
        if args.image_folder != '':
            image_filename = os.path.join(args.image_folder, timestamp)
            image.save('{}.jpg'.format(image_filename))
    else:

        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(angle_direction, throttle):
    sio.emit(
        "steer",
        data={
            'anlge_direction': angle_direction.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='conduit')
    parser.add_argument(
        'model',
        type=str,

    )
    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='',

    )
    args = parser.parse_args()

    #charger le modele
    model = load_model(args.model)

    if args.image_folder != '':
        print("Creating image folder at {}".format(args.image_folder))
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
        else:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 3000)), app)

