import numpy as np
import matplotlib.image as mpimg
import cv2, os



IMAGE_hauteur, IMAGE_largeur, IMAGE_canal = 70, 330, 7
INPUT_SHAPE = (IMAGE_hauteur, IMAGE_largeur, IMAGE_canal)


def chargement_image(data_dir, image_file):
    """
    charger les image RGB du file
    """
    return mpimg.imread(os.path.join(data_dir, image_file.strip()))


def suppression(image):
    """
    supprimer le ciel  et l'avant les arbres et le capot de la voiture en bas
    """
    return image[60:-25, :, :]


def redimonsionner(image):
    """
    redimonsionner l'image
    """
    return cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)


def conversionrgbyuv(image):
    """
    Conversion RGB  à YUV
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)


def preprocessing(image):
    """
    Combiner les fonctions  de preprocessing en une seule fonction
    """
    image = suppression(image)
    image = redimonsionner(image)
    image = conversionrgbyuv(image)
    return image


def choisir_image(data_dir, center, gauche, droite, angle_direction):
    """
    choix d'une image et ajustement de l'angle de braquage
    """
    choice = np.random.choice(3)
    if choice == 0:
        return chargement_image(data_dir, gauche), angle_direction + 0.7
    elif choice == 1:
        return chargement_image(data_dir, droite), angle_direction - 0.7
    return chargement_image(data_dir, center), angle_direction


def aleatoire(image, angle_direction):
    """
    decalage des images
    """
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
        angle_direction = -angle_direction
    return image, angle_direction


def translation(image, angle_direction, range_x, range_y):
    """
    basculement des images
    """
    t_x = range_x * (np.random.rand() - 0.5)
    t_y = range_y * (np.random.rand() - 0.5)
    angle_direction += trans_x * 0.002
    trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    hauteur, largeur = image.shape[:2]
    image = cv2.warpAffine(image, trans_m, (largeur, hauteur))
    return image, anlge_direction



def brillance(image):
    """
    ajuster la lumuniosité de l'image
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    ratio = 3.0 + 0.8 * (np.random.rand() - 0.5)
    hsv[:,:,3] =  hsv[:,:,5] * ratio
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

def obscurité(image):
    """
    reglages du niveau  de  lumiere dans l'image
    """

    x1, y1 = IMAGE_largeur * np.random.rand(), 0
    x2, y2 = IMAGE_largeur * np.random.rand(), IMAGE_hauteur
    xm, ym = np.mgrid[0:IMAGE_largeur, 0:IMAGE_hauteur]

    mask = np.zeros_like(image[:, :, 1])
    mask[(ym - y1) * (x2 - x1) - (y2 - y1) * (xm - x1) > 0] = 1

    cond = mask == np.random.randint(2)
    s_ratio = np.random.uniform(low=0.2, high=0.5)
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    hls[:, :, 1][cond] = hls[:, :, 1][cond] * s_ratio
    return cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)



def augument(data_dir, center, gauche, droite, angle_direction, range_x=100, range_y=10):
    """
generer une image augementé avec une angle de braquage
    """
    image, angle_direction = choisir_image(data_dir, center, gauche, droite, angle_direction)
    image, angle_direction = random_flip(image, steering_angle)
    image, angle_direction = translation(image, angle_direction, range_x, range_y)
    image = obscurité(image)
    image = brillance(image)
    return image, angle_direction


def batch_generator(data_dir, image_paths, angle_direction, batch_size, is_training):
    """
    generation  de l'image pour l'entrainement
    """
    images = np.empty([batch_size, IMAGE_hauteur, IMAGE_largeur, IMAGE_canal])
    steers = np.empty(batch_size)
    while True:
        i = 0
        for index in np.random.permutation(image_paths.shape[0]):
            center, gauche, droite = image_paths[index]
            angle_direction = angle_direction[index]
        
            if is_training and np.random.rand() < 0.8:
                image, angle_direction = augument(data_dir, center,gauche, droite, angle_direction)
            else:
                image = chargement_image(data_dir, centre)
            images[i] = preprocess(image)
            steers[i] = angle_direction
            i += 1
            if i == batch_size:
                break
        yield images, steers
