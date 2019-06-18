# conduite-autonome-d-une-voiture-CNN-
l’idée c'est  de construire une intelligence artificielle (un modèle) le former  et après on va tester ce modèle  afin  de simuler une voiture dans le mode autonome d'AUDACITY
# 1)	Description des fichiers et répertoires de la repository
dans la repository on trouve le fichier preproccessing-augmentation-image dans lequel on trouve  les differentes manipulations qu"on a fait afin de generer une image augmenté à la fin avec une angle de braquage bien ajustée et on trouve le  fichier python construction-formation-modele.py dans lequel on trouve  le code du construction du modele  basé  sur les reseaux de neurones convolutifs  et on trouve comment  on forme le modele afin de l'utiliser apres dans la simulation .
on trouve aussi le fichier python test du modele dans lequel on va charger le modele et on va etablir  une communication entre le serveur (le simulateur) et  les clients (les scripts construction-formation-modele et test du modele .
on trouve aussi dans la repository  le fichier read.md qui explique comment on demarre la simulation , aussi  precise  les algorithmes les bibliotheques utilisées au cours du projet et bien sur on trouve aussi la description des fichiers crees et le repertoire de la repository .

preproccessing-augmentation-image.py

construction-formation-modele.py

test du modele .py

README.md

driving_log.csv



# 2) Comment démarrer  la simulation ?

L’idée ici c’est d’exécuter le modèle pré-entrainé, donc on démarre le simulateur de conduite automatique audacity et on fait le choix d’une  scène et on  appuie  sur le bouton Mode autonome. Ensuite, on  exécute le modèle comme suit:


Python  drive.py model.h5


Avec model.h5 c’est le fichier qui contient   notre  réseau de neurones de convolution formés.
On a besoin bien sur du dossier qui contient les images d’apprentissage 

Python model.py 


 et cela générera un fichier à model-<epoch>.h5
Le script drive.py c’est le script de test pour conduire la voiture en mode autonome   qui représente le client et le  simulateur représente  le serveur et ce script  prend en charge un flux constant d'images, les manipule (redimensionnement et découpage) dans la forme d'entrée du modèle, puis transmet la matrice d'images transformées au modèle, qui fait la prédiction d’un angle de braquage approprié en fonction de l'image. L'angle de braquage est ensuite transmis à la voiture en tant que commande et la voiture se guide en conséquence. La voiture  autonome parcourt ainsi le parcours en émettant constamment des images et en recevant les angles de braquage. On espère bien sur que le modèle aura été suffisamment entraîné pour que les angles de direction qu’il reçoit permettent au véhicule de rouler en toute sécurité au milieu de la voie et de ne pas dériver sur la route ou faire autre chose qui serait considéré dangereux.





# 3)ALGORITHMES et bibliothèques utilisées
# 	Algorithmes 
# preprocessing   des images :  
afin de pouvoir conserver simultanément plus d’images en mémoire et d’accélérer la formation, qui serait extrêmement lente sur les images de taille normale, on reduire la taille de l'image .
On Recadre  l'image dans le sens on enleve le ciel en haut et l'avant de la voiture en bas  
on fait la conversion RGB  to  YUV  .
on combine le redimensionnement + recadrement+conversion dans une seule fonction 
# augmentation  des images : 
on Bascule l'image au hasard à gauche et à droite et on  ajuste l'angle de braquage et on décale  l'image horizontalement et verticalement  parceque on ne veut pas avoir des angles de braquages negatifs.

# Génération d’une image augmentée 


# Algorithme principal :
On va créer un réseau de neurones  convolutionel CNN inspiré de NVIDIA (par regression) qui lira les données  puis donne une sortie   qui va être la commande de pilotage. Le conducteur conduit et Machine va cloner  ce comportement  et on appelle ce processus : clonage comportemental.
Dans la phase de formation : (script de formation)  (model.py) 
La 1ere étape c’est de  charger le fichier.csv  
On fait entrer  les entrées X  et  Y va représenter  les données de sortie  qui seront notre commande de pilotage .On va essayer de  trouver la correspondance entre les deux (X et Y) c’est un apprentissage supervisé =>  lorsque on trouve  la correspondance entre les deux =>  on peut sortir le  prédictive output label après on peut  scinder les données de formation et de test 80 pourcent training et 20 pourcent de test  et puis nous avons le code pour construire notre modèle et ensuite la formation du modèle.
# Algorithme  de construction du modèle : 
On  veut que notre première couche soit une couche de normalisation d’image en utilisant la fonction lambda pour  éviter les saturations et  améliorer les fonctionnements du gradient parce-que les images  peuvent être un peu  vague aussi les valeurs de correction de couleurs peuvent être fausses et donnent des résultats fausses.
	On veut  dire  formater ou rémoduler ces valeurs  de tenseurs d’image en valeurs qui nous donnent  des bonnes prévisions 
Maintenant on va faire des couches convolutifs : la premier va avoir un filtre de taille 24, une convolution 5by5 après on met une fonction d’activation ELU : unités linéaire exponentielle et nous allons sous échantillonner deux par deux parce que c’est la longueur de nos pas => elle s’occupe  du problème du gradient en voie de disparition => on va faire  5 couches. 
Apres on va ajouter une couche de suppression  (dropout)  on va  ‘surfacer’  parce que on va  commencer à alimenter une série de couches  entièrement connectes aussi  les couches convolutifs  sont conçues pour générer l’ingénierie de caractéristiques (feature ingenieering).
Chacun des couches va créer des filtres  de plus en plus abstraites , ils commenceront  par utiliser des fonctionnalités  de bas niveau  donc ils vont pouvoir détecter des images mais  à la sortie  on veut pas une image , on veut à la fin une valeur et une seule valeur  correspond  à la commande du pilotage  c’est une direction => comment on va déplacer cette roue afin d’obtenir une valeur unique à partir de ces tenseurs d’images de grandes dimensions => on doit écraser ces données  et pour faire  ça on applique une série de couche entièrement connectés et   chaque  couche  va être  au fur et à mesure plus petit   en termes de  nombre de neurones .La dernière couche  sans  nombre de neurones car c’est la dernière couche  qui va sortir notre conduite ( valeur de l’angle )
Et à la fin on retourne le modèle  donc ca va être comme un triangle  en termes de nombres,
 les matrices qui propagent dans notre réseau matrice avec  indice 10 après   avec un indice  5 après une matrice avec  2 indices après une   matrice avec un indice =>  c’est notre output.
 Le black box  de neural network   on ne connait pas ou  une feature commence  et  ou une autre se termine  de son emplacement  dans cette abstraction mais on sait qu’il ya une certaine  connectivité.
# Algorithme de Formation du modèle : 
On définit  le modèle  et on l’enregistre  dans un check point il va  être modélisé donc on va dire mode automatique après on veut dire qu’on veut enregistrer le meilleur modèle.
Aussi  on ajoute la fonction du quadratique  erreur  qui va  donner une prédictive angle de braquage et puis on a une réelle   angle de braquage du simulateur non-autonome et on veut trouver une différence entre les deux  donc  on mets  la différence aux carrés et  et puis on somme  les différences après on divise par leur nombre on utilise  après l’optimiseur d’atomes  qui est la descente du gradient  après la compilation on peut générer des données. On utilise le générateur d’ajustement real time  data argumentation sur les images sur le cpu en parallèle.
=> On génère des lots de données à partir de nos données de formation.
# Algorithme du test : 
Il s’agit du principe  serveur-client, ça veut dire que le simulateur c’est le serveur et les clients  sont les scripts que nous avons écrit.
# INTIALISATION : 
On commence par initialiser notre serveur, il s’agira d’un serveur de type  io et on utilise flask  pour faire ça on initialise notre modèle et le taux d’image aussi  nous allons définir  une vitesse maximale et minimale pour notre voiture autonome 10 miles et   25 miles à l’heure et on définit une limite de vitesse 
# MAIN FONCTION : 
On charge le modèle et  on indique ou se trouve .Le middelware qui va permettre à notre client de communiquer avec le serveur après nous allons déployer le serveur wsGi . on ne  va pas coder la partie serveur  et on va coder la partie client  il ya d’autres façon mais c’est la façon la plus simple  , 
# FONCTION TELEMETRY : 
Faire la prédiction et ensuite l’envoyer pour savoir la valeur de l’angle  de braquage  puis on envoie au serveur nous allons donc intégrer  ça dans nos données.
=> on veut ces valeurs (throttle, angle de braquage)  pour les manipuler et les transformer en une valeur scalaire qui indiquera à notre voiture ou aller .
Maintenant  on va effectuer de tenseurs  sur cette image nous devons donc insérer cette image dans notre réseau  pour la faire convertir l’image en un tableau on applique le preproccing on prédite l’angle de braquage a partir de notre modèle  si la vitesse est maximale on la change  et elle devient  une vitesse minimale  donc ca veut dire on veut ralentir et  si c’est pas le cas on va dire ici qu’on veut pas dépasser la vitesse maximale.
Apres on peut envoyer le contrôle  en utilisant la fonction send_control  on a preprocessé l’image , on  a introduit dans notre modèle,  on trouve à la sortie  the steering angle et nous pouvons utiliser après pour envoyer ce contrôle directement au serveur  via la fonction send_control c’est comme un paquet  pour le serveur et le serveur lira cette information de manière a ce que la logique de gestion des événements soit sous le capot du simulateur et nous pouvons simplement l’envoyer via notre client .

# Bibliothèques

Avant de  commencer la programmation, on  a installé  l’environnement anaconda  et aussi  l’éditeur de texte Atom. Au lieu  d’installer manuellement les bibliothèques requises à l’aide de pip on a choisi de lancer cette commande dans l’anaconda Prompt et de cette façon on peut installer toutes les dépendances en une seule ligne du code   : 
conda env create –f  environments.yml
Les bibliothèques utilisées et leur role : 

![alt text](https://github.com/oussema95/conduite-autonome-d-une-voiture-CNN-/blob/master/rapport1.PNG)
![alt text](https://github.com/oussema95/conduite-autonome-d-une-voiture-CNN-/blob/master/rapport%202.PNG)




