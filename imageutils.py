import pickle
from datetime import datetime
from os import walk, path

import cv2
from PIL import Image

from modelutils import getSingleEmbedding, detectFace

embeddings_file = "embeddings.database"


def getEmbedding(path):
    """
    Returns embeddings of a persons face in a single frame.
    :param path: Path of the frame or image.
    :return: Array of embeddings
    """
    return getSingleEmbedding(detectFace(Image.open(path)))


def walkThroughoutDatabase(base_path):
    """
    Get embeddings for all of the images in base_path path and create an embedding database from it.
    :param base_path: Base path which consists the images to create database.
    :return: Number of created embeddings from dataset.
    """
    embeddings = {}
    for (dirpath, dirnames, filenames) in walk(base_path):
        for file in filenames:
            try:
                embeddings.update({path.splitext(file)[0]: getEmbedding(path.join(base_path, file))})
            except:
                continue
    with open(embeddings_file, "wb") as fp:
        pickle.dump(embeddings, fp)

    return len(embeddings)


def loadDatabase(database_file=embeddings_file):
    """
    Loads the created embeddings database.
    :param database_file: (Default: embeddings_file) The absolute path of the database file.
    :return: Nothing
    """
    with open(database_file, "rb") as fp:
        return pickle.load(fp)

    raise Exception("Error occured while loading file");


def appendToDatabase(img_path):
    """
    Appends one single face embedding array to the existing database file.
    :param img_path: Image path
    :return: Boolean (If successful)
    """
    embeddings = loadDatabase(embeddings_file)
    try:
        embeddings.append(getEmbedding(img_path))
    except:
        return False
    with open(embeddings_file, "wb") as fp:
        pickle.dump(embeddings, fp)
    return True




def saveSpoofedFrame(name, img):
    """
    Saves the current frame if spoofing is detected.
    :param name: Face used for spoofing.
    :param img: The processed frame.
    :return: Nothing
    """
    cv2.imwrite(path.join("spoofed", name + " - " + str(datetime.now()) + ".jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def saveFrame(name, img):
    """
    Saves the current frame
    :param name: Name of the human
    :param img: The current image
    :return: Nothing
    """
    cv2.imwrite(path.join("opened", name + " - " + str(datetime.now()) + ".jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
