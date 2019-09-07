import cv2
from PIL import Image
from modelutils import getSingleEmbedding, detectFace
from os import walk, path
import pickle
from datetime import datetime
import os

embeddings_file = "embeddings.database"


def getEmbedding(path):
    return getSingleEmbedding(detectFace(Image.open(path)))


def walkThroughoutDatabase(base_path):
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
    with open(database_file, "rb") as fp:
        return pickle.load(fp)

    raise Exception("Error occured while loading file");


def appendToDatabase(img_path):
    embeddings = loadDatabase(embeddings_file)
    try:
        embeddings.append(getEmbedding(img_path))
    except:
        return False
    with open(embeddings_file, "wb") as fp:
        pickle.dump(embeddings, fp)
    return True




def saveSpoofedFrame(name, img):
    cv2.imwrite(path.join("spoofed", name + " - " + str(datetime.now()) + ".jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def saveFrame(name, img):
    cv2.imwrite(path.join("opened", name + " - " + str(datetime.now()) + ".jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
