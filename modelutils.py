import imp

import cv2
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from scipy.spatial.distance import cosine, euclidean

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

mtcnn = MTCNN(device=device)

resnet = InceptionResnetV1(pretrained='casia-webface').eval().to(device)

MainModel = imp.load_source('MainModel', "model/spoof_detection.py")
spoof_detector = torch.load("model/spoof_detection.pth").float().to(device)
spoof_detector.eval()

threshold_cosine = .5
threshold_euc = .5

input_vid = []


def getMTCNN():
    return mtcnn


def getFeatureExtractor():
    return resnet


def changeDevice(device):
    global mtcnn
    global resnet
    global spoof_detector
    mtcnn = MTCNN(device=device)
    resnet = InceptionResnetV1(pretrained='casia-webface').eval().to(device)
    MainModel = imp.load_source('MainModel', "model/spoof_detection.py")
    spoof_detector = torch.load("model/spoof_detection.pth").float().to(device)
    spoof_detector.eval()


def resetFrames():
    global input_vid
    input_vid = []


def addFrame(frame):
    global input_vid
    img = cv2.resize(frame, (100, 100))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) / 255
    input_vid.append(torch.tensor(img).float())


def detectFace(img):
    return mtcnn(img)


def changeThresholds(cos=.5, euc=.7):
    global threshold_cosine, threshold_euc
    threshold_cosine = cos
    threshold_euc = euc


def getSpoofDetector():
    return spoof_detector


def getSingleEmbedding(cropped):
    return resnet(cropped.unsqueeze(0).to(device)).cpu().detach().numpy()


def getDistances(embedding1, embedding2):
    return {"cosine": cosine(embedding1, embedding2),
            "euclidian": euclidean(embedding1, embedding2)}


def checkForSpoofing(threshold=.9):
    global input_vid
    if len(input_vid) < 24:
        return False

    frame_stack = torch.stack(input_vid[-24:])
    pred = spoof_detector(frame_stack.unsqueeze(0).unsqueeze(0).to(device))
    input_vid = input_vid[-25:]
    if pred.data[0][0] > threshold:
        return True
    return False


def compareEmbeddings(embedding1, embedding2, distance_type="cos"):
    assert ((distance_type != "cos") or (
            distance_type != "euc")), 'distance_type parameter should be either "cos" or "euc"'
    distances = getDistances(embedding1, embedding2)
    if distance_type is "cos":
        return distances["cosine"] < threshold_cosine
    elif distance_type is "euc":
        return distances["euclidian"] < threshold_euc


def compareFaces(img1, img2, distance_type="cos"):
    embedding1 = resnet(img1.unsqueeze(0).to(device)).cpu().detach().numpy()
    embedding2 = resnet(img2.unsqueeze(0).to(device)).cpu().detach().numpy()

    return compareEmbeddings(embedding1, embedding2, distance_type=distance_type)


def compareWithAllEmbeddings(embeddings, embedding, distance_type="cos"):
    for elem in embeddings:
        if compareEmbeddings(embeddings[elem], embedding, distance_type=distance_type):
            name = elem
            return name

    return False

def compareFaceWithAllEmbeddings(embeddings, face, distance_type="cos"):
    embedding = getSingleEmbedding(face)

    return compareWithAllEmbeddings(embeddings, embedding, distance_type)