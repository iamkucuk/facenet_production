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
    """
    Gets an instance of MTCNN
    :return: MTCNN pipeline
    """
    return mtcnn


def getFeatureExtractor():
    """
    Gets the feature extractor network
    :return: InceptionResnetV1 instance
    """
    return resnet


def changeDevice(device):
    """
    Changes the whole pipeline's device.

    Important notice: The running device of the pipeline will be selected automatically among existing options. The code
     will choose its best option.
    :param device: (String) either 'cpu' or a cuda device name.
    :return: Nothing
    """
    global mtcnn
    global resnet
    global spoof_detector
    mtcnn = MTCNN(device=device)
    resnet = InceptionResnetV1(pretrained='casia-webface').eval().to(device)
    MainModel = imp.load_source('MainModel', "model/spoof_detection.py")
    spoof_detector = torch.load("model/spoof_detection.pth").float().to(device)
    spoof_detector.eval()


def resetFrames():
    """
    Resets the frame list.
    :return: Nothing
    """
    global input_vid
    input_vid = []


def addFrame(frame):
    """
    Adds a processed frame to frame list for spoofing detection.
    :param frame: Current frame in RGB format.
    :return: Nothing
    """
    global input_vid
    img = cv2.resize(frame, (100, 100))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) / 255
    input_vid.append(torch.tensor(img).float())


def detectFace(img):
    """
    Detects faces and aligns it in a frame.
    :param img: Current frame
    :return: Cropped and aligned face image.
    """
    return mtcnn(img)


def changeThresholds(cos=.5, euc=.7):
    """
    Tunes the threshold used for comparison. Choosing smaller threshold values leads to stricter decisions for recognition pipeline.
    Larger values lead to more loose decisions.
    :param cos: Threshold for cosine similarity method.
    :param euc: Threshold for euclidian distance method
    :return: Nothing
    """
    global threshold_cosine, threshold_euc
    threshold_cosine = cos
    threshold_euc = euc


def getSpoofDetector():
    """
    Returns the spoof detector network
    :return: Spoof Detector Network
    """
    return spoof_detector


def getSingleEmbedding(cropped):
    """
    Gets embedding of a single face image
    :param cropped: Aligned face image
    :return: Embeddings
    """
    return resnet(cropped.unsqueeze(0).to(device)).cpu().detach().numpy()


def getDistances(embedding1, embedding2):
    """
    Calculates similarities for both of methods
    :param embedding1: Embedding of a face
    :param embedding2: Other embedding of a face
    :return: (Dict) Calculated distances.
    """
    return {"cosine": cosine(embedding1, embedding2),
            "euclidian": euclidean(embedding1, embedding2)}


def checkForSpoofing(threshold=.9):
    """
    Spoofing detection pipeline.
    :param threshold: Threshold for spoofing decisions. (Closer to 1: making strict rules, closer to 0: making loose rules)
    :return: (Boolean) True if frame passes the test. False otherwise.
    """
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
    """
    Compares embeddings according to given distance type.
    :param embedding1:
    :param embedding2:
    :param distance_type: (String) It should be either 'cos' or 'euc'.
    :return: Comparison of distances according to determined threshold for desired distance type.
    """
    assert ((distance_type != "cos") or (
            distance_type != "euc")), 'distance_type parameter should be either "cos" or "euc"'
    distances = getDistances(embedding1, embedding2)
    if distance_type is "cos":
        return distances["cosine"] < threshold_cosine
    elif distance_type is "euc":
        return distances["euclidian"] < threshold_euc


def compareFaces(img1, img2, distance_type="cos"):
    """
    Compares images according to given distance type.
    :param img1:
    :param img2:
    :param distance_type: (String) It should be either 'cos' or 'euc'.
    :return: Comparison of distances according to determined threshold for desired distance type.
    """
    embedding1 = resnet(img1.unsqueeze(0).to(device)).cpu().detach().numpy()
    embedding2 = resnet(img2.unsqueeze(0).to(device)).cpu().detach().numpy()

    return compareEmbeddings(embedding1, embedding2, distance_type=distance_type)


def compareWithAllEmbeddings(embeddings, embedding, distance_type="cos"):
    """
    Compares extracted embedding with a set of embeddings according to given distance type
    :param embeddings: (Dict) A set of embeddings.
    :param embedding: Embedding for a single face
    :param distance_type: (String) It should be either 'cos' or 'euc'.
    :return: Found name. If none of the faces match, returns False.
    """
    for elem in embeddings:
        if compareEmbeddings(embeddings[elem], embedding, distance_type=distance_type):
            name = elem
            return name

    return False


def compareFaceWithAllEmbeddings(embeddings, face, distance_type="cos"):
    """
    Compares the given aligned face image with a set of embeddings according to given distance type.
    :param embeddings: (Dict) A set of embeddings
    :param face: Cropped and aligned face image
    :param distance_type: (String) It should be either 'cos' or 'euc'.
    :return: Found name. If none of the faces match, returns False.
    """
    embedding = getSingleEmbedding(face)

    return compareWithAllEmbeddings(embeddings, embedding, distance_type)
