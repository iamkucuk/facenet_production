import cv2
from datetime import datetime

from PIL import Image

from imageutils import loadDatabase, walkThroughoutDatabase, saveSpoofedFrame, saveFrame
from modelutils import detectFace, resetFrames, addFrame, checkForSpoofing, compareFaceWithAllEmbeddings

embeddings = {}
missingCount = 0
foundFaces = {}
spoofedFaces = {}


def loadEmbeddings(dataset="dataset/"):  # , device = 'cpu', threshold_cosine = .5, threshold_euc = .7):
    global embeddings
    isDatasetLoaded = False
    try:
        embeddings = loadDatabase()
        isDatasetLoaded = True
    except FileNotFoundError:
        walkThroughoutDatabase(dataset)

    if not isDatasetLoaded:
        embeddings = loadDatabase()

    return embeddings


def processByFrame(frame):
    global missingCount, embeddings

    if len(embeddings) == 0:
        embeddings = loadEmbeddings()
        print(embeddings)

    pil_image = Image.fromarray(frame)
    detected_face = detectFace(pil_image)

    if detected_face is None:
        missingCount = missingCount + 1

        if missingCount >= 24:
            resetFrames()
            missingCount = 0

        return

    addFrame(frame)
    isSpoofingChecked = checkForSpoofing()

    if not isSpoofingChecked:
        raise NameError("Spoofing Detected!")

    ret = compareFaceWithAllEmbeddings(embeddings, detected_face)

    return ret


def processVideo(video_path):
    cap = cv2.VideoCapture(video_path)
    while True:
        _, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processByFrame(frame)


def processStream(stream_path):
    cap = cv2.VideoCapture(stream_path)
    openCounter = 0
    spoofCounter = 0
    foundList = []
    prev_ret = 0
    while True:
        _, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            ret = processByFrame(frame)
            print(ret)
        except NameError:
            # spoofedFaces.update({spoofed_face, datetime.now()})
            if spoofCounter > 30:
                pil_image = Image.fromarray(frame)
                detected_face = detectFace(pil_image)
                spoofed_face = compareFaceWithAllEmbeddings(embeddings, detected_face, distance_type="euc")
                saveSpoofedFrame(spoofed_face, frame)
                print("Spoofing Detected!")
                spoofCounter = 0
            spoofCounter = spoofCounter + 1
            openCounter = 0
            prev_ret = 0
            continue

        spoofCounter = 0

        if ret == prev_ret:
            openCounter = openCounter + 1

            if openCounter > 24:
                saveFrame(ret, frame)
                # openDoor()
                openCounter = 0
                print("Door is now opened for " + ret)

                # Wait till door is closed.

        else:
            openCounter = 0

        prev_ret = ret