
import cv2
import numpy as np
import os
import errno
import time

def rectDistance(r1, r2):
    dx = r1[0] + r1[2] / 2 - r2[0] + r2[2] / 2
    dy = r1[1] + r1[3] / 2 - r2[1] + r2[3] / 2
    return pow(dx * dx + dy * dy, 0.5)

def getNearestFromCenter(bbox, bboxes):
    if len(bboxes) == 0:
        return np.array([0, 0, 0, 0, 0])
    min_dist = rectDistance(bbox, bboxes[0])
    nearest_index = 0
    for i in range(len(bboxes)-1):
        dist = rectDistance(bbox, bboxes[i+1])
        if dist < min_dist:
            min_dist = dist
            nearest_index = i + 1

    return bboxes[nearest_index]

def IoU(r1, r2, size):
    x = max(r1[0], r2[0])
    y = max(r1[1], r2[1])
    w = min(r1[0]+r1[2], r2[0]+r2[2]) - x
    h = min(r1[1]+r1[3], r2[1]+r2[3]) - y
    width, height = size

    if w > 0 and h > 0:
        im = np.zeros([height, width], dtype=np.uint8)
        cv2.rectangle(im, (r1[0], r1[1]), (r1[0]+r1[2], r1[1]+r1[3]), 255, -1)
        cv2.rectangle(im, (r2[0], r2[1]), (r2[0]+r2[2], r2[1]+r2[3]), 255, -1)

        inter_area = w * h
        union_area = cv2.countNonZero(im)

        return inter_area / union_area

    return 0.0

def evaluateResult(eval_mask, result, cntTN, cntTP, cntFN, cntFP, name, frame, img):

    if (eval_mask.shape[0] != result.shape[0]) & (eval_mask.shape[1] != result.shape[1]):
        print("Error! The size of the image of the evaluation and results must be equal .")
        return 0,0,0,0

    if (len(eval_mask.shape) != 2) | (len(result.shape) != 2):
        print("Error! Evaluation and results image must be in grayscale .")
        return 0,0,0,0

    _, _, stats, _ = cv2.connectedComponentsWithStats(eval_mask, 4)
    stats = stats[1:len(stats)+1]

    _, _, stats2, _ = cv2.connectedComponentsWithStats(result, 4)
    stats2 = stats2[1:len(stats2)+1]
    
    # width, height = size
    height, width = eval_mask.shape[0:2]
    color_stats = np.zeros([height, width, 3], dtype=np.uint8)
    B, G, R = cv2.split(color_stats)

    AND = cv2.bitwise_and(result, eval_mask)
    XOR = cv2.bitwise_xor(result, eval_mask)
    ANDR = cv2.bitwise_and(result, XOR)
    ANDM = cv2.bitwise_and(XOR, eval_mask)

    G = cv2.bitwise_or(G, AND)
    R = cv2.bitwise_or(R, ANDR)
    G = cv2.bitwise_or(G, ANDM)
    R = cv2.bitwise_or(R, ANDM)

    color_stats = cv2.merge([B, G, R])
    cv2.imshow('Color stats', color_stats)
    cv2.waitKey(1)

    IoUs = []
    for i in range(len(stats)):
        nearest = getNearestFromCenter(stats[i], stats2)
        VIoU = IoU(nearest, stats[i], (width,height))
        IoUs.append(VIoU)
    
    mean_IoU = round(np.mean(IoUs), 2)

    breaks_gt = len(stats)
    breaks_detected = len(stats2)

    if breaks_detected != 0 or breaks_gt != 0:
        if mean_IoU > 0.4:
            cntTP += 1
            cv2.imwrite('results/' + name + '/TP/' + str(frame) + '.png', img)
        else:
            if breaks_detected >= breaks_gt:
                cntFP += 1
                cv2.imwrite('results/' + name + '/FP/' + str(frame) + '.png', img)
            else:
                cntFN += 1
                cv2.imwrite('results/' + name + '/FN/' + str(frame) + '.png', img)
    elif breaks_detected == 0 and breaks_gt == 0:
        cntTN += 1
        cv2.imwrite('results/' + name + '/TN/' + str(frame) + '.png', img)

    return cntTN, cntTP, cntFN, cntFP

def mainEvaluate(name, size, roi, breaks, img):
    cntTN = 0
    cntTP = 0
    cntFN = 0
    cntFP = 0

    ti = time.time()

    try:
        os.mkdir('results')
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    os.makedirs('results/' + name + '/TP', exist_ok=True)
    os.makedirs('results/' + name + '/FP', exist_ok=True)
    os.makedirs('results/' + name + '/FN', exist_ok=True)
    os.makedirs('results/' + name + '/TN', exist_ok=True)
    file = open('results/' + name + '/results.txt', 'w')
    file.close()

    width0, height0 = size
    frames = 0

    pathMask = 'masks/videos/'
    cap = cv2.VideoCapture(pathMask + name + '.mp4')

    while cap.isOpened():
        ret, imMask = cap.read()

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        imMask = cv2.resize(imMask, (width0, height0))

        if roi[2] != 0 or roi[3] != 0:
            imMask = imMask[int(roi[1]):int(roi[1]+roi[3]),
                int(roi[0]):int(roi[0]+roi[2])]
            height, width = imMask.shape[0:2]

        # mask adjustment
        imMask = cv2.cvtColor(imMask, cv2.COLOR_BGR2GRAY)
        _, imMask = cv2.threshold(imMask, 200, 255, 0)
        _, labels, stats, _ = cv2.connectedComponentsWithStats(imMask, 4)
        for i in range(len(stats)):
            if i != 0:
                x, y = stats[i][0], stats[i][1]
                w, h = stats[i][2], stats[i][3]
                if x+w >= width or y+h >= height or x == 0 or y == 0:
                    imMask[labels == i] = 0

        (cntTN, cntTP, cntFN, cntFP) = evaluateResult(imMask, breaks[frames], 
            cntTN, cntTP, cntFN, cntFP, name, frames, img[frames])
        frames += 1

    file = open('results/' + name + '/results.txt', 'w')
    file.write('-Total frames: ' + str(frames) + '\n')
    file.write('-TP: ' + str(cntTP) + '\n')
    file.write('-FP: ' + str(cntFP) + '\n')
    file.write('-TN: ' + str(cntTN) + '\n')
    file.write('-FN: ' + str(cntFN) + '\n')
    file.write('-TPR: ' + str(round(cntTP/(cntTP+cntFN),2)) + '\n')
    file.write('-FPR: ' + str(round(cntFP/(cntFP+cntTN),2)) + '\n')
    file.close()

    print('Processing time: ' + str(round(time.time()-ti,2)))
