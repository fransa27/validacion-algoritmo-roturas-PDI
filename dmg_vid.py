import cv2
import numpy as np
from evaluation import mainEvaluate

candidates = []
nAlpha = 6
nNeighbors = 8
breakID = 0
seqBreaks = []
results = []
width0, height0 = (640, 480)
width, height = width0, height0
is_select = False

path = 'videos/'
name = 'video1'
cap = cv2.VideoCapture(path + name + '.mp4')
while cap.isOpened():
    ret, img = cap.read()

    # If frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Image resize, estandard resolution
    img = cv2.resize(img, (width0, height0))

    # Select ROI
    if is_select == False:
        roi = cv2.selectROI(img)
        cv2.destroyAllWindows()
        is_select = True
    if roi[2] != 0 or roi[3] != 0:
        img = img[int(roi[1]):int(roi[1] + roi[3]),
              int(roi[0]):int(roi[0] + roi[2])]
        height, width = img.shape[0:2]

    # Space-color conversion and channels separation
    # to obtain a grayscale image
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    l, _, _ = cv2.split(lab)

    # Net or background segmentation
    th = cv2.adaptiveThreshold(l, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY, 75, 0)

    nWhitePixels = cv2.countNonZero(th)
    nPixels = height * width
    nBlackPixels = nPixels - nWhitePixels
    ratioPixels = nBlackPixels / nWhitePixels
    if ratioPixels < 1:
        th = cv2.bitwise_not(th)

    # Search for connected components
    _, labels, stats, _ = cv2.connectedComponentsWithStats(th, 4)

    # Find connected component of larger area (net)
    maxArea = 0
    maxAreaIdx = 0
    for idx, info in enumerate(stats):
        if idx == 0:
            continue
        area = info[4]
        if area > maxArea:
            maxAreaIdx = idx
            maxArea = area
    mask = np.zeros([height, width], dtype=np.uint8)
    mask[labels == maxAreaIdx] = 255

    # Morphological closing operation
    # to noise reduction
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel2, iterations=2)
    mask = cv2.erode(mask, kernel, iterations=1)

    cv2.imshow('Net', mask)

    # Get all holes in the net
    mask = cv2.bitwise_not(mask)
    _, labels, stats, cts = cv2.connectedComponentsWithStats(mask)

    # Discard all incomplete holes at
    # the edge of the image and
    areas = []
    holeStats = []
    holeCts = []
    # imHoles = np.zeros([height, width, 3], dtype=np.uint8)
    for idx, info in enumerate(stats):
        if idx == 0:
            continue
        area = info[4]
        x, y, w, h = info[0:4]
        if x + w < width and y + h < height and x > 0 and y > 0:
            areas.append(area)
            holeStats.append(info)
            holeCts.append(cts[idx])
            # imHoles[labels==idx] = 255

    # Mean and standard deviation of the
    # areas of the holes under consideration
    if len(areas) != 0:
        meanAreas = np.mean(areas)
        stdAreas = np.std(areas)

    # Outlier detection by normal distribution
    # with exponential penalty
    candidates0 = []
    z = 3 * (1 - np.exp(-0.1 * len(areas)))
    for idx0, area0 in enumerate(areas):
        if area0 > meanAreas + z * stdAreas:
            x0, y0, w0, h0 = holeStats[idx0][0:4]
            ct0 = np.array([holeCts[idx0][0], holeCts[idx0][1]], int)

            # drawing centroids
            cv2.circle(img, ct0, 3, (0, 0, 255), cv2.FILLED, 4)

            # nearest neighbor search
            neighborsArea = []
            neighbors = []
            distNeighbors = []
            for idx, area in enumerate(areas):
                x, y, w, h = holeStats[idx][0:4]
                ct = np.array([holeCts[idx][0], holeCts[idx][1]], int)
                dist_cts = np.linalg.norm(ct - ct0)
                distNeighbors.append(dist_cts)

            if len(distNeighbors) < nNeighbors + 1:
                continue

            distNeighborsSort = distNeighbors.copy()
            distNeighborsSort.sort()
            distNeighborsSort = distNeighborsSort[1:nNeighbors + 1]

            for n in range(nNeighbors):
                neighborIdx = distNeighbors.index(distNeighborsSort[n])
                neighborsArea.append(areas[neighborIdx])
                xN, yN, wN, hN = holeStats[neighborIdx][0:4]
                ctN = np.array([(xN + xN + wN) / 2, (yN + yN + hN) / 2], int)
                cv2.circle(img, ctN, 3, (255, 0, 0), cv2.FILLED, 4)

            # Only the nearest neighbor with the
            # largest area is retained
            maxNeighbor = np.max(neighborsArea)
            candidates0.append([ct0, (x0, y0), 0, 0, area0, maxNeighbor])

    # breaks = np.zeros([height, width], dtype=np.uint8)
    if candidates:
        for candidate in candidates:
            im2 = np.zeros([height, width, 3], dtype=np.uint8)
            ct = candidate[0]
            area = candidate[4]
            alpha = candidate[2]

            for candidate0 in candidates0:
                ct0 = candidate0[0]
                area0 = candidate0[4]
                dist = np.linalg.norm(np.array(ct) - np.array(ct0))
                areaN = candidate0[5]
                x, y = candidate0[1]
                ct0 = candidate0[0]

                # conditions of spatio-temporal coherence
                # and area for first detection
                if alpha < nAlpha and dist < 10 and area0 > 2.5 * areaN:
                    candidate[2] += 1
                    candidate0[2] = candidate[2]
                    candidate0[3] = candidate[3]

                    if candidate0[2] == nAlpha:
                        breakID += 1
                        candidate0[3] = breakID
                        cv2.rectangle(im2, (x, y), (x + 2 * (ct0[0] - x), y + 2 * (ct0[1] - y)),
                                      (0, 255, 0), cv2.FILLED, 4)
                        rect1 = im2[y:y + 2 * (ct0[1] - y), x:x + 2 * (ct0[0] - x)]
                        rect2 = img[y:y + 2 * (ct0[1] - y), x:x + 2 * (ct0[0] - x)]
                        rect = cv2.addWeighted(rect1, 0.3, rect2, 0.7, 0)
                        img[y:y + 2 * (ct0[1] - y), x:x + 2 * (ct0[0] - x)] = rect
                        cv2.putText(img, ' id%s' % candidate0[3], ct0,
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, 8)
                        # Print every time if the brekage has been confirmed
                        print("Daño", str(breakID))
                        # Store in a local folder all images of brekages.
                        cv2.imwrite('pruebas_video/pv1/img' + str(breakID) + '.jpg', img)
                        # breaks[labels == candidate0[6]] = 255

                # if a breakage has been confirmed,
                # relax distance condition
                elif alpha >= nAlpha and dist < 150 and area0 > 3 * areaN:
                    candidate[2] += 1
                    candidate0[2] = candidate[2]
                    candidate0[3] = candidate[3]
                    cv2.rectangle(im2, (x, y), (x + 2 * (ct0[0] - x), y + 2 * (ct0[1] - y)),
                                  (0, 255, 0), cv2.FILLED, 4)
                    rect1 = im2[y:y + 2 * (ct0[1] - y), x:x + 2 * (ct0[0] - x)]
                    rect2 = img[y:y + 2 * (ct0[1] - y), x:x + 2 * (ct0[0] - x)]
                    rect = cv2.addWeighted(rect1, 0.3, rect2, 0.7, 0)
                    img[y:y + 2 * (ct0[1] - y), x:x + 2 * (ct0[0] - x)] = rect
                    cv2.putText(img, ' id%s' % candidate0[3], ct0,
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, 8)
                    # breaks[labels == candidate0[6]] = 255

    candidates = candidates0
    # seqBreaks.append(breaks)
    # results.append(img)

    cv2.imshow('Detection', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# mainEvaluate(name, (width0,height0), roi, seqBreaks, results)