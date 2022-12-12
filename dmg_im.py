import cv2
import sys
import numpy as np

nNeighbors = 8
breakID = 0

# Read image
path = 'data/images/'
name = 'net1.png'
img = cv2.imread(path + name)

if img is None:
    sys.exit('Could not read the image')

# Image resize, estandard resolution
width, height = (640, 480)
img = cv2.resize(img, (width, height))

# Select ROI
roi = cv2.selectROI(img)
cv2.destroyAllWindows()
if roi[2] != 0 and roi[3] != 0:
    img = img[int(roi[1]):int(roi[1] + roi[3]),
          int(roi[0]):int(roi[0] + roi[2])]
    height, width = img.shape[0:2]

# Space-color conversion and channels separation
lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
l, a, b = cv2.split(lab)

# Image filtering
# l = cv2.GaussianBlur(l, (15,15), 0)

# Segmentation with adaptive theshold with THRESH GAUSSIAN

th = cv2.adaptiveThreshold(l, 255,
	cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 85, 0)  

#th = cv2.adaptiveThreshold(l, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
#                           cv2.THRESH_BINARY, 85, 0)

 

# _,th2 = cv2.threshold(l,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imshow('Th', th)
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
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
cv2.imshow('Net', mask)
# cv2.imwrite('segmentation.png', mask)

# Reverse segmentation to obtain holes
mask = cv2.bitwise_not(mask)

# New search for connected components (holes)
_, labels, stats, cts = cv2.connectedComponentsWithStats(mask, 4)

# Discard all incomplete holes at the edge
# of the image
areas = []
holeStats = []
holeCts = []
imHoles = np.zeros([height, width, 3], dtype=np.uint8)
for idx, info in enumerate(stats):
    if idx == 0:
        continue
    area = info[4]
    x, y, w, h = info[0:4]
    if x + w < width and y + h < height and x > 0 and y > 0:
        areas.append(area)
        holeStats.append(info)
        holeCts.append(cts[idx])
        imHoles[labels == idx] = 255

# mean and standard deviation of holes found
if len(areas) != 0:
    meanAreas = np.mean(areas)
    stdAreas = np.std(areas)

# find all the holes that have an area
# greater than average
candidates0 = []
im2 = np.zeros([height, width, 3], dtype=np.uint8)
z = 3 * (1 - np.exp(-0.1 * len(areas)))
neighborsAreasAll=[]
for idx0, area0 in enumerate(areas):
    if area0 > 1.5*meanAreas:# + z * stdAreas
        x0, y0, w0, h0 = holeStats[idx0][0:4]
        ct0 = np.array([holeCts[idx0][0], holeCts[idx0][1]], int)

        # drawing centroids
        cv2.circle(img, ct0, 3, (0, 0, 255), cv2.FILLED, 4)

        # nearest neighbor search
        neighborsArea = []
        neighbors = []
        distNeighbors = []
        for idx, area in enumerate(areas):
            if idx == idx0:
                continue
            x, y, w, h = holeStats[idx][0:4]
            ct = np.array([(x + x + w) / 2, (y + y + h) / 2], int)
            dist_cts = np.linalg.norm(ct - ct0)
            distNeighbors.append(dist_cts)

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
        neighborsAreasAll.append(neighborsArea)
        maxNeighbor = np.max(neighborsArea)
        candidates0.append([ct0, (x0, y0), 0, 0, area0, maxNeighbor])

print(neighborsAreasAll)

for indx0, candidate0 in enumerate(candidates0):
    meanNeighborsArea = np.mean(neighborsAreasAll[indx0])
    x, y = candidate0[1]
    ct0 = candidate0[0]
    area0 = candidate0[4]
    areaN = candidate0[5]

    # conditions of area for detection

    if area0 > 2 * meanNeighborsArea:    # 2.5 * meanNeighborsArea #2.5 * areaN
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

cv2.imshow('Detection', img)
cv2.imshow('Holes', imHoles)
cv2.imwrite('Detection2.png',img)
cv2.waitKey(0)

# cv2.imwrite('results.png', img)
