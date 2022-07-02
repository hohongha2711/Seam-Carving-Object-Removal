import cv2
import numpy as np 
from matplotlib import pyplot as plt

ENERGY_MASK_CONST = 10000000.0             
MASK_THRESHOLD = 10

#visualization
def visualize(img, seam, rot):
    vis = img.astype(np.uint8)
    for i in range(img.shape[0]):
        vis[i,seam[i]] = (0,0,255)
    if rot:
        vis = np.rot90(vis, 3)
    cv2.imshow("visualization", vis)
    cv2.waitKey(1)
    return vis

# Forward energy
def forward_energy(img):
    h, w = img.shape[:2]
    img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float64)

    energy = np.zeros((h, w))
    m = np.zeros((h, w))

    U = np.roll(img, 1, axis=0)
    L = np.roll(img, 1, axis=1)
    R = np.roll(img, -1, axis=1)

    cU = np.abs(R - L)
    cL = np.abs(U - L) + cU
    cR = np.abs(U - R) + cU

    for i in range(1, h):
        mU = m[i-1]
        mL = np.roll(mU, 1)
        mR = np.roll(mU, -1)
        
        mULR = np.array([mU, mL, mR])
        cULR = np.array([cU[i], cL[i], cR[i]])
        mULR += cULR

        argmins = np.argmin(mULR, axis=0)
        m[i] = np.choose(argmins, mULR)
        energy[i] = np.choose(argmins, cULR)     
        
    return energy

# Minimum seam
def get_minimum_seam(img, rmask=None):
    h, w = img.shape[:2]

    M = forward_energy(img)

    if rmask is not None:
        M[np.where(rmask > MASK_THRESHOLD)] = -ENERGY_MASK_CONST * 100

    backtrack = np.zeros_like(M, dtype='int') 

    for i in range(1, h):
        for j in range(0, w):
            if j == 0:
                idx = np.argmin(M[i - 1, j:j + 2])
                backtrack[i, j] = idx + j
                min_energy = M[i-1, idx + j]
            else:
                idx = np.argmin(M[i - 1, j - 1:j + 2])
                backtrack[i, j] = idx + j - 1
                min_energy = M[i - 1, idx + j - 1]

            M[i, j] += min_energy
    
    seam_idx = []
    boolmask = np.ones((h, w), dtype=bool)
    j = np.argmin(M[-1])

    for i in range(h-1, -1, -1):
        boolmask[i, j] = False
        seam_idx.append(j)
        j = backtrack[i, j]

    seam_idx.reverse()

    return np.array(seam_idx), boolmask

# Remove a seam
def remove_seam(img, boolmask):
    h, w = img.shape[:2]
    boolmask3c = np.stack([boolmask] * 3, axis=2)
    return img[boolmask3c].reshape((h, w - 1, 3))


def remove_seam_gsc(img, boolmask): 
    h, w = img.shape[:2]
    return img[boolmask].reshape((h, w - 1))

# Seams removal
def seams_removal(img, num_remove):
    for _ in range(num_remove):
        seam_idx, boolmask = get_minimum_seam(img)
        img = remove_seam(img, boolmask)

    return img

# Add a seam
def add_seam(img, seam_idx):
    h, w = img.shape[:2]
    output = np.zeros((h, w + 1, 3), dtype=np.uint8) 
    for row in range(h):
        col = seam_idx[row]
        for ch in range(3):
            if col == 0:
                # p
                p = np.average(img[row, col: col + 2, ch])
                output[row, col, ch] = img[row, col, ch]
                output[row, col + 1, ch] = p
                output[row, col + 1:, ch] = img[row, col:, ch]
            else:
                p = np.average(img[row, col - 1: col + 1, ch])
                output[row, : col, ch] = img[row, : col, ch]
                output[row, col, ch] = p
                output[row, col + 1:, ch] = img[row, col:, ch]

    return output


def add_seam_grayscale(img, seam_idx): # Used for binary masks   
    h, w = img.shape[:2]
    output = np.zeros((h, w + 1))
    for row in range(h):
        col = seam_idx[row]
        if col == 0:
            p = np.average(img[row, col: col + 2])
            output[row, col] = img[row, col]
            output[row, col + 1] = p
            output[row, col + 1:] = img[row, col:]
        else:
            p = np.average(img[row, col - 1: col + 1])
            output[row, : col] = img[row, : col]
            output[row, col] = p
            output[row, col + 1:] = img[row, col:]

    return output

# Seams insertion
def seams_insertion(img, num_add, horizontal):
    seams_record = []
    if horizontal:
        img = np.rot90(img, 1)
    temp_img = img.copy()

    for _ in range(num_add):
        seam_idx, boolmask = get_minimum_seam(temp_img)
        if horizontal:
            visualize(temp_img, seam_idx, rot = True)
        else:
            visualize(temp_img, seam_idx, rot = False)
        seams_record.append(seam_idx)
        temp_img = remove_seam(temp_img, boolmask)

    seams_record.reverse()

    for _ in range(num_add):
        seam = seams_record.pop()
        img = add_seam(img, seam)
        for remaining_seam in seams_record:
            remaining_seam[np.where(remaining_seam >= seam)] += 2         
    return (np.rot90(img,-1) if horizontal else img )

# ----- Object removal -----
def object_removal(img, rmask, horizontal): 
    
    img = img.astype(np.float64)
    rmask = rmask.astype(np.float64)
    output = img
    h, w = img.shape[:2]
 
    while len(np.where(rmask > MASK_THRESHOLD)[0]) > 0:
        if horizontal:
            output = np.rot90(output, 1)
            rmask = np.rot90(rmask, 1)
        seam_idx, boolmask = get_minimum_seam(output, rmask)
        if horizontal:
            visualize(output, seam_idx, rot = True)
            output = remove_seam(output, boolmask)
            rmask = remove_seam_gsc(rmask, boolmask)
            output = np.rot90(output,-1)
            rmask = np.rot90(rmask,-1)
        else:
            visualize(output, seam_idx, rot = False)
            output = remove_seam(output, boolmask)
            rmask = remove_seam_gsc(rmask, boolmask)         
        

    num_add = ((h - output.shape[0]) if horizontal else (w - output.shape[1]))
    output = seams_insertion(output, num_add, horizontal)
    return output

def create_mask(image):
    global pts; pts = []
    global x_lst; x_lst = []
    global y_lst; y_lst = []
    global img_m; img_m = image.copy()
    global mask; mask = np.zeros(img_m.shape[:2])

    def draw_mask(pts):
        pts = np.array(pts, dtype=np.int32)
        pts.reshape((-1,1,2))
        cv2.fillPoly(mask,[pts],(255))
        cv2.imshow("image",mask)
        cv2.imwrite("mask.jpg", mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def mousePoints(event, x,y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            img_1 = cv2.circle(img_m, (x,y), radius=2, color=(0, 0, 255), thickness=-1)
            cv2.imshow("original", img_1)
            pts.append((x,y))
            x_lst.append(x)
            y_lst.append(y)
        elif event == cv2.EVENT_RBUTTONDOWN:
            draw_mask(pts)

    cv2.imshow("original", img_m)
    cv2.setMouseCallback("original", mousePoints)
    cv2.waitKey(0)
    if((max(x_lst) - min(x_lst)) > (max(y_lst) - min(y_lst))):
        horizontal = True
    else:
        horizontal = False

    return mask, horizontal    

def resize(image, width):
    dim = None
    h, w = image.shape[:2]
    dim = (width, int(h * width / float(w)))
    return cv2.resize(image, dim)

img = cv2.imread('Image/input.jpg')
img = resize(img, 600)
rmask, horizontal = create_mask(img)
res = object_removal(img, rmask, horizontal=horizontal)


'''img = cv2.imread('image3.png')
img = resize(img, 600)
rmask = cv2.imread('mask2.png',0)
rmask = resize(rmask, 600)
res = object_removal(img, rmask, horizontal=False)
cv2.imwrite("remove1.png", res)'''

cv2.imwrite("Image/remove_input.png", res)
cv2.imshow('Input_image', img)
cv2.imshow('Output_Image', res)

cv2.waitKey(0)