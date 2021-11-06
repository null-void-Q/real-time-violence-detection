import cv2
import numpy as np
def loopVideo(clip,currentLength):
    i = currentLength
    j = 0 
    while(i < len(clip)):
        clip[i] = np.copy(clip[j])
        i+=1
        j+=1
    return clip    
    
def centerCrop(image,dim = 224):
    h,w = image.shape[:2]
    y = int((h - dim)/2)
    x = int((w-dim)/2)
    return image[y:(dim+y), x:(dim+x)]  


def imageResize(image, dim, inter = cv2.INTER_LINEAR):

    reDim = None
    (h, w) = image.shape[:2]

    
    if(h > w):
        r = dim / float(w)
        reDim = (dim, int(h * r))
    else:      
        r = dim / float(h)
        reDim = (int(w * r), dim)

    resized = cv2.resize(image, reDim, interpolation = inter)

    return resized

def preprocess_input(img):
    frame = imageResize(img,256)
    frame = centerCrop(frame,224)
    frame = (frame/255.)*2 - 1  
    return frame
