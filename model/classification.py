from .transforms import preprocess_input
import numpy as np


def classify_clip(model,clip):

    processed_clip = []
    for frame in clip:
        processed_frame = preprocess_input(frame)
        processed_clip.append(processed_frame)
    processed_clip = np.expand_dims(processed_clip,axis=0)
    predictions = model.predict(processed_clip, batch_size=len(clip), verbose=0, steps=None)
    predictions = predictions[0]
    return predictions          

def getTopNindecies(array,n):
    sorted_indices = np.argsort(array)[::-1]
    return sorted_indices[:n]
    
def calculate_prediction(predictions, class_map, threshold):
    final_prediction= np.zeros((len(class_map)))
    for pred in predictions:
        final_prediction+=pred
    final_prediction/=len(predictions)

    
    top1indices = getTopNindecies(final_prediction,1)
    index = top1indices[0]
    
    #thresholding
    if index and final_prediction[index] < threshold:
        index = 0
    if not index and ((1-final_prediction[index]) > threshold):
        index = 1
    
    result =  {'label': class_map[index], 'score':round(final_prediction[index]*100,2)} 
    
    return result
        


        