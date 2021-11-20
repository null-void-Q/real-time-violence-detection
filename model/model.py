import numpy as np
from keras.layers import Dropout
from keras.layers import Reshape
from keras.layers import Lambda
from .i3d_inception import Inception_Inflated3d,conv3d_bn
from keras.models import Model
from keras.layers import Activation
from keras import backend as K
from model.classification import classify_clip, calculate_prediction
from collections import deque


LABEL_FILE = './txt/violence_labels.txt'
labels = [x.strip() for x in open(LABEL_FILE)]



class ViolenceModel():
    
    def __init__(self, clip_size = 64, memory = 3, threshold = 60 ,frame_dims = (224,224,3)):
        
        self.model = ViolenceModel.loadModel(numberOfClasses = len(labels), inputFrames = clip_size,frameDims= frame_dims
                                    ,withWeights= 'v_inception_i3d' , withTop= True)
        
        self.frame_dims = frame_dims
        self.threshold = threshold/100
        self.memory = memory
        self.clip_size = clip_size
        
        
        self.prediction_buffer = deque([])
        
    def classify(self, clip):
        prediction = classify_clip(self.model,clip)
        self.remember(prediction)
        label = calculate_prediction(self.prediction_buffer,labels,self.threshold)
        return label
    
    def remember(self,prediction):
                
        if len(self.prediction_buffer) == self.memory:
            self.prediction_buffer.popleft()
            
        self.prediction_buffer.append(prediction)

    def update(self,config):
        
        self.threshold = config.threshold/100
        self.memory = config.memory
        self.prediction_buffer = deque([])
        
        if self.clip_size != config.clip_size:
            self.clip_size = config.clip_size
            self.model = None
            K.clear_session()
            self.model = ViolenceModel.loadModel(numberOfClasses = len(labels), inputFrames = self.clip_size,frameDims= self.frame_dims
                                    ,withWeights= 'v_inception_i3d' , withTop= True)
        
    @staticmethod
    def loadModel(numberOfClasses,inputFrames, frameDims,withWeights = None , withTop = False):

        weights = None
        if withWeights : weights = withWeights
        model = Inception_Inflated3d(
                    include_top=withTop,
                    weights=weights,
                    input_shape=(inputFrames, *frameDims),
                    dropout_prob=0.5,
                    endpoint_logit=False,
                    classes=numberOfClasses,
                    )

        if not withTop:    
            x = model.output
            x = Dropout(0.5)(x)

            x = conv3d_bn(x,numberOfClasses, 1, 1, 1, padding='same', 
                            use_bias=True, use_activation_fn=False, use_bn=False)
            
            num_frames_remaining = int(x.shape[1])
            x = Reshape((num_frames_remaining, numberOfClasses))(x)

                    # logits (raw scores for each class)
            x = Lambda(lambda x: K.mean(x, axis=1, keepdims=False),
                            output_shape=lambda s: (s[0], s[2]))(x)

            predictions = Activation('softmax')(x)
            model = Model(model.input, predictions)

        #model._make_predict_function()
        data = np.random.rand(1,inputFrames,224,224,3)
        model.predict(data)
            
        return model