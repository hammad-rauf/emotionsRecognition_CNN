from numpy import array,reshape

from Pickle import Pickle
from Images_func import Image_func
from Dataset import Dataset


#0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral

#### loading dataset from csv and loading in pickle

def Save_pickle():

    dataset = Dataset()
    save = dataset.read_training_dataset("train")
    label, feature = dataset.label_feature(save,"Emotion","Pixels")
    pickle = Pickle()
    pickle.save_label_pickle(label)
    pickle.save_feature_pickle(feature)



##### unloading data set from pickle 

def Load_pickle():

    pickle = Pickle()

    label =  pickle.load_label_pickle("label_pickle")
    feature = pickle.load_feature_pickle("feature_pickle")

    return feature,label


if __name__ == "__main__":
    
    #Save_pickle()
    #Load_pickle()

    pass