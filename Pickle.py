import pickle


class Pickle:

    def save_label_pickle(self,label): # label = list
        pickle_label = open("label_pickle","wb")
        pickle.dump(label,pickle_label)
        pickle_label.close()

    def save_feature_pickle(self,feature): # feature = list
        pickle_feature = open("feature_pickle","wb")
        pickle.dump(feature,pickle_feature)
        pickle_feature.close()


    # label_pickle
    def load_label_pickle(self,label): # label = name

        label_pickle = open(label,"rb")
        Label = pickle.load(label_pickle)
        return Label


    #feature_pickle
    def load_feature_pickle(self,feature): # label = name
        feature_pickle = open(feature,"rb")
        Feature = pickle.load(feature_pickle)
        return Feature