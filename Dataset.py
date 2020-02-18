from csv import DictReader
from numpy import array,reshape

class Dataset:

    def read_training_dataset(self,file_name): #file_name + "train"

        save = []

        with open(f'{file_name}.csv') as csv_file:
            csv_reader = DictReader(csv_file)

            for a in csv_reader:
                save.append(a)

        return save
        

    def label_feature(self,save,label,feature): # save = list,label = "emotion",feature = "pixels"

        Label = []
        Feature = []

        for dic in save:
            Label.append( dic[label] )
            matrix = list(map(int,dic[feature].split(" ")))
            matrix = reshape(matrix, (48, 48))
            Feature.append( matrix )

        return Label,Feature

