from PIL import Image
from numpy import uint8
import matplotlib.pyplot as plt

class Image_func:

    def Make_Image(self,image_array,name): #convert pixels in grey scale image

        img = Image.fromarray(image_array.astype(uint8))
        img.save(f"{name}.png")
        img.show()

    def Show_Image(self,image_array): #convert pixels in grey scale image
                
        img = Image.fromarray(image_array.astype(uint8))
        img.show()

    def plot_image(self,feature,label): #Plot pixels in grey scale image

        label = ["Angry","Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

        plt.title(label[int(label)])
        plt.imshow(image_array,cmap="gray")
        plt.show()
