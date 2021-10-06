# -*- coding: utf-8 -*-
"""
@author: Zeel Mehta 2020csm1021
"""

import skimage
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
from skimage import io
from scipy.spatial import distance as dist
import random
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
import warnings



# Global Variables
dict_images     = {}   # corresponds to the dictionary which store images as values and imageName as keys.


warnings.filterwarnings("ignore")            # ignoring filter warnings


'''
resizeImageForCollage(sourceFolderPath)
    Input: 1.'sourceFolderPath' refers to path of folder which contains images.
           2. 'Image Name' refers to entry of dictionary,  imagename[0] refers to the key which is Image name, that needs to be resized.
    Output: returns the resized image nparray.
'''

def resizeImageForCollage(imageName, sourceFolderPath, verticalDim = 280, horizontalDim = 640):
    imageName = imageName[0]
    Image = imread(sourceFolderPath + imageName)
    resizedImage = resize(Image,  (verticalDim,horizontalDim))  
    resizedImage = np.array(resizedImage)
    
    return resizedImage




'''
getEdgeDetectionHogVariance(sourceFolderPath)
    Input: 'sourceFolderPath' refers to path of folder which contains images.
    Output: returns the dictionary variances of Hog features.
'''


def getEdgeDetectionHogVariance(sourceFolderPath):
    
    
    print('\n\t -> Computing Hog Features for Edge Detection...')
    ListOfEdgeVariance = []
    DictionaryVariances = {}
    
    Images = [ f for f in os.listdir(sourceFolderPath) if f.endswith('.jpg')] # get names of all images

    
    for img in Images:
        ReadImage = imread(sourceFolderPath + img)       # read each image.
            
        _ , HImage = hog(ReadImage,orientations=9, pixels_per_cell=(8,8),cells_per_block=(2,2),visualize=True, multichannel=True) 
        
        ListOfEdgeVariance.append(np.var(HImage.ravel())) # computing and appending the variance values of each image.
        DictionaryVariances[img] = np.var(HImage.ravel())
      
    print('\t -> Variance computation in Edge Detection Done...\n')    
    # normalize the variance values within [0-1]
    maximumVarianceVal = np.max(np.array(list(DictionaryVariances.values())))
    for (key, value) in DictionaryVariances.items():
        DictionaryVariances[key] = value /maximumVarianceVal

    return DictionaryVariances


'''
getRGBHist(sourceFolderPath)
    Input: 'sourceFolderPath' refers to path of folder which contains images.
    Output: returns the Dictionary object of RGB histogram of the images.
'''
def getRGBHist(sourceFolderPath):
    histDict = {}
    imagesList = [ f for f in os.listdir(sourceFolderPath) if f.endswith('.jpg')]

    # for every image retrieve histogram.
    for imgName in imagesList:
        filename = sourceFolderPath + imgName
        image = io.imread(filename)
        dict_images[imgName] = image
        hist = np.asarray(skimage.exposure.histogram(image , nbins = 256)).flatten()  # retrive RGB hist and flatten it.
        histDict[imgName] = hist
    return histDict



'''
doChebychevSorting(dict_histograms)
    Input: takes the dictionary object of histograms.
    Output: returns the dictionary of images with value as chebychev distance with respect to random image.
'''

def doChebychevSorting(dict_histograms):
    
    print('\t -> Computing RGB histograms for Color based processing...')
    dictionaryDistances = {}
    randomImageFromDict = random.choice(list(dict_histograms.keys()))   # take random key out of image collection.

    # for every image compute the chebychev distance wrt to random image selected.
    for (key, histValue) in dict_histograms.items():
        
        currentDistanceFromRandomImage = dist.chebyshev(dict_histograms[randomImageFromDict], histValue) 
        dictionaryDistances[key] = currentDistanceFromRandomImage                    # store the distance value.
    
    print('\t -> Chebychev distances are computed for Color based processing...\n')
    # normalize the variance values within [0-1]
    maximumColorVariance = np.max(np.array(list(dictionaryDistances.values())))
    for (key, value) in dictionaryDistances.items():
        dictionaryDistances[key] = value /maximumColorVariance
    
    return dictionaryDistances


'''
refineCollageEdges(collageArray)
    Input: takes the image of Collage.
    Output: returns the collage with blurring of edges between the images.
'''

def refineCollageEdges(collageArray):

    # normalizing the values in collage.
    maxValueInCollage = np.max(collageArray)
    collageImage = np.array(collageArray) / maxValueInCollage

    # creating the blur rectangle mask1.
    rectangle1_TopCoordinates = (310,0 )
    rectangle1_BottomCoordinates = (330,290)

    # generating blur1 rectangle coordinates.
    coordinateX, coordinateY = rectangle1_TopCoordinates[0], rectangle1_TopCoordinates[1]
    width, height = rectangle1_BottomCoordinates[0] - rectangle1_TopCoordinates[0], rectangle1_BottomCoordinates[1] - rectangle1_TopCoordinates[1]

    # creating Region of Interest for mask1.
    roiMask1 = collageImage[ coordinateY:coordinateY + height, coordinateX:coordinateX + width]
    blurMask1 = skimage.filters.gaussian( roiMask1, sigma = 5, cval = 0.2, mode = 'reflect')

    # apply blurMask1 on the collageImage
    collageImage[ coordinateY:coordinateY + height, coordinateX:coordinateX + width] = blurMask1

    # creating the blur rectangle mask2.
    rectangle2_TopCoordinates = (0,270)
    rectangle2_BottomCoordinates = (640,290)

    # generating blur2 rectangle coordinates
    coordinateX, coordinateY = rectangle2_TopCoordinates[0], rectangle2_TopCoordinates[1]
    width, height = rectangle2_BottomCoordinates[0] - rectangle2_TopCoordinates[0], rectangle2_BottomCoordinates[1] - rectangle2_TopCoordinates[1]

    # creating Region of Interest for mask2 
    roiMask2 = collageImage[ coordinateY:coordinateY + height, coordinateX:coordinateX + width]
    blurMask2 = skimage.filters.gaussian( roiMask2, sigma = 5, cval = 0.2, mode = 'reflect')

    # apply blurMask2 on the collageImage
    collageImage[ coordinateY:coordinateY + height, coordinateX:coordinateX + width] = blurMask2
    
    # creating the blur rectangle mask3.
    rectangle3_TopCoordinates = (0,550)
    rectangle3_BottomCoordinates = (640,570)

    # generating blur3 rectangle coordinates
    coordinateX, coordinateY = rectangle3_TopCoordinates[0], rectangle3_TopCoordinates[1]
    width, height = rectangle3_BottomCoordinates[0] - rectangle3_TopCoordinates[0], rectangle3_BottomCoordinates[1] - rectangle3_TopCoordinates[1]

    # creating Region of Interest for mask3
    roiMask3 = collageImage[ coordinateY:coordinateY + height, coordinateX:coordinateX + width]
    blurMask3 = skimage.filters.gaussian( roiMask3, sigma = 5, cval = 0.2, mode = 'reflect')

    # apply blurMask2 on the collageImage
    collageImage[ coordinateY:coordinateY + height, coordinateX:coordinateX + width] = blurMask3

    # creating the blur rectangle mask4.
    rectangle4_TopCoordinates = (190,290)
    rectangle4_BottomCoordinates = (220,560)

    # generating blur4 rectangle coordinates
    coordinateX, coordinateY = rectangle4_TopCoordinates[0], rectangle4_TopCoordinates[1]
    width, height = rectangle4_BottomCoordinates[0] - rectangle4_TopCoordinates[0], rectangle4_BottomCoordinates[1] - rectangle4_TopCoordinates[1]

    # creating Region of Interest for mask4 
    roiMask4 = collageImage[ coordinateY:coordinateY + height, coordinateX:coordinateX + width]
    blurMask4 = skimage.filters.gaussian( roiMask4, sigma = 5, cval = 0.2, mode = 'reflect')

    # apply blurMask2 on the collageImage
    collageImage[ coordinateY:coordinateY + height, coordinateX:coordinateX + width] = blurMask4
    
    
    # creating the blur rectangle mask5.
    rectangle5_TopCoordinates = (430,290)
    rectangle5_BottomCoordinates = (460,560)

    # generating blur5 rectangle coordinates
    coordinateX, coordinateY = rectangle5_TopCoordinates[0], rectangle5_TopCoordinates[1]
    width, height = rectangle5_BottomCoordinates[0] - rectangle5_TopCoordinates[0], rectangle5_BottomCoordinates[1] - rectangle5_TopCoordinates[1]

    # creating Region of Interest for mask5 
    roiMask5 = collageImage[ coordinateY:coordinateY + height, coordinateX:coordinateX + width]
    blurMask5 = skimage.filters.gaussian( roiMask5, sigma = 5, cval = 0.2, mode = 'reflect')

    # apply blurMask2 on the collageImage
    collageImage[ coordinateY:coordinateY + height, coordinateX:coordinateX + width] = blurMask5

    return collageImage



'''
CollageCreate(sourceFolderPath) 

    Input: 'sourceFolderPath' refers to path of folder which contains 10 sample images.
    Output(s): creates collages based on hybrid approach taking Edge Detection and Color Information.
               returns the generated collage.

'''

def CollageCreate(sourceFolderPath):
    
    # Use edge detection hog features variance. 
    edgeDictionary = {}
    edgeDictionary = getEdgeDetectionHogVariance(sourceFolderPath)
    

    # create collage using color based information.
    '''
        RGB color space is used and corresponding histograms are compared using chebychev distances.
        The distances values are computed with respect to a random image from the pool of 10 images.
        And then images are sorted based on the values of chebychev distances amongst them.
    '''
    dict_histograms = {}                                            # corresponds to the dictionary which store RGB histograms as values and imageName as keys.
    dict_histograms = getRGBHist(sourceFolderPath)                  # get RGB histograms of images.
    colorDictionary = {}
    colorDictionary = doChebychevSorting(dict_histograms)     # sorting is done based on chebychev distance.
    

    
    hybridDictionary = {}
    
    print('\t -> Combining Edge based and Color based approach...')
    for (edgeKey, edgeVarianceValue) in edgeDictionary.items():
        
        # find the corresponding distance value from colorDictionary
        colorDistance = colorDictionary[edgeKey]
        hybridDictionary[edgeKey] = colorDistance + edgeVarianceValue
    
    print('\t -> Sorting the values based on Hybrid approach...\n')
    SortedDictionary = sorted(hybridDictionary.items(), key = lambda KeyValue:(KeyValue[1],KeyValue[0]))    # sort the variance values.
    CollageImages = SortedDictionary[0:6]
    
    print('\t -> Creating collage...')
    sortedImagesForCollage5 = (resizeImageForCollage(CollageImages[5], sourceFolderPath, 280, 320 ))
    sortedImagesForCollage2 =  (resizeImageForCollage(CollageImages[2], sourceFolderPath,280, 640 ))

    vertical1 = np.hstack((sortedImagesForCollage5, resizeImageForCollage(CollageImages[4],sourceFolderPath, 280, 320)))
    verticalMid = np.hstack((resizeImageForCollage(CollageImages[1],sourceFolderPath,280,200), resizeImageForCollage(CollageImages[0],sourceFolderPath, 280, 240), resizeImageForCollage(CollageImages[3],sourceFolderPath, 280, 200)))
    collageImage = np.vstack((np.array(vertical1), np.array(verticalMid), np.array(sortedImagesForCollage2)))
    
    # deleting old collage if present.
    if(os.path.exists(path + 'output')):
        shutil.rmtree(path + 'output')
    
    os.mkdir(path + 'output')
    
    print('\t -> Refining collage to add blurring effect within image edges...')
    
    collageImage = refineCollageEdges(collageImage)
    # save the collage in the output folder.
    print('\t -> Saving collage in output folder...')
    plt.imsave(path + 'output\\HybridCollage.jpg', collageImage)




print('================================================[START]==========================================================')

path = os.getcwd() + '\\images\\'   # corresponds to the path containing sample images.

CollageCreate(path)
print('\n\t -> Please check the output folder for collage')
print('================================================[END]==========================================================')

