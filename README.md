# Collage Creation using Skimage library
Task : Make a collage of images in python where six images are randomly chosen from given ten     images, and collage is created based on color/edge similarity.

Constraints : PIL and open-cv libraries cannot be used.

Data :
Frames are extracted from a video of a game named “Valorant” using FFMPEG
(https://www.youtube.com/watch?v=e_E9W2vsRbQ)
Frame resolution: 640 x 480 (width*height)

Method :
Collage made finally, is based on a hybrid approach combining “Histogram of Oriented Gradients” and “Chebyshev distance” between images.
To extract Edge information

➢	skimage.feature.hog is used for obtaining ndarray of oriented gradients for each image. Variance is calculated for each ndarray and stored in a dictionary of format {imagename: variance}. 

Extracting Color information

➢	Randomly one image is chosen for reference from the folder and Chebyshev distance between the reference image and all images in the folder is calculated. These distances are stored in a dictionary of format {imagename: distance}. 

Sorting of Images (Hybrid approach):

➢	The values (variance and distance) of both the dictionaries above formed are normalized so that both approaches are given equal weightage. Then, both values are added to form a new dictionary of the format {imagename: hybrid weight}. Then this hybrid dictionary is sorted on the basis of values and the first 6 images are chosen.  

The image with the least variance is placed at the center.
After this, the ends of images in the collage are blurred using the gaussian filter.


Note:
-	“path” variable in the “collage.py” file contains the path of folder containing sample images.
-	The collage generated is random and will be stored in a folder named “output”.
