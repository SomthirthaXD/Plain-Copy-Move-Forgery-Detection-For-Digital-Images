import cv2
import numpy as np
import time
import os

totime=0.0 #Variable to store total time elapsed during entire program.
#............Taking RGB image input...........
#............The image has to be present in the current PyCharm Project in this case, subject to further rectifications..................
img=cv2.imread('Image (4).png')
amask=cv2.imread('Mask (4).png')
amask=cv2.threshold((cv2.cvtColor(amask, cv2.COLOR_BGR2GRAY)), 0, 255, cv2.THRESH_BINARY)[1] #Converting 'amask' to binary image.
h=img.shape #h[0] is the height.
w=img.shape #w[1] is the width.
print ("Image height:", h[0])
print ("Image width:", w[1])

#.............Now we divide the image into blocks of unit size bsizexbsize........................
bsize=int(input("Enter block size!"))
print ("Block division started!")
st=time.time()
#.............We will need one huge list with the provision of containing all the unit sized block arrays stored...............
#.............Start of block division process....................
blstr=[] #List to store blocks.
blinf=[] #To store pixel information for every corresponding blstr block.
blcount=0 #Block counter.
for i in range (h[0]):
    for j in range (w[1]):
        blstr.append(img[i:i+bsize, j:j+bsize]) #Appends individual blocks of size "bszie" to blstr.
        blinf.append([i, j]) #Storing start pixel of each block by storing it's i and j co-ordinates.
        blcount+=1 #Counting number of blocks.
print ("Block division finsihed!")
print ("Total unit blocks obtained:", blcount)
et=time.time()
totime+=(et-st)
print("Time elapsed in block division and counting:", (et-st), "seconds!")

#..............Now we start separating all the blocks in blstr into R, G and B grayscales...............................
#..............We then store these grayscale images for each block into a list. Thus there will be a total of blcountx3 gray images..................
chnlst=[] #List that stores channels of RGB.
chn=[] #Temporary storage for each block's channels.
print("Channel division started!")
st=time.time() #Initiates timer.
for i in blstr:
    for j in range(3):
        i1 = i.copy()
        if (j==0):
            i1[:, :, 1] = 0
            i1[:, :, 2] = 0 #Set green and red channels to 0.
        elif (j==1):
            i1[:, :, 0] = 0
            i1[:, :, 2] = 0 #Set blue and red channels to 0.
        else:
            i1[:, :, 0] = 0
            i1[:, :, 1] = 0 #Set blue and green channels to 0.
        chn.append(cv2.cvtColor(i1, cv2.COLOR_BGR2GRAY)) #Appending the sliced up R, G, and B images on a temporary list. Value of j defines which if R, G or B image is being extracted.
    #cv2.cvtColor(i1, cv2.COLOR_BGR2GRAY) converst an RGB image to a grayscale image using library function of OpenCV.
    chn1=chn[:] #Temporary copy of R, G and B grayscale images.
    chnlst.append(chn1) #Appending the copy list to the main storage list.
    chn.clear()
print("Channel division finished!")
et=time.time() #Stops timer.
totime+=(et-st)
print("Time elapsed in R, G, B gray channel division:", (et-st), "seconds!")

#............Feauture extraction is a process in which various features of a channel of an image block such as it's mean, median and standard deviation are calculated and stored in a way that it generates a word..............
#............The generated feature word acts as an unique identifier for the particular block..............
#............In this section, the features of RGB channels of each block of the image are extracted and stored into individual blocks..............
word=[] #List to store the mean, median and standard deviation of each channel.
wtemp=[] #Temporary storage of image features.
c1=0 #Counter for counting number of blocks moved ahead.
print("Word generation started!")
st=time.time()
for i in chnlst:
    tempw=[] #Temporary list to store the mean,median and standard deviation of each channel.
    for j in i:
        m=np.mean(j) #Calculates mean of each channel for a particular block.
        sd=np.std(j) #Calculates standard deviation of each channel for a particular block.
        md=np.median(j) #Calculates median of each channel for a particular block.
        tempw.append(m) #Appends mean to temporary list.
        tempw.append(sd) #Appends standard deviation to temporary list.
        tempw.append(md) #Appends median to temporary list.
    wtemp=tempw[:] #Temporary copy of image features.
    word.append([wtemp, blinf[c1]]) #Appends features to final list along with pixel information for each corresponding block simultaneously.
    c1=c1+1 #Counter incremented by one with each block iteration
print("Word generation finished!")
et=time.time()
totime+=(et-st)
print("Time elapsed in word generation:", (et-st), "seconds!")

#............Normalising the word....................
#............This is done in order to bring values within a range of 0 to 1 in the word............
st=time.time()
print("Word normalisation started!")
max=np.zeros(9) #Taking an array of 9 places to store max value.
for i in range(0, blcount):
    c=0
    for j in range(0,9):
        if (word[i][0][j]>max[c]):
            max[c]=word[i][0][j] #Storing max value.
        c+=1
for i in range(0, blcount):
    c=0
    for j in range(0,9):
        if (max[c]!=0):
            word[i][0][j]=word[i][0][j]/max[c] #Dividing each element by corresponding max value.
        c+=1
et=time.time()
totime+=(et-st)
print("Word normalisation finished!")
print("Time elapsed in word normalisation:", (et-st), "seconds!")

#............Now we perform lexicographic sorting of the elements of the word to make similar blocks come next to each other.................
print("Lexicographic sorting of word started!")
st=time.time()
wordsrt=sorted(word, key=lambda i: i[0]) #List 'word' is lexicographically sorted using this function line.
#i[0] denotes sorting is being done by taking only the features into consideration and not the pixel information.
print("Lexicographic sorting of word finished!")
et=time.time()
totime+=(et-st)
print("Time elapsed in lexicographic sorting:", (et-st), "seconds!")

#............Next up, we calculate the euclidean distance between two totally similar blocks...............
st=time.time()
tdist=2*bsize*bsize #Minimum threshold euclidean pixel distance for suspicious block markings based on formula.
bdist=tdist #Maximun distance of checking for identical blocks in list. Equal to tdist in case of unisized blocks.
print("Painting prediction mask started!")
pmask=np.zeros((h[0], w[1]), dtype='uint8') #Prediction mask.
for i in range(0, blcount-bdist-1):
    print (i)
    for j in range(i+1, i+bdist):
        edist=np.sqrt(((wordsrt[j][1][0]-wordsrt[i][1][0])*(wordsrt[j][1][0]-wordsrt[i][1][0]))+((wordsrt[j][1][1]-wordsrt[i][1][1])*(wordsrt[j][1][1]-wordsrt[i][1][1]))) #Euclidean distance of concerned blocks.
        if(edist>tdist): #Checking if euclidean distance of blocks greter than threshold.
            if(wordsrt[i][0]==wordsrt[j][0]): #Checking if features of block is similar.
                pmask[wordsrt[i][1][0]:wordsrt[i][1][0]+bsize, wordsrt[i][1][1]:wordsrt[i][1][1]+bsize]=255 #Painting identical blocks white in output image.
                pmask[wordsrt[j][1][0]:wordsrt[j][1][0] + bsize, wordsrt[j][1][1]:wordsrt[j][1][1] + bsize]=255
et =time.time()
totime+= (et - st)
print("Painting prediction mask done!")
print("Time elapsed in painting prediction mask:", (et - st), "seconds!")

#.................Now we eliminate invalid non suspicious blocks.....................
print ("Invalid block detection and re-painting started!")
st=time.time()
#For achieving this, we use the connected components detection procedure on the prediction mask and discard the components that are not sufficiently big in size.
con=cv2.connectedComponentsWithStats(pmask, 8, cv2.CV_32S) #Connected components function by cv2.
(numLabels, labels, stats, centroids)=con #Storing the stats returned by the function to the variable 'con' in 4 sized tuple.
# Loop over the number of unique connected component labels, skipping over the first label (as label zero is the background).
fmask=np.zeros_like(img, dtype='uint8') #Final mask variable matrix.
minarea=np.sqrt(np.sqrt(h[0])*np.sqrt(w[1]))*bsize #Minimum area required to be a valid suspicious block.
x=0
for i in range(1, numLabels):
    #Extracting the connected component statistics for the current label.
    areacc = stats[i, cv2.CC_STAT_AREA]
    #Ensuring the area is not too small.
    if (areacc>minarea and x<3):
        #Constructing a mask for the current connected component and then summing with the 'fmask'.
        label_hue = np.uint8(255*(labels==i)/np.max(labels)) #Giving current component a hue according to label number.
        blank_ch=255*np.ones_like(label_hue)
        labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
        labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR) #Convert to BGR for display.
        #Set background label to black.
        labeled_img[label_hue==0]=0
        fmask+=labeled_img #Add each new component to 'fmask'.
        x+=1
gray=cv2.cvtColor(fmask, cv2.COLOR_BGR2GRAY)
fmask=cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1] #Converting 'fmask' to binary image.
et=time.time()
totime+=(et - st)
print ("Invalid block deetction and re-painting finished!")
print ("Time elapsed in invalid block detction and re-painting:", (et - st), "seconds!")

#.................Now we find eroded image of predicted mask and the reduced image.....................
st=time.time()
print("Image erosion and reduction started!")
eele=np.ones((5,5)) #Creating a 5x5 structuring element for image erosion.
eimgtemp=cv2.erode(fmask, eele) #Eroding image.
eimg=fmask-eimgtemp #Finalising reduced image.
et = time.time()
totime+=(et - st)
print("Image erosion and reduction done!")
print("Time elapsed in image erosion and reduction:", (et - st), "seconds!")

#..................Finally, we paint the absolute white outlines of eroded image on the original image to produce the output final image................
print("Final image painting started!")
st=time.time()
fimg=img.copy() #Final image.
for i in range (h[0]):
    for j in range (w[1]):
        if (eimg[i][j]==255): #If a pixel in the eroded image is white, we paint that pixel in the final image to absolute red.
            fimg[i][j][0]=0
            fimg[i][j][1]=0
            fimg[i][j][2]=255
#This gives us the perfect markings of the predicted mask on the concerned image.
et=time.time()
totime+=(et - st)
print("Final image painting done!")
print("Time elapsed in final image painting:", (et - st), "seconds!")

#....................This section calculates the accuracy of the predicted mask with reference to the provided actual mask........................
print("Pixel match accuracy calculation started!")
st=time.time()
x=0
y=0
z=0
for i in range (h[0]):
    for j in range (w[1]):
        if(fmask[i][j]==255==amask[i][j]):
            x+=1
        if(amask[i][j]==255):
            y+=1
diff=abs(x-y)
acc1=100-(diff/y*100.0)
for i in range (h[0]):
    for j in range (w[1]):
        if(fmask[i][j]==amask[i][j]):
            z+=1
acc2=z/(h[0]*w[1])*100.0
et=time.time()
totime+=(et - st)
print("Accuracy calculation done!")
print("TPR:", acc1)
print("Pixel count accuracy:", acc2)
print("Time elapsed in accuracy calculation:", (et - st), "seconds!")
#....................This section prints all the output images and total time taken in the entire process.....................
print("Total time elapsed in process:", totime, "seconds!")
cv2.imshow('Original image.', img)
cv2.imshow('Initial forgery prediction mask.', pmask)
cv2.imshow('Final forgery prediction mask.', fmask)
cv2.imshow('Eroded image.', eimg)
cv2.imshow('Final output image.', fimg)
cv2.waitKey(0)
cv2.imshow('Image Input.', img) #Input image displayed.
cv2.waitKey(0)
path='C:/Users/Somthirtha/Desktop/Output' #Path for saving final image.
cv2.imwrite(os.path.join(path , 'Final mask.png'), fmask) #Saving final prediction mask image in the path folder.
cv2.imwrite(os.path.join(path , 'Final image.png'), fimg) #Saving final image in the path folder.
print ("FInal image and prediction mask saved in path:", path)
cv2.destroyAllWindows()
