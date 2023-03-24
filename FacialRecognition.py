import numpy as np

'''
Samuel Ironkwe

A simple facial recognition algorithm. The algorithm crops and increases the contrast
of the images before calculation.
'''

'''
Modify these constants accordingly for your own library/test images.
Note that the image size must be identical for every image.
INPUT_FILE and LIBRARY_LABELS should be located in the current working directory.
'''
NUM_FILES = 300 #number of library images
IMAGE_WIDTH = 250   #size of face images
IMAGE_HEIGHT = 250

INPUT_FILE = '/Users/sam/Desktop/Facial Recognition/duplicates/Ahmed_Chalabi_0005.pgm'  #input image
LIBRARY_LABELS = 'library.txt'  #text file containing file containing the name of each library image
LIBRARY_FILE_PATH = '/Users/sam/Desktop/Facial Recognition/library/' #file path of face image library

CROP_WIDTH = 150  #desired size to crop images to 
CROP_HEIGHT = 150
CONTRAST = 0.89 #percentage increase in contrast

def main():
    
    image = read_PGM(INPUT_FILE)
    
    '''
    Performance testing code. Input a filepath to a library of test images 
    and a text file containing the names of the images. 
    '''
    test_images('/Users/sam/Desktop/Facial Recognition/duplicates/','duplicates.txt')   

'''
Crops an image to a given width and height, centered on the middle of the original image.
'''
def crop(image_data):
    #amount of pixels to remove from each side
    x_pixels =  int((IMAGE_WIDTH-CROP_HEIGHT)/2)
    y_pixels =  int((IMAGE_HEIGHT-CROP_WIDTH)/2)

    #crop image
    new_data = image_data[x_pixels:IMAGE_WIDTH-x_pixels,y_pixels:IMAGE_HEIGHT-y_pixels]
   
    #if a new dimension is to be odd, we need to remove 1 more pixel from one side
    if(CROP_WIDTH % 2 != 0):
        new_data = new_data[:CROP_WIDTH,:]
    if(CROP_HEIGHT % 2 != 0):
        new_data = new_data[:,:CROP_HEIGHT]

    return new_data


'''
Increases the contrast of an image by a percentage.
'''
def increase_contrast(image_data):
    max_shade = np.amax(image_data) 
    threshold = max_shade/2 #half the max shade of the image
    
    data = image_data.astype(float)

    #increase pixels above the threshhold by the percentage, and 
    #decrease pixels below the threshhold by the pecentage.
    np.putmask(data, data >= threshold, data+(data*CONTRAST))
    np.putmask(data, data < threshold, data-(data*CONTRAST))
    
    #ensure the pixel data is within the range [0,max_shade]
    np.putmask(data, data > max_shade, max_shade)
    np.putmask(data, data < 0, 0)

    return data.astype(int)


'''
Tests a list of images. filepath is the location of the folder
containing the test images, and label file is a text file containing the 
names of the images. Finds the closest match of each image, the average distance of 
these matches, and the number of successful matches. Prints a report of the results.

If modified == False the unmodified algorithm will be used instead. (no cropping/contrast adjustment)
By default, modified will be True.
'''
def test_images(filepath, label_file, modified=True):

    distance_total = 0 #sum of all the distances in the test folder
    matches = 0 #number of sucessfull matches

    f = open(LIBRARY_LABELS)
    labels = f.readlines()
    f.close()

    L_matrix = concatentate_pixel_columns(LIBRARY_FILE_PATH,labels,modified)

    #calculate the average of each row of the library matrix
    average_vector = np.mean(L_matrix, axis=1)
    #subtract the average vector from each column of the library matrix
    L_matrix = L_matrix - np.vstack(average_vector)

    eigfaces = eigenfaces(L_matrix)
    weights = column_weights(eigfaces,L_matrix)

    f = open(label_file)
    test_labels = f.readlines()
    f.close()

    #find the closest match for every image in the test folder
    for l in test_labels:
        image_data = read_PGM(filepath+l.strip())

        if(modified == True):
            image_data = crop(image_data)
            image_data = increase_contrast(image_data)

        distances = distance_array(image_data, average_vector, eigfaces, weights)
        min_distance = np.amin(distances)

        distance_total += min_distance

        index = np.where(distances == min_distance)[0][0]

        #remove the suffix of the label and check if names match
        name_index = l.rindex("_")
        if(labels[index][:name_index] == l[:name_index]):
            matches+=1
         
        print("----------------------------------\nTesting {}...\nBEST MATCH: {}DISTANCE:{}".format(l.strip(),labels[index],min_distance))
           
    #compute average
    avg_distance = distance_total/len(test_labels)

    print("----------------------------------\nAVERAGE DISTANCE: {}\nSUCCESSFUL MATCHES: {}".format(avg_distance, matches))
    

'''
Creates an matrix from a library of face images where each column of the matrix is the data
from a single face image in the library. Crops and increases the contrast of the images before 
creating the matrix. Accepts a file path to a library of images as 
a paramater, and returns the resulting matrix. 

If modified == False, the unmodified algorithm will be used (no cropping/contrast reduction).
By default, modified will be True.
'''
def concatentate_pixel_columns(filepath, labels, modified=True):

    if(modified == True):
        cols = CROP_WIDTH*CROP_HEIGHT
    else:
       cols = IMAGE_WIDTH*IMAGE_HEIGHT

    matrix = np.zeros((NUM_FILES,cols))

    #transform each input file into a single array of data and add it to the matrix
    i = 0
    while(i < NUM_FILES):
        image = read_PGM(filepath+labels[i].strip()) 

        if(modified == True):
            image = crop(image)
            image = increase_contrast(image)
         
            
        matrix[i] = np.ravel(image)
        i+=1


    #take the transpose of the matrix so that the arrays are organized into columns
    matrix = np.transpose(matrix)

    return matrix


'''
Given a matrix L, compute the eigenfaces of the matrix.
Returns an array of eigenface vectors.
'''
def eigenfaces(L_matrix):
    
    sigma_x = 1/(NUM_FILES-1) * np.matmul(np.transpose(L_matrix), L_matrix)
    eigvals, eigvecs = np.linalg.eig(sigma_x)

    #sort eigenvalues and corresponding eigenvectors in descending order
    inds = np.flip(eigvals.argsort())
    eigvals = eigvals[inds]
    eigvecs = eigvecs[:,inds]

    #calculate the sum of the eigenvectors and the cutoff
    #for calculating the eigenfaces
    eig_sum = np.sum(eigvals)
    cutoff = 0.95 * eig_sum
    
    #find k, the number of eigenvalues that make up
    #95% of the sum of all eigenvalues
    k = 0
    total = 0
    while(total <= cutoff):
        total += eigvals[k] 
        k+=1
    
    #compute k eigenvectors of L and renormalize to length 1
    #eigfaces = np.zeros((k,CROP_WIDTH*CROP_HEIGHT))
    eigfaces = np.zeros((k,np.shape(L_matrix)[0]))
    for i in range(k):
        eigfaces[i] = np.matmul(L_matrix,eigvecs[i])
        eigfaces[i] = eigfaces[i]/np.linalg.norm(eigfaces[i])
    
    return eigfaces


'''
For each column in the matrix L, compute the weight vector.
Returns an array of weight vectors corresponding to each column vector.
'''
def column_weights(eigfaces, L):
    num_cols = np.shape(L)[1] #number of columns in L
    num_vecs = np.shape(eigfaces)[0] #number of eigenvectors 
    
    weights = np.zeros((num_cols, num_vecs)) 
    
    #compute the weight vector for each column of L:
    for j in range(num_cols):
        col = L[:,j]
        weights[j] = weight_vector(col,eigfaces)
       
    return weights


'''
Given a column vector and an array of eigenvectors, compute the weight vector.
The weight vector is composed of the projection of the column vector onto each eigenvector
Returns the resulting weight vector.
'''
def weight_vector(col_vector, eigvecs):
    size = np.shape(eigvecs)[0]
    weights = np.zeros(size)

    for i in range(size):
        weights[i] = np.dot(col_vector,eigvecs[i])
    
    return weights


'''
Find the distance between each library vector and the input data vector.
Each number represents the distance between the test image and the corresponding
library image.
Accepts input image data, the average library vector, an array of eigenfaces, and
an array of column weights as parameters.
Returns an array of distances.
'''
def distance_array(image_data, average_vector, eigfaces, col_weights):
    #transform the input image into a 1d array and subtract 
    #the average of the library vectors
    image_data = np.ravel(image_data)
    image_data = image_data - average_vector

    #calculate the corresponding weight vector
    image_weight = weight_vector(image_data, eigfaces)
  
    #for each column in the library matrix, compute the distance
    #this represents the distance between the input image and the
    distances = np.zeros(NUM_FILES) 
    for j in range(NUM_FILES):
        distances[j] = np.linalg.norm(col_weights[j] - image_weight)
    
    return distances


"""
Given a file name, return a numpy array containing the pixel values of a pgm file.
"""
def read_PGM(file_name):
    f = open(file_name)
    lines = f.read()
    
    #remove comments
    comment_index = lines.find('#')
    while(comment_index != -1):
        newline_index = lines.find('\n',comment_index)
        lines = lines[:comment_index] + lines[newline_index:]
        comment_index = lines.find('#')

    #split on whitespace and remove blank lines
    lines = lines.split()

    #assert that the file is of type P2
    assert lines[0] == 'P2', 'Not a PGM file of type P2'

    #read in size and convert to integer tuple
    size = lines[1:3]
    size_tuple = tuple(int(n) for n in size)

    #convert data to numpy array (ignoring header)
    data = np.array(lines[4:]).reshape(size_tuple[1], size_tuple[0])
    data = data.astype(int)

    f.close()
    return data



if __name__ == "__main__":
    main()
