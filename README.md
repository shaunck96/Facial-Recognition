# Facial-Recognition

1. Summary

In this study, face recognition was performed using the face images in the Olivetti data set. The steps for face recognition are as follows:

Principal components of face images were obtained by PCA.
Adequate number of principal components determined
According to three different classification models, accuracy score obtained.
According to three different classification models, cross-validation accuracy score were obtained.
Parameter optimization of the best model has been made.

2.Face Recognition

The first study on automatic facial recognition systems was performed by Bledsoe between 1964 and 1966. This study was semi-automatic. The feature points on the face are determined manually and placed in the table called RAND. Then, a computer would perform the recognition process by classifying these points. However, a fully functional facial recognition application was performed in 1977 by Kanade. A feature-based approach was proposed in the study. After this date, two-dimensional (2D) face recognition have studied intensively. Three-dimensional (3D) face studies were started to be made after the 2000s.

3D facial recognition approaches developed in a different way than 2D facial recognition approaches. Therefore, it will be more accurate to categorize in 2D and 3D when discussing face recognition approaches.

We can classify the face recognition researches carried out with 2D approach in three categories; analytical (feature-based, local), global (appearance) and hybrid methods. While analytical approaches want to recognize by comparing the properties of the facial components, global approaches try to achieve a recognition with data derived from all the face. Hybrid approaches, together with local and global approaches, try to obtain data that expresses the face more accurately.

Face recognition performed in this kernel can assessed under global face recognition approaches.

In analytical approaches, the distance of the determined feature points and the angles between them, the shape of the facial features or the variables containing the regional features are obtained from the face image are used in face recognition. Analytical methods examine the face images in two different ways according to the pattern and geometrical properties. In these methods, the face image is represented by smaller size data, so the big data size problem that increases the computation cost in face recognition is solved.

Global-based methods are applied to face recognition by researchers because they perform facial recognition without feature extraction which is troublesome in feature based methods. Globally based methods have been used in face recognition since the 1990s, since they significantly improve facial recognition efficiency. Kirby and Sirovich (1990) first developed a method known as Eigenface, which is used in facial representation and recognition based on Principal Component Analysis . With this method, Turk and Pentland transformed the entire face image into vectors and computed eigenfaces with a set of samples. PCA was able to obtain data representing the face at the optimum level with the data obtained from the image. The different facial and illumination levels of the same person were evaluated as the weakness point of PCA.

The face recognition performend in this kernel totally based on Turk and Pentland work.

3. Olivetti Dataset

Brief information about Olivetti Dataset:

Face images taken between April 1992 and April 1994.
There are ten different image of each of 40 distinct people
There are 400 face images in the dataset
Face images were taken at different times, variying ligthing, facial express and facial detail
All face images have black background
The images are gray level
Size of each image is 64x64
Image pixel values were scaled to [0, 1] interval
Names of 40 people were encoded to an integer from 0 to 39
