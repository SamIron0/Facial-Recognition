# Facial Recognition Algorithm

This project implements a simple facial recognition algorithm using eigenface analysis.

## Running the Program

To run the program, execute the following command in your terminal:

```bash
python FacialRecognition.py
```

### IDEA 
The algorithm reads in pre-processed PGM image files from the `/duplicate` folder, a simple gray-scale image format and utilizes eigenvalues, eigenvectors, and vector projection to compute the weights and distances between input images and images in `/library` folder. The algorithm attempts to match the same individuals appearing in both libraries.
Images are pre-processed by zooming into head region and ensuring face is centered in image and forward facing as shown below.
![preprocessImage](./ss.png)

## Algorithm Overview

The algorithm processes pre-processed PGM (Portable Gray Map) image files from the `/duplicate` folder. It utilizes eigenvalues, eigenvectors, and vector projection to compute weights and distances between input images and images in the `/library` folder. The goal is to match individuals appearing in both libraries.

### Image Pre-processing

Images are pre-processed by:
1. Zooming into the head region
2. Ensuring the face is centered in the image
3. Ensuring the face is forward-facing

Example of a pre-processed image:

![Pre-processed Image](./ss.png)

## Implementation Details

The implementation uses Python and NumPy to:
1. Represent face images as matrices
2. Perform mathematical operations to isolate distinguishing facial features
3. Calculate eigenvalues and eigenvectors of these matrices
4. Identify significant features of each image (eigenface analysis)

## Performance

The algorithm's performance has been improved through image processing techniques:

- Initial hit rate: 8%
- Improved hit rate: 9%

Improvements were achieved by:
1. Cropping images
2. Increasing image contrast before calculations

![Result Image](./resultss.png)

## Future Improvements

Potential areas for future enhancement include:
1. Experimenting with different pre-processing techniques
2. Implementing more advanced machine learning algorithms
3. Optimizing performance for larger datasets

## Contributing

Contributions to improve the algorithm or extend its functionality are welcome. Please submit a pull request or open an issue to discuss proposed changes.

## License

[Add your chosen license here]
