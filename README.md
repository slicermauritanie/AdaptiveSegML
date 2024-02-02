Welcome to AdaptiveAlgorithm-3DSlicer, a repository dedicated to integrating adaptive algorithms into Slicer to enhance the analysis of 3D medical images.


<!-- Add pictures and links to videos that demonstrate what has been accomplished. -->

We have developed an extension for 3D Slicer to perform medical image (volume) segmentation using the K-Means algorithm. Specifically, we have implemented an adaptive version of K-Means, which allows segmentation based on pixel intensity.

We encountered several challenges during volume processing and rendering, as well as in finding alternatives to libraries like scikit-learn, NumPy, and OpenCV to integrate them into the 3D Slicer API.

Ultimately, we successfully segmented the images using both the adaptive K-Means and the classic K-Means methods. However, these results still require improvement and testing on various types of medical images to ensure their reliability


**Adaptive Algorithm Result**

[![Image 1](https://github.com/slicermauritanie/AdaptiveSegML/blob/main/result_images/apaptive_rim1_1.png)](https://github.com/slicermauritanie/AdaptiveSegML/blob/main/result_images/apaptive_rim1_1.png) [![Image 2](https://github.com/slicermauritanie/AdaptiveSegML/blob/main/result_images/apaptive_rim1_2.png)](https://github.com/slicermauritanie/AdaptiveSegML/blob/main/result_images/apaptive_rim1_2.png) [![Image 3](https://github.com/slicermauritanie/AdaptiveSegML/blob/main/result_images/apaptive_rim1_3.png)](https://github.com/slicermauritanie/AdaptiveSegML/blob/main/result_images/apaptive_rim1_3.png)


**Classic Algorithm Result**

![Image 1](https://github.com/slicermauritanie/AdaptiveSegML/blob/main/result_images/classic_al/classic_algorithm_1.png) ![Image 2](https://github.com/slicermauritanie/AdaptiveSegML/blob/main/result_images/classic_al/classic_algorithm_2_99.png) ![Image 3](https://github.com/slicermauritanie/AdaptiveSegML/blob/main/result_images/classic_al/classic_algorithm_3_80.png) ![Image 4](https://github.com/slicermauritanie/AdaptiveSegML/blob/main/result_images/classic_al/classic_algorithm_4_50.png) ![Image 5](https://github.com/slicermauritanie/AdaptiveSegML/blob/main/result_images/classic_al/classic_algorithm_5_20.png) ![Image 6](https://github.com/slicermauritanie/AdaptiveSegML/blob/main/result_images/classic_al/classic_algorithm_6_10.png) ![Image 7](https://github.com/slicermauritanie/AdaptiveSegML/blob/main/result_images/classic_al/classic_algorithm_7_5.png) ![Image 8](https://github.com/slicermauritanie/AdaptiveSegML/blob/main/result_images/classic_al/classic_algorithm_8_2.png)
