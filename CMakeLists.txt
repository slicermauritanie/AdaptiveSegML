cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(AdaptiveSegML)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://www.slicer.org/wiki/Documentation/Nightly/Extensions/AdaptiveSegML")
set(EXTENSION_CATEGORY "Segmentaion")
set(EXTENSION_CONTRIBUTORS "Muhamedoufi (FST - University of Nouakchott), Mohamed Abdellahi Sidi Mohamed Blal (FST - University of Nouakchott), Elhacen (FST - University of Nouakchott)")
set(EXTENSION_DESCRIPTION "The  AdaptiveSegML" extension for 3D Slicer is a powerful toolkit for image segmentation using adaptive machine learning algorithms. This extension incorporates a range of advanced algorithms, including Fuzzy C-Mean, K-Means, Otsu Thresholding, and other adaptive methods designed to enhance image segmentation through intelligent pixel intensity analysis.

Key Features:

Fuzzy C-Mean (FCM) Algorithm:

Leverage the FCM algorithm for soft clustering, allowing nuanced assignment of pixel intensities to multiple segmentation classes.
K-Means Algorithm:

Utilize the K-Means clustering algorithm to partition image data into distinctive clusters, enhancing segmentation accuracy based on pixel intensity.
Otsu Thresholding:

1. Apply Otsu's method for automatic threshold selection, optimizing the separation of foreground and background by analyzing the pixel intensity distribution.
Adaptive Segmentation:

2. Achieve adaptive segmentation by dynamically adjusting algorithm parameters based on the pixel intensity characteristics of the input image. This ensures robust segmentation results across various imaging modalities and conditions.
User-Friendly Interface:

1. Benefit from an intuitive and user-friendly interface within 3D Slicer, providing seamless integration with the existing platform for medical image analysis.
Versatility:

3. Support a wide range of medical imaging applications, including but not limited to MRI, CT, and other modalities, making the extension a versatile tool for researchers and clinicians.
Enhanced Workflow:

4. Improve workflow efficiency with the integration of adaptive machine learning algorithms, reducing the need for manual intervention and providing accurate and consistent segmentation results.

The AdaptiveSegML extension empowers users with a comprehensive set of tools for adaptive image segmentation, making it an invaluable resource for medical professionals, researchers, and developers working with 3D Slicer.""AdaptiveSegML" extension for 3D Slicer is a powerful toolkit for image segmentation using adaptive machine learning algorithms. This extension incorporates a range of advanced algorithms, including Fuzzy C-Mean, K-Means, Otsu Thresholding, and other adaptive methods designed to enhance image segmentation through intelligent pixel intensity analysis.

Key Features:

Fuzzy C-Mean (FCM) Algorithm:

Leverage the FCM algorithm for soft clustering, allowing nuanced assignment of pixel intensities to multiple segmentation classes.
K-Means Algorithm:

Utilize the K-Means clustering algorithm to partition image data into distinctive clusters, enhancing segmentation accuracy based on pixel intensity.
Otsu Thresholding:

1. Apply Otsu's method for automatic threshold selection, optimizing the separation of foreground and background by analyzing the pixel intensity distribution.
Adaptive Segmentation:

2. Achieve adaptive segmentation by dynamically adjusting algorithm parameters based on the pixel intensity characteristics of the input image. This ensures robust segmentation results across various imaging modalities and conditions.
User-Friendly Interface:

1. Benefit from an intuitive and user-friendly interface within 3D Slicer, providing seamless integration with the existing platform for medical image analysis.
Versatility:

3. Support a wide range of medical imaging applications, including but not limited to MRI, CT, and other modalities, making the extension a versatile tool for researchers and clinicians.
Enhanced Workflow:

4. Improve workflow efficiency with the integration of adaptive machine learning algorithms, reducing the need for manual intervention and providing accurate and consistent segmentation results.

The AdaptiveSegML extension empowers users with a comprehensive set of tools for adaptive image segmentation, making it an invaluable resource for medical professionals, researchers, and developers working with 3D Slicer.")
set(EXTENSION_ICONURL "https://www.example.com/Slicer/Extensions/AdaptiveSegML.png")
set(EXTENSION_SCREENSHOTURLS "https://www.example.com/Slicer/Extensions/AdaptiveSegML/Screenshots/1.png")
set(EXTENSION_DEPENDS "NA") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(kmeans)
add_subdirectory(classique_kmeans)
add_subdirectory(classique_kmeans)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
