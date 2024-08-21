Sub-folder - models 


MONO_VGG19.py: Implements the VGG19 model using solar cell images from monocrystalline PV modules. 

MONO_RESNET50.py: Implements the ResNet50 model using solar cell images from monocrystalline PV modules.

POLY_VGG19.py: Implements the VGG19 model using solar cell images from polycrystalline PV modules.

POLY_RESNET50.py: Implements the ResNet50 model using solar cell images from polycrystalline PV modules.

ML_replicate_MONO.py: Includes machine learning models like logistic regression, SVM, and random forest, based on monocrystalline PV modules.

ML_replicate_POLY.py: Includes machine learning models like logistic regression, SVM, and random forest, based on polycrystalline PV modules.

Running the .py files above could produce test results for 50 replicates, which will be saved as a CSV file.

elpv_reader.py: A script created by the author of the EL image dataset to assist with data loading. In Python, use the utils/elpv_reader module from this repository to load the images along with their corresponding annotations. (https://github.com/zae-bayern/elpv-dataset)

mono_poly: A set of images sorted by the PV module type to help train machine learning models. It contains the same images as the "images" folder, but they are manually divided into two types: monocrystalline and polycrystalline PV modules. This folder helps read data for machine learning model part. 

    