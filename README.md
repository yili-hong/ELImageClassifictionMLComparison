#Comparisons of ML and DL Methods for EL Image Classifiction 
This repo contains the data and code for the paper titled "A Comprehensive Case Study on the Performance of Machine Learning Methods on the Classification of Solar
Panel Electroluminescence Images," by Xinyi Song, Kennedy Odongo, Francis G. Pascual, and Yili Hong. The paper is published by Journal of Quality Technology, and its DOI is: 10.1080/00224065.2024.2394604. A preprint versin of the paper is also available at arXiv: 10.48550/arXiv.2408.06229


There are three main folders in this repo and one .csv file. 

Three Folders: 

1. visualization: This folder includes the model results stored in the csv_result directory, as well as the code used to generate the plots mentioned in the paper.

2. models: This folder includes the code of both machine and deep learning models.

3. images: Those individual EL images are saved in the directory of images and this folder is derived from the public dataset (website: https://github.com/zae-bayern/elpv-dataset). similar to path of labels.csv file, this folder should be placed outside the model training folder for proper use.

One .csv file:

labels.csv: This csv file is created by the authors of the EL image dataset: each image is labeled with a defect probability (a floating number ranging from 0 to 1) and specifies the type of solar module (either monocrystalline or polycrystalline) from which the solar cell image was derived. The notations of individual images are located in the labels.csv file. This file and EL image data should be placed outside the model training folder for proper use.