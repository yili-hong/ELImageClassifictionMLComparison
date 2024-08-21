Sub-folder - visualization

elpv_reader.py: A script created by the author of the EL image dataset to assist with data loading. In Python, use the utils/elpv_reader module from this repository to load the images along with their corresponding annotations. (https://github.com/zae-bayern/elpv-dataset) to prepare the histogram plots. 

csv_result: This folder gives test results of 50 replicates for both machine learning and deep learning models among both monocrystalline and polycrystalline PV modules. Considering the problem of limited computation resources, we ran 5 jobs for each of the VGG19 and ResNet50 models, with each job handling 10 replicates based on images from each type of PV module.

Polycrystalline_PV_Module.py: Execute this py file will read the model test results saved in directory "csv_result" and generate all the figures based on polycrystalline PV modules mentioned in this paper.

Monocrystalline_PV_Module.py: Execute this py file will read the model test results saved in directory "csv_result" and generate all the figures based on monocrystalline PV modules mentioned in this paper.




    