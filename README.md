Lung Tumor Detection and Description Using Convolutional Neural Networks (CNN)

This repository contains the code and data for my Master's thesis on lung tumor detection and description using Convolutional Neural Networks (CNNs). 
The goal of this project is to develop and evaluate deep learning models that can accurately identify and describe lung tumors from medical imaging data.

Table of Contents
-Introduction
-Data
-Methodology
-Model Architecture
-Results
-Conclusion
-Usage
-License
-Acknowledgements

Introduction:
Lung cancer is one of the leading causes of cancer-related deaths worldwide. 
Early detection and accurate characterization of lung tumors are crucial for effective treatment planning and improving patient outcomes. 
In this thesis, I explore the use of Convolutional Neural Networks (CNNs) for the detection and description of lung tumors from medical imaging data.

Data:
The dataset used in this project is the publicly available LIDC-IDRI dataset from The Cancer Imaging Archive (TCIA). 
This dataset includes thoracic CT scans with annotated lesions, providing a rich resource for developing and evaluating algorithms for lung tumor detection and characterization.

Methodology:
The project involves several key steps:

-Data preprocessing: Normalizing and augmenting the imaging data to improve model performance.
-Model training: Developing and training CNN models using PyTorch to detect and describe lung tumors.
-Evaluation: Assessing the models using standard metrics such as accuracy, precision, recall, and F1-score.
-Interpretation: Analyzing the model's predictions and understanding its decision-making process.

Model Architecture:
The CNN model architecture used in this project includes multiple convolutional layers, followed by max-pooling layers, and fully connected layers. 
Various architectures were experimented with, including popular models like VGG16, ResNet50, and custom-designed networks.

Results:
The trained CNN models achieved promising results in detecting and describing lung tumors. 
Detailed results, including performance metrics and visualizations of model predictions, can be found in the results section of this repository.

Conclusion:
The use of CNNs for lung tumor detection and description shows great potential. 
This project demonstrates the effectiveness of deep learning models in medical image analysis and highlights areas for future research and improvement.

License:
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgements:
This work was conducted at the Interdisciplinary Centre for Mathematical and Computational Modelling (ICM), University of Warsaw, under the supervision of Dr. Jakub Zieliński. 
I would like to thank my academic advisors, the contributors to the LIDC-IDRI dataset, and the open-source community for their invaluable support and resources. 
Special thanks to the PyTorch community for their excellent tools and documentation.

