# Medify
## Problem Statement
1. Identifying the right diagnosis for a given person at the earliest is one of the fundamental problems faced by healthcare providers.

2. For certain diseases like Cancer, late diagnosis threatens the chances of survival, chances of healthy lifestyle and also increases treatment costs.

3. Then there are problems such as is a lack of certified medicine practitioners,

4. Affordability of medical expenditures,

5. Significant amount of time taken to diagnose that particular disease. 

6. The physical access to hospitals is still the major barrier to both preventive and curative health services, forming the major differences between the Rural and the Urban India

## Medify.AI is a project which uses advanced techniques like AI to detect sudden health issues
 

## WHY MEDIFY.AI? 
 
Need to reduce the diagnosing time taken
The challenge here is that a trained human eye of a medical professional cannot identify subtle signs on these tests.
Low Accessibility and affordability:
The physical access to hospitals is still the major barrier to both preventive and curative health services, and also the major differences between the Rural and the Urban India. Besides, affordability of medical expenditures is one of the biggest problems faced by patients.

## Solution
Our web application using CNNs and ML that facilitates to test and predict the health risk of the patient based on his/her reports for which we have trained the models on unbiased and distributed datasets. 
To use our web-application, users have to simply upload their MRI scans depending upon the type of disease or cancer that they want to get diagnosed for. 
After taking the uploaded image as input, the CNNs model at backend will generate required heatmaps and predict whether the user has that disease or not. 
If yes, then the model would predict the stage of that disease/cancer with the probabilities. 
The model uses EfficientNet architecture for convolutional neural networks with the state of art techniques to predict results for different scans and inputs.

## Steps To Run the app on local host:

1. Create a virtual environment with the dependencies listed in requirements.txt
2. Make a subfolder and clone rest of the files there
3. Activate the virtual environment with command on root dir : source your_env_name/bin/activate
4. Change the location to subfolder and run app.py using python3 app.py


