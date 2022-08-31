# topology_based_framework
this repository includes codes to implement the approach in the paper 
"Early- and in-season crop type mapping without current-year ground truth: Generating labels from historical information via a topology-based approach" (DOI: 10.1016/j.rse.2022.112994)  
Step 1. segment satellite imagery into image patches and build training, validation and testing dataset using <b>Segmentation.py</b>  
Step 2. generate heat maps and their targets using <b>Generate_heat_map.py</b>  
Step 3. (optional) visually check your type-I and type-II heat maps and make revisions if there's any  
Step 4. train deep learning models using <b>Execuation.py</b>
## testing data
You can access the toy data via the link: https://drive.google.com/drive/folders/1Vu5SRpjzIuJQNsXQmklYFmTBu7_OJIeB?usp=sharing  
-- the toy data contains "raw data", "segmentation", "training", "validation", "testing"  
-- the "raw data" folder contains 24 Sentinel-2 composite (every 5 days from June to September), 2017 and 2018 CDL, and binary historcal CDL (0 for areas that have never been planted with corn/soybeans, 1 for areas that have been planted one of corn/soybeans)
-- the "segmentation" folder contains segmented image patch from all imagery in the "raw data" folder, which is actually the results of <b>Segmentation.py</b>  
-- "training", "validation", "testing" folders are image patches assigned for training, validation and testing purposes
