# topology_based_framework
this repository includes codes to implement the approach in the paper 
"Early- and in-season crop type mapping without current-year ground truth: Generating labels from historical information via a topology-based approach" (DOI: 10.1016/j.rse.2022.112994)  
Step 1. segment satellite imagery into image patches and build training, validation and testing dataset using <b>Segmentation.py</b>  
Step 2. generate heat maps and their targets using <b>Generate_heat_map.py</b>  
Step 3. (optional) visually check your type-I and type-II heat maps and make revisions if there's any  
Step 4. train deep learning models using <b>Execuation.py</b>  

