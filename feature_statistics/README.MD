## Radiomic feature anlayser pipeline 

This script collection compares the repeatability of 100 radiomics features applied on cardiac T1 and T2 MRI maps  

The following processing steps are performed:

1. Load and pairweise (repeat 1 and repeat 2) measures 
2. Determine various correlation values between samples and classify features into four groups (poor, moderate, good and excellent)
3. Divide results into feature classes each of the four variations (view/resolution) and represent results as color coded heatmap for each class 
4. Generate heatmaps for subsets of best reproducible features  
5. Compare correlation values 

