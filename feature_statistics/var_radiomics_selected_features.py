selecteted_features = [
    "diagnostics_Image-original_Mean",
    "diagnostics_Image-original_Maximum",
    "diagnostics_Mask-original_VoxelNum",
    "diagnostics_Image-interpolated_Mean", 
    "diagnostics_Image-interpolated_Maximum", 
    "diagnostics_Mask-interpolated_VoxelNum",
    ##
    "original_shape_Elongation",            
    "original_shape_Flatness",              
    "original_shape_LeastAxisLength",       
    "original_shape_MajorAxisLength",       
    "original_shape_Maximum2DDiameterColumn", 
    "original_shape_Maximum2DDiameterRow", 
    "original_shape_Maximum2DDiameterSlice", 
    "original_shape_MinorAxisLength",
    "original_shape_Sphericity",            
    ##
    "original_firstorder_10Percentile",
    "original_firstorder_90Percentile",
    "original_firstorder_Energy",
    "original_firstorder_Entropy",
    "original_firstorder_InterquartileRange",
    "original_firstorder_Kurtosis",
    "original_firstorder_Maximum",
    "original_firstorder_MeanAbsoluteDeviation",
    "original_firstorder_Mean",
    "original_firstorder_Median",
    "original_firstorder_Minimum",
    "original_firstorder_Range",
    "original_firstorder_RobustMeanAbsoluteDeviation",
    "original_firstorder_RootMeanSquared",
    "original_firstorder_Skewness",
    "original_firstorder_TotalEnergy",
    "original_firstorder_Uniformity",
    "original_firstorder_Variance",
    ##
    "original_glcm_Autocorrelation",
    "original_glcm_JointAverage",
    "original_glcm_ClusterProminence",
    "original_glcm_ClusterShade",
    "original_glcm_ClusterTendency",
    "original_glcm_Contrast",
    "original_glcm_Correlation",
    "original_glcm_DifferenceAverage",
    "original_glcm_DifferenceEntropy",
    "original_glcm_DifferenceVariance",
    "original_glcm_JointEnergy",
    "original_glcm_JointEntropy",
    "original_glcm_Imc1",
    "original_glcm_Imc2",
    "original_glcm_Idm",
    "original_glcm_Idmn",
    "original_glcm_Id",
    "original_glcm_Idn",
    "original_glcm_InverseVariance",
    "original_glcm_MaximumProbability",
    "original_glcm_SumEntropy",
    "original_glcm_SumSquares",
    ##
    "original_glrlm_GrayLevelNonUniformity",
    "original_glrlm_GrayLevelNonUniformityNormalized",
    "original_glrlm_GrayLevelVariance",
    "original_glrlm_HighGrayLevelRunEmphasis",
    "original_glrlm_LongRunEmphasis",
    "original_glrlm_LongRunHighGrayLevelEmphasis",
    "original_glrlm_LongRunLowGrayLevelEmphasis",
    "original_glrlm_LowGrayLevelRunEmphasis",
    "original_glrlm_RunEntropy",
    "original_glrlm_RunLengthNonUniformity",
    "original_glrlm_RunLengthNonUniformityNormalized",
    "original_glrlm_RunPercentage",
    "original_glrlm_RunVariance",
    "original_glrlm_ShortRunEmphasis",
    "original_glrlm_ShortRunHighGrayLevelEmphasis",
    "original_glrlm_ShortRunLowGrayLevelEmphasis",
    ##
    "original_glszm_GrayLevelNonUniformity",
    "original_glszm_GrayLevelNonUniformityNormalized",
    "original_glszm_GrayLevelVariance",
    "original_glszm_HighGrayLevelZoneEmphasis",
    "original_glszm_LargeAreaEmphasis",
    "original_glszm_LargeAreaHighGrayLevelEmphasis",
    "original_glszm_LargeAreaLowGrayLevelEmphasis",
    "original_glszm_LowGrayLevelZoneEmphasis",
    "original_glszm_SizeZoneNonUniformity",
    "original_glszm_SizeZoneNonUniformityNormalized",
    "original_glszm_SmallAreaEmphasis",
    "original_glszm_SmallAreaHighGrayLevelEmphasis",
    "original_glszm_SmallAreaLowGrayLevelEmphasis",
    "original_glszm_ZoneEntropy",
    "original_glszm_ZonePercentage",
    "original_glszm_ZoneVariance",
    ##
    "original_gldm_DependenceEntropy",
    "original_gldm_DependenceNonUniformity",
    "original_gldm_DependenceNonUniformityNormalized",
    "original_gldm_DependenceVariance",
    "original_gldm_GrayLevelNonUniformity",
    "original_gldm_GrayLevelVariance",
    "original_gldm_HighGrayLevelEmphasis",
    "original_gldm_LargeDependenceEmphasis",
    "original_gldm_LargeDependenceHighGrayLevelEmphasis",
    "original_gldm_LargeDependenceLowGrayLevelEmphasis",
    "original_gldm_LowGrayLevelEmphasis",
    "original_gldm_SmallDependenceEmphasis",
    "original_gldm_SmallDependenceHighGrayLevelEmphasis",
    "original_gldm_SmallDependenceLowGrayLevelEmphasis",
    ## 
    "original_ngtdm_Busyness",
    "original_ngtdm_Coarseness",
    "original_ngtdm_Complexity",
    "original_ngtdm_Contrast",
    "original_ngtdm_Strength"
    ]


class_selecteted_features = [
    ["2D-Shape", "original_shape_Elongation"],            
    ["2D-Shape", "original_shape_Flatness"],             
    ["2D-Shape", "original_shape_LeastAxisLength"],       
    ["2D-Shape", "original_shape_MajorAxisLength"],       
    ["2D-Shape", "original_shape_Maximum2DDiameterColumn"], 
    ["2D-Shape", "original_shape_Maximum2DDiameterRow"], 
    ["2D-Shape", "original_shape_Maximum2DDiameterSlice"], 
    ["2D-Shape", "original_shape_MinorAxisLength"],
    ["2D-Shape", "original_shape_Sphericity"],            
    ##
    ["First_Order", "original_firstorder_10Percentile"],
    ["First_Order", "original_firstorder_90Percentile"],
    ["First_Order", "original_firstorder_Energy"],
    ["First_Order", "original_firstorder_Entropy"],
    ["First_Order", "original_firstorder_InterquartileRange"],
    ["First_Order", "original_firstorder_Kurtosis"],
    ["First_Order", "original_firstorder_Maximum"],
    ["First_Order", "original_firstorder_MeanAbsoluteDeviation"],
    ["First_Order", "original_firstorder_Mean"],
    ["First_Order", "original_firstorder_Median"],
    ["First_Order", "original_firstorder_Minimum"],
    ["First_Order", "original_firstorder_Range"],
    ["First_Order", "original_firstorder_RobustMeanAbsoluteDeviation"],
    ["First_Order", "original_firstorder_RootMeanSquared"],
    ["First_Order", "original_firstorder_Skewness"],
    ["First_Order", "original_firstorder_TotalEnergy"],
    ["First_Order", "original_firstorder_Uniformity"],
    ["First_Order", "original_firstorder_Variance"],
    ##
    ["GLCM", "original_glcm_Autocorrelation"],
    ["GLCM", "original_glcm_JointAverage"],
    ["GLCM", "original_glcm_ClusterProminence"],
    ["GLCM", "original_glcm_ClusterShade"],
    ["GLCM", "original_glcm_ClusterTendency"],
    ["GLCM", "original_glcm_Contrast"],
    ["GLCM", "original_glcm_Correlation"],
    ["GLCM", "original_glcm_DifferenceAverage"],
    ["GLCM", "original_glcm_DifferenceEntropy"],
    ["GLCM", "original_glcm_DifferenceVariance"],
    ["GLCM", "original_glcm_JointEnergy"],
    ["GLCM", "original_glcm_JointEntropy"],
    ["GLCM", "original_glcm_Imc1"],
    ["GLCM", "original_glcm_Imc2"],
    ["GLCM", "original_glcm_Idm"],
    ["GLCM", "original_glcm_Idmn"],
    ["GLCM", "original_glcm_Id"],
    ["GLCM", "original_glcm_Idn"],
    ["GLCM", "original_glcm_InverseVariance"],
    ["GLCM", "original_glcm_MaximumProbability"],
    ["GLCM", "original_glcm_SumEntropy"],
    ["GLCM", "original_glcm_SumSquares"],
    ##
    ["GLRLM", "original_glrlm_GrayLevelNonUniformity"],
    ["GLRLM", "original_glrlm_GrayLevelNonUniformityNormalized"],
    ["GLRLM", "original_glrlm_GrayLevelVariance"],
    ["GLRLM", "original_glrlm_HighGrayLevelRunEmphasis"],
    ["GLRLM", "original_glrlm_LongRunEmphasis"],
    ["GLRLM", "original_glrlm_LongRunHighGrayLevelEmphasis"],
    ["GLRLM", "original_glrlm_LongRunLowGrayLevelEmphasis"],
    ["GLRLM", "original_glrlm_LowGrayLevelRunEmphasis"],
    ["GLRLM", "original_glrlm_RunEntropy"],
    ["GLRLM", "original_glrlm_RunLengthNonUniformity"],
    ["GLRLM", "original_glrlm_RunLengthNonUniformityNormalized"],
    ["GLRLM", "original_glrlm_RunPercentage"],
    ["GLRLM", "original_glrlm_RunVariance"],
    ["GLRLM", "original_glrlm_ShortRunEmphasis"],
    ["GLRLM", "original_glrlm_ShortRunHighGrayLevelEmphasis"],
    ["GLRLM", "original_glrlm_ShortRunLowGrayLevelEmphasis"],
    ##
    ["GLSZM", "original_glszm_GrayLevelNonUniformity"],
    ["GLSZM", "original_glszm_GrayLevelNonUniformityNormalized"],
    ["GLSZM", "original_glszm_GrayLevelVariance"],
    ["GLSZM", "original_glszm_HighGrayLevelZoneEmphasis"],
    ["GLSZM", "original_glszm_LargeAreaEmphasis"],
    ["GLSZM", "original_glszm_LargeAreaHighGrayLevelEmphasis"],
    ["GLSZM", "original_glszm_LargeAreaLowGrayLevelEmphasis"],
    ["GLSZM", "original_glszm_LowGrayLevelZoneEmphasis"],
    ["GLSZM", "original_glszm_SizeZoneNonUniformity"],
    ["GLSZM", "original_glszm_SizeZoneNonUniformityNormalized"],
    ["GLSZM", "original_glszm_SmallAreaEmphasis"],
    ["GLSZM", "original_glszm_SmallAreaHighGrayLevelEmphasis"],
    ["GLSZM", "original_glszm_SmallAreaLowGrayLevelEmphasis"],
    ["GLSZM", "original_glszm_ZoneEntropy"],
    ["GLSZM", "original_glszm_ZonePercentage"],
    ["GLSZM", "original_glszm_ZoneVariance"],
    ##
    ["GLDM", "original_gldm_DependenceEntropy"],
    ["GLDM", "original_gldm_DependenceNonUniformity"],
    ["GLDM", "original_gldm_DependenceNonUniformityNormalized"],
    ["GLDM", "original_gldm_DependenceVariance"],
    ["GLDM", "original_gldm_GrayLevelNonUniformity"],
    ["GLDM", "original_gldm_GrayLevelVariance"],
    ["GLDM", "original_gldm_HighGrayLevelEmphasis"],
    ["GLDM", "original_gldm_LargeDependenceEmphasis"],
    ["GLDM", "original_gldm_LargeDependenceHighGrayLevelEmphasis"],
    ["GLDM", "original_gldm_LargeDependenceLowGrayLevelEmphasis"],
    ["GLDM", "original_gldm_LowGrayLevelEmphasis"],
    ["GLDM", "original_gldm_SmallDependenceEmphasis"],
    ["GLDM", "original_gldm_SmallDependenceHighGrayLevelEmphasis"],
    ["GLDM", "original_gldm_SmallDependenceLowGrayLevelEmphasis"],
    ## 
    ["NGTDM", "original_ngtdm_Busyness"],
    ["NGTDM", "original_ngtdm_Coarseness"],
    ["NGTDM", "original_ngtdm_Complexity"],
    ["NGTDM", "original_ngtdm_Contrast"],
    ["NGTDM", "original_ngtdm_Strength"]]