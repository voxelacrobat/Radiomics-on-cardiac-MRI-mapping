import os
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.cm import get_cmap
from matplotlib import cm
from matplotlib.colors import to_hex
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap, BoundaryNorm
from mycolorpy import colorlist as mcp
from radiomics_tools import normalize_between, remove_substrings_in_list
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

"""
===========================================================================================
@brief      remove_prefix()
@details
@author     MM
@date       24.08.2024
@param[in]  feature_name
@return     void
@note  
===========================================================================================
"""   
def remove_prefix(feature_name):
    prefixes = ["original_shape_", "original_firstorder_", "original_glszm_", 
                "original_glrlm_", "original_gldm_", "original_glcm_", "original_ngtdm_"]
    for prefix in prefixes:
        if feature_name.startswith(prefix):
            return feature_name[len(prefix):]
    return feature_name

"""
===========================================================================================
@brief      remove_postfix()
@details
@author     MM
@date       24.08.2024
@param[in]  class_name
@return     void
@note  
===========================================================================================
"""   
def remove_postfix(class_name):
    postfixes = ["1stOrder", "GLCM", "GLDM", "GLRLM", "GLSZM", "NGTDM", "Shape"]
    for postfix in postfixes:
        if class_name.startswith(postfix):
            class_name = ('{: >8}'.format(class_name))
            return class_name
    return class_name

"""
===========================================================================================
@brief      plot_heatmap_by_class()
@details
@author     MM
@date       24.08.2024
@param[in]  df
@param[in]  _result_path
@param[in]  _mrSeq
@param[in]  disValues
@return     void
@note  
===========================================================================================
"""   
def plot_heatmap_by_class(df, _result_path, _mrSeq, disValues):

    if(_mrSeq=='T1'):
        icc_cols = ['SAX/SR', '4Ch/SR','SAX/HR','4Ch/HR']
        fig = plt.figure(figsize=(10, 12))  
        _bar_height = 0.905
    elif(_mrSeq=='T2'):
        icc_cols = ['SAX', '4Ch']
        fig = plt.figure(figsize=(8,12)) 
        _bar_height = 0.905

    col_good = "#B3DE69" 
    col_moderate = '#FFDE00'
    colors = ['red', col_moderate, col_good, 'green']
    cmap = ListedColormap(colors)
    bounds = [0, 0.5, 0.75, 0.9, 1.0]
    norm = BoundaryNorm(bounds, cmap.N)
    
    # Plot the heatmap with custom color mapping and a fixed size
    ax = sns.heatmap(df, annot=disValues, fmt=".2f", cmap=cmap, norm=norm, linewidths=0.6, cbar=False, annot_kws={"size": 14})
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.title("Excellent and good features on %s maps"%(_mrSeq), pad=20, fontsize = 16, fontweight='bold')
    plt.xticks(ha='center',fontsize = 14, fontweight='bold')
    plt.yticks(fontsize = 14, fontweight='bold')
    plt.tight_layout()

    tmpdf = df[df[icc_cols].isna().any(axis=1)]
    print(tmpdf.head())
    labels_with_values = tmpdf.axes[0].tolist()
    for text in ax.get_yticklabels():
        for best_feature in labels_with_values:
            if text.get_text() == best_feature:  
                text.set_weight('normal')
    
    ## Create the axis for the colorbars
    axl = fig.add_axes([1.01, 0.02, 0.02, _bar_height])
    mycmap = (mpl.colors.ListedColormap(colors))
    # If a ListedColormap is used, the length of the bounds array must be
    # one greater than the length of the color list.  The bounds must be
    # monotonically increasing.
    mynorm = mpl.colors.BoundaryNorm(bounds, mycmap.N)
    cb = fig.colorbar(mpl.cm.ScalarMappable(cmap=mycmap, norm=mynorm),
                cax=axl,
                spacing='proportional',
                orientation='vertical')
    _bound_ticks = bounds
    cb.set_ticklabels(_bound_ticks, fontsize=12) 
      
    if disValues==True:
        filename = "final_%s_best_Robustness_w_values.png"%(_mrSeq)
    else:
        filename = "final_%s_best_Robustness_wo_values.png"%(_mrSeq)

    filepath = os.path.join(_result_path, filename)
    plt.savefig(filepath, pad_inches = 0.2, bbox_inches='tight', transparent=True, dpi=600)
    print(filepath + " saved")

"""
===========================================================================================
@brief      clean_feature_prefix()
@details
@author     MM
@date       24.08.2024
@param[in]  df
@return     df
@note  
===========================================================================================
"""  
def clean_feature_prefix(df):
    # Funktion zum Entfernen der Präfixe
    def remove_prefix(feature_name, prefixes):
        for prefix in prefixes:
            if feature_name.startswith(prefix):
                return feature_name[len(prefix):]
        return feature_name
    
    prefixes = ["original_shape_", "original_firstorder_", "original_glszm_", 
                "original_glrlm_", "original_gldm_", "original_glcm_", "original_ngtdm_"]
    
    df["feature"] = df["feature"].apply(lambda x: remove_prefix(x, prefixes))
    return df

"""
===========================================================================================
@brief      clean_class_postfix()
@details
@author     MM
@date       24.08.2024
@param[in]  df
@return     df
@note  
===========================================================================================
"""  
def clean_class_postfix(df):
    def remove_postfix(class_name, postfixes):
        for postfix in postfixes:
            if class_name.startswith(postfix):
                #class_name = ('{: >15}'.format(class_name))
                class_name = ('{: >15}'.format(class_name))
                return class_name
        return class_name
    
    postfixes = ["First_Order", "GLCM", "GLDM", "GLRLM", "GLSZM", "NGTDM", "2D-Shape"]
    df["class"] = df["class"].apply(lambda x: remove_postfix(x, postfixes))
    return df

"""
===========================================================================================
@brief      create_T1_best_heatmap_from_df()
@details
@author     MM
@date       24.04.2024
@param[in]  df
@param[in]  _root_path
@param[in]  _result_attribute
@param[in]  disValues
@return     void
@note  
===========================================================================================
"""   
def create_T1_best_heatmap_from_df(df, _root_path, _result_attribute, disValues):
    _result_rev_path = os.path.join(_root_path, _result_attribute)
    dfSAX_LR = df[(df['View'] == 'SAX') & (df['Res'] == 'LR') & ((df['robustness'] == 'good') | (df['robustness'] == 'excellent'))]
    df4Ch_LR = df[(df['View'] == '4Ch') & (df['Res'] == 'LR') & ((df['robustness'] == 'good') | (df['robustness'] == 'excellent'))]
    dfSAX_HR = df[(df['View'] == 'SAX') & (df['Res'] == 'HR') & ((df['robustness'] == 'good') | (df['robustness'] == 'excellent'))]
    df4Ch_HR = df[(df['View'] == '4Ch') & (df['Res'] == 'HR') & ((df['robustness'] == 'good') | (df['robustness'] == 'excellent'))]
    merged_df_LR = dfSAX_LR.merge(df4Ch_LR, on=['feature', 'class'], how='outer', suffixes=('_LRsax', '_LR4ch'))
    merged_df_HR = dfSAX_HR.merge(df4Ch_HR, on=['feature', 'class'], how='outer', suffixes=('_HRsax', '_HR4ch'))
    merged_df_LRHR = merged_df_LR.merge(merged_df_HR, on=['feature', 'class'], how='outer', suffixes=('_dfLR', '_dfHR'))
    
    # Rename columns
    merged_df_LRHR.rename(columns={
        'ICC(2,1)_LRsax': 'SAX/SR',
        'ICC(2,1)_LR4ch': '4Ch/SR',
        'ICC(2,1)_HRsax': 'SAX/HR',
        'ICC(2,1)_HR4ch': '4Ch/HR'
    }, inplace=True)
    
    merged_df_LRHR["feature"] = merged_df_LRHR["feature"].str.replace("  ", "")
    merged_df_cleaned = clean_feature_prefix(merged_df_LRHR)
    merged_df_cleaned = clean_class_postfix(merged_df_cleaned)
    df_icc = merged_df_cleaned[['class','feature','SAX/SR', '4Ch/SR','SAX/HR','4Ch/HR']]
    df_icc["label"] = df_icc["feature"] + "" + df_icc["class"]
    heatmap_data = df_icc[["label", 'SAX/SR', '4Ch/SR','SAX/HR','4Ch/HR']].set_index("label")
    heatmap_data_sorted = heatmap_data.iloc[::-1]
    plot_heatmap_by_class(heatmap_data_sorted, _result_rev_path, "T1", disValues)

"""
===========================================================================================
@brief      create_T2_best_heatmap_from_df()
@details
@author     MM
@date       24.04.2024
@param[in]  df
@param[in]  _root_path
@param[in]  _result_attribute
@param[in]  disValues
@return     void
@note  
===========================================================================================
"""   
def create_T2_best_heatmap_from_df(df, _root_path, _result_attribute, disValues):
    _result_rev_path = os.path.join(_root_path, _result_attribute)
    dfSAX = df[(df['View'] == 'SAX') & ((df['robustness'] == 'good') | (df['robustness'] == 'excellent'))]
    df4Ch = df[(df['View'] == '4Ch') & ((df['robustness'] == 'good') | (df['robustness'] == 'excellent'))]
    merged_df = dfSAX.merge(df4Ch, on=['feature', 'class'], how='outer', suffixes=('_df1', '_df2'))
   
    # Rename columns
    merged_df.rename(columns={
        'ICC(2,1)_df1': 'SAX',
        'ICC(2,1)_df2': '4Ch'
    }, inplace=True)
    
    merged_df["feature"] = merged_df["feature"].str.replace("  ", "")
    merged_df_cleaned = clean_feature_prefix(merged_df)
    merged_df_cleaned = clean_class_postfix(merged_df_cleaned)
    df_icc = merged_df_cleaned[['class','feature','SAX', '4Ch']]
    df_icc["label"] = df_icc["feature"] + "" + df_icc["class"]
    heatmap_data = df_icc[["label", "SAX", "4Ch"]].set_index("label")
    heatmap_data_sorted = heatmap_data.iloc[::-1]
    plot_heatmap_by_class(heatmap_data_sorted, _result_rev_path, "T2", disValues)


"""
===========================================================================================
@brief      create_ICC_CCC_Rcorr_T1andT2_from_df()
@details
@author     MM
@date       24.04.2024
@param[in]  dfall1
@param[in]  dfall2
@param[in]  _result_rev_path
@return     void
@note  
===========================================================================================
"""   
def create_ICC_CCC_Rcorr_T1andT2_from_df(dfall1, dfall2, _result_rev_path):
    df1 = dfall1[(dfall1['View'] == 'SAX')]
    df2 = dfall2[(dfall2['View'] == 'SAX')]
    threeParam1= df1[['ICC(2,1)', 'CCC', 'R_pearson']]
    threeParam2= df2[['ICC(2,1)', 'CCC', 'R_pearson']]

    _fontsize = 15
    _xy_ticks = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]
    sns.set_style('darkgrid')  
    fig, ax = plt.subplots(2, 3, figsize=(16,9))
    plt.suptitle(r"Pairwise comparison between reproducibility measures ICC, CCC and $r$", fontsize=18)
    fig.subplots_adjust(wspace=None)

    p1 = sns.scatterplot(data=threeParam1, ax=ax[0,0], x="ICC(2,1)", y="CCC", color="blue", alpha=0.4)
    ax[0,0].set_xlabel(r"ICC", fontsize=_fontsize)
    ax[0,0].set_ylabel(r"CCC", fontsize=_fontsize)
    ax[0,0].set_xticks(_xy_ticks)
    ax[0,0].set_yticks(_xy_ticks)

    p2 = sns.scatterplot(data=threeParam1, ax=ax[0,1], x="ICC(2,1)", y="R_pearson", color="blue", alpha=0.4)
    ax[0,1].set_xlabel(r"ICC", fontsize=_fontsize)
    ax[0,1].set_ylabel(r"$r$", fontsize=_fontsize)
    ax[0,1].set_xticks(_xy_ticks)
    ax[0,1].set_yticks(_xy_ticks)
  
    p3 = sns.scatterplot(data=threeParam1, ax=ax[0,2], x="CCC", y="R_pearson", color="blue", alpha=0.4)
    ax[0,2].set_xlabel(r"CCC", fontsize=_fontsize)
    ax[0,2].set_ylabel(r"$r$", fontsize=_fontsize)
    ax[0,2].set_xticks(_xy_ticks)
    ax[0,2].set_yticks(_xy_ticks)
    p4 = sns.scatterplot(data=threeParam2, ax=ax[1,0], x="ICC(2,1)", y="CCC", color="blue", alpha=0.4)
    ax[1,0].set_xlabel(r"ICC", fontsize=_fontsize)
    ax[1,0].set_ylabel(r"CCC", fontsize=_fontsize)
    ax[1,0].set_xticks(_xy_ticks)
    ax[1,0].set_yticks(_xy_ticks)
    p5 = sns.scatterplot(data=threeParam2, ax=ax[1,1], x="ICC(2,1)", y="R_pearson", color="blue", alpha=0.4)
    ax[1,1].set_xlabel(r"ICC", fontsize=_fontsize)
    ax[1,1].set_ylabel(r"$r$", fontsize=_fontsize)
    ax[1,1].set_xticks(_xy_ticks)
    ax[1,1].set_yticks(_xy_ticks)
    p6 = sns.scatterplot(data=threeParam2, ax=ax[1,2], x="CCC", y="R_pearson", color="blue", alpha=0.4)
    ax[1,2].set_xlabel(r"CCC", fontsize=_fontsize)
    ax[1,2].set_ylabel(r"$r$", fontsize=_fontsize)
    ax[1,2].set_xticks(_xy_ticks)
    ax[1,2].set_yticks(_xy_ticks)
    filename = "pairwise_Scatterplot.png"
    filepath = os.path.join(_result_rev_path, filename)
    plt.savefig(filepath, pad_inches = 0.2, bbox_inches='tight', transparent=False, dpi=600)
    print(filepath + " saved")

    mseT1 = mean_squared_error(threeParam1["CCC"], threeParam1["ICC(2,1)"])
    maeT1 = mean_absolute_error(threeParam1["CCC"], threeParam1["ICC(2,1)"])
    r2T1 = r2_score(threeParam1["CCC"], threeParam1["ICC(2,1)"])
    
    mseT2 = mean_squared_error(threeParam2["CCC"], threeParam2["ICC(2,1)"])
    maeT2 = mean_absolute_error(threeParam2["CCC"], threeParam2["ICC(2,1)"])
    r2T2 = r2_score(threeParam2["CCC"], threeParam2["ICC(2,1)"])

    print("MSE_T1/T2: %.2f/%.2f"%(mseT1, mseT2))
    print("MAE_T1: %.2f/%.2f"%(maeT1, maeT2))
    print("R²_T1: %.2f/%.2f"%(r2T1, r2T2))


