import os
import seaborn as sns        
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
import pandas as pd
from  matplotlib.colors import ListedColormap, BoundaryNorm

 # Define a function to remove prefixes from feature names
def remove_prefix(feature_name):
    prefixes = ["original_shape_", "original_firstorder_", "original_glszm_", 
                "original_glrlm_", "original_gldm_", "original_glcm_", "original_ngtdm_"]
    for prefix in prefixes:
        if feature_name.startswith(prefix):
            return feature_name[len(prefix):]
    return feature_name

def remove_postfix(class_name):
    postfixes = ["1stOrder", "GLCM", "GLDM", "GLRLM", "GLSZM", "NGTDM", "Shape"]
    for postfix in postfixes:
        if class_name.startswith(postfix):
            class_name = ('{: >8}'.format(class_name))
            return class_name
    return class_name

"""
===========================================================================================
@brief      plot_multi_heatmap()
@details
@author     MM
@date       28.08.2024
@param[in]  _df
@param[in]  _pivot_combined_sorted
@param[in]  _result_path
@param[in]  _mrSeq
@param[in]  disValues
@return     void
@note  
===========================================================================================
"""   
def plot_multi_heatmap(_df, _pivot_combined_sorted, _result_path, _mrSeq, disValues):
    # Define the custom colormap based on ICC value ranges
    colors = ['red', '#FFDE00', "#B3DE69", 'green']
    cmap = ListedColormap(colors)
    bounds = [0, 0.5, 0.75, 0.9, 1.0]
    norm = BoundaryNorm(bounds, cmap.N)

    if(_mrSeq=='T1'):
        fig, axs = plt.subplots(4, 2, figsize=(25, 28), sharex=False)  
        linewidth = 0.2, 
    elif(_mrSeq=='T2'):
        fig, axs = plt.subplots(4, 2, figsize=(25, 28), sharex=False)  # 1 Zeile, 3 Spalten
        linewidth = 0.2, 

    # Let's plot heatmaps for each class
    i = 0
    j = 0
    k = 0
    for class_name in _df['class'].unique():
        if(i==4):
            i = 0
            j += 1 
             
        # Apply the function to remove prefixes from the feature names in the pivot table
        pivot_combined_sorted_cleaned = _pivot_combined_sorted.rename(remove_prefix, level='feature')
        class_data = pivot_combined_sorted_cleaned.loc[class_name]
        class_data = pd.DataFrame(class_data)
        df_reversed = class_data[class_data.columns[::-1]] ## umsortieren der spalten
        df_reversed = df_reversed.iloc[::-1] ## umsortieren der zeilen
        ax = sns.heatmap(df_reversed, ax=axs[i,j], annot=disValues, fmt=".2f", cmap=cmap, norm=norm, linewidths=linewidth,  cbar=False, annot_kws={"size": 20})
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticklabels(df_reversed.columns, ha='center', fontsize=20)
        ax.set_yticklabels(ax.get_yticklabels(), size = 20)
        ax.tick_params(axis='y', labelrotation=0)
        ax.set_title(f"{class_name}", pad=20, fontsize=21, fontweight='bold')
        i += 1 
        k += 1

    axs[3,1].set_axis_off() # Disable last plot
    plt.tight_layout(rect=[0, 0, .9, 1])
    if disValues==True:
        filename = "%s_multi_heatmap_all_w_values.png"%(_mrSeq)
    else:
        filename = "%s_multi_heatmap_all_wo_values.png"%(_mrSeq)

    filepath = os.path.join(_result_path, filename)
    plt.savefig(filepath, pad_inches = 0.1, bbox_inches='tight', transparent=True, dpi=600) 
    print(filepath + " saved")

"""
===========================================================================================
@function   create_heatmaps_for_all_T1_robustnesses_multi_plot()
@details
@author     MM
@date       24.08.2024
@param[in]  _df
@param[in]  _root_path
@param[in]  _result_attribute
@return     void
@note  
===========================================================================================
"""   
def create_heatmaps_for_all_T1_robustnesses_multi_plot(df, _root_path, _result_attribute):

    _result_path = os.path.normpath(os.path.join(_root_path,_result_attribute))
    print(df.head())
    df['Res+View'] = df['View'] + '+' + df['Res']

    df_LR_SAX = df[(df['Res'] == 'LR') & (df['View'] == 'SAX')]
    df_LR_4Ch = df[(df['Res'] == 'LR') & (df['View'] == '4Ch')]
    df_HR_SAX = df[(df['Res'] == 'HR') & (df['View'] == 'SAX')]
    df_HR_4Ch = df[(df['Res'] == 'HR') & (df['View'] == '4Ch')]

    # First, we need to create a combined dataframe for all Res+View combinations
    df_combined = pd.concat([df_LR_SAX, df_LR_4Ch, df_HR_SAX, df_HR_4Ch])
    # Create a pivot table where rows are features, columns are Res+View, and values are ICC, sorted by class
    pivot_combined = df_combined.pivot_table(index=['class', 'feature'], columns='Res+View', values='ICC(2,1)', aggfunc='first')
    # Sort the pivot table by class
    pivot_combined_sorted = pivot_combined.sort_index(level='class')

     # Rename columns
    pivot_combined_sorted.rename(columns={
        'SAX+LR': 'SAX/SR',
        '4Ch+LR': '4Ch/SR',
        'SAX+HR': 'SAX/HR',
        '4Ch+HR': '4Ch/HR'
    }, inplace=True)

    # Spalten B und C tauschen
    df_swapped = pivot_combined_sorted.copy()
    # Spalten umsortieren, indem B und C getauscht werden
    df_swapped[['SAX/HR', '4Ch/SR']] = df_swapped[['4Ch/SR', 'SAX/HR']]
    df_swapped.rename(columns={
        '4Ch/SR': 'SAX/HR',
        'SAX/HR': '4Ch/SR'
    }, inplace=True)

    plot_multi_heatmap(df, df_swapped, _result_path, "T1", True)
    plot_multi_heatmap(df, df_swapped, _result_path, "T1", False)

"""
===========================================================================================
@function   create_heatmaps_for_all_T2_robustnesses_multi_plot()
@details
@author     MM
@date       28.08.2024
@param[in]  _df
@param[in]  _root_path
@param[in]  _result_attribute
@return     void
@note  
===========================================================================================
"""   
def create_heatmaps_for_all_T2_robustnesses_multi_plot(df, _root_path, _result_attribute):
    _result_path = os.path.normpath(os.path.join(_root_path,_result_attribute))
    print(df.head())
    df_SAX = df[(df['View'] == 'SAX')]
    df_4Ch = df[(df['View'] == '4Ch')]
    
    # First, we need to create a combined dataframe for all Res+View combinations
    df_combined = pd.concat([df_SAX, df_4Ch])
    # Create a pivot table where rows are features, columns are Res+View, and values are ICC, sorted by class
    pivot_combined = df_combined.pivot_table(index=['class', 'feature'], columns='View', values='ICC(2,1)', aggfunc='first')
    # Sort the pivot table by class
    pivot_combined_sorted = pivot_combined.sort_index(level='class')

    plot_multi_heatmap(df, pivot_combined_sorted, _result_path, "T2", True)
    plot_multi_heatmap(df, pivot_combined_sorted, _result_path, "T2", False)


