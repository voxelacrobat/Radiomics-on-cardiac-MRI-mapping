import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import os
import pandas as pd
from radiomics_tools import normalize_between, normalize_relative_percent

col_poor = "#FB8072" 
col_moderate = "#FDB462" 
col_good = "#80B1D3" 
col_excellent = "#B3DE69"

glob_robustness_colors = [col_excellent, col_good, col_moderate, col_poor]   ## "#FF0B04", "#4374B3"
glob_robustness_pair_colors = [col_poor, col_moderate, col_good, col_excellent]   ## "#FF0B04", "#4374B3"

"""
    ===========================================================================================
    @fn         my_bland_altman_plot()
    @details
    @author     MM
    @date       12.01.2023
    @param[in]  filepath - path to excel file
    @param[in]  mha_array - mah file information matrix
    @note  
    ===========================================================================================
    """     
def bland_altman_plot(vector1, 
                         vector2,
                         _figurepath,
                         _description,
                         title1="LoE", 
                         axLabels1=["R1 T1 [ms]", "R2 T1 [ms]"], 
                         title2="Bland altman",
                         axLabels2=["T1 [ms]", "ΔT1 [ms]"], 
                         _labelfontsize=22):
   
        data1 = np.array(vector1).astype(np.float32)
        data2 = np.array(vector2).astype(np.float32)
        mean      = np.mean([data1, data2], axis=0)
        diff      = data1 - data2                   # Difference between data1 and data2
        md        = np.mean(diff)                   # Mean of the difference
        #sd        = np.std(diff, axis=0)            # Standard deviation of the difference

        # Average difference (aka the bias)
        bias = md
        # Sample standard deviation
        sd        = np.std(diff, ddof=1)            # Standard deviation of the difference
        upper_loa = bias + 2 * sd
        lower_loa = bias - 2 * sd

        n = len(data1)
        # Variance
        var = sd**2
        # Standard error of the bias
        se_bias = np.sqrt(var / n)
        # Standard error of the limits of agreement
        se_loas = np.sqrt(3 * var / n)
        # Endpoints of the range that contains 95% of the Student’s t distribution
        t_interval = stats.t.interval(confidence=0.95, df=n - 1)
        # Confidence intervals
        ci_bias = bias + np.array(t_interval) * se_bias
        ci_upperloa = upper_loa + np.array(t_interval) * se_loas
        ci_lowerloa = lower_loa + np.array(t_interval) * se_loas

        #title_str = "Manually/automatic segmented ROIs Original vs Resized | %s | %dx%d | ROIs=%d | %s"%(self.Resolution,self.X_shape,self.X_shape, len(data1), self.kind_interpol)
        fig = plt.figure(figsize=(16,9))
        ax = plt.axes()
        plt.title(title1, fontsize = 24)
        plt.xlabel(axLabels1[0],fontsize=_labelfontsize)
        plt.ylabel(axLabels1[1],fontsize=_labelfontsize)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.grid()
        
        # Scatter plot
        # ax.scatter(
        #     df['Wright Large'], df['Wright Mini'],
        #     c='k', s=20, alpha=0.6, marker='o'
        # )
        #ax.scatter(data1, data2, c='k', s=80, alpha=0.6, marker='o')
        plt.scatter(data1, data2, color="royalblue", alpha=0.6, s=80) # cornflowerblue
        # Get axis limits
        left, right = plt.xlim()
        # Set axis limits
        ax.set_xlim(left, right)
        ax.set_ylim(left, right)
        # Reference line
        ax.plot([0, right], [0, right], c='grey', ls='--', label='Line of Equality')
        # Set aspect ratio
        ax.set_aspect('equal')
        # Legend
        ax.legend(frameon=False)
        # Show plot
        filename1 = "%s_line_of_equality.png"%(_description)
        figurepath = os.path.join(_figurepath, filename1) 
        plt.savefig(figurepath, dpi=600)
        print(filename1 + " saved")
        #plt.show()

        ## ===================================================================================================================================
        fig = plt.figure(figsize=(16,9))
        ax = plt.axes()
        #fig.suptitle(title, fontsize = 40)
        plt.title(title2, fontsize = 24)
        plt.xlabel(axLabels2[0])
        plt.ylabel(axLabels2[1])
        
        plt.scatter(mean, diff, color="royalblue", alpha=0.6, s=80) # cornflowerblue
        #plt.scatter(mean, diff, c='k', s=20, alpha=0.6, marker='o')
        plt.axhline(md,           color='blue', linestyle='--')
        plt.axhline(md + 1.96*sd, color='red', linestyle='--')
        plt.axhline(md - 1.96*sd, color='red', linestyle='--')

        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.grid()
        
        # Get axis limits
        left, right = plt.xlim()
        bottom, top = plt.ylim()
        # Set y-axis limits
        max_y = max(abs(bottom), abs(top))
        ax.set_ylim(-max_y * 1.1, max_y * 1.1)
        ax.yaxis.get_label().set(fontsize=_labelfontsize)  
    
        # Set x-axis limits
        domain = right - left
        ax.set_xlim(left, left + domain * 1.1)
        ax.xaxis.get_label().set(fontsize=_labelfontsize)
        # Add the annotations
        ax.annotate('+1.96×SD', (right, upper_loa), (0, 35), textcoords='offset pixels', fontsize=15)
        ax.annotate(f'{upper_loa:+4.2f}', (right, upper_loa), (0, -115), textcoords='offset pixels', fontsize=14)
        ax.annotate('Bias', (right, bias), (0, 35), textcoords='offset pixels', fontsize=15)
        ax.annotate(f'{bias:+4.2f}', (right, bias), (0, -115), textcoords='offset pixels', fontsize=15)
        ax.annotate('-1.96×SD', (right, lower_loa), (0, 35), textcoords='offset pixels', fontsize=15)
        ax.annotate(f'{lower_loa:+4.2f}', (right, lower_loa), (0, -115), textcoords='offset pixels', fontsize=14)
        
        #title_str = "%s | %s | %dx%d | ROIs=%d "%(title, self.Resolution,self.X_shape,self.X_shape, len(data1))
        
        filename2 = "%s_bland_altman.png"%(_description)
        figurepath = os.path.join(_figurepath, filename2) 
        plt.savefig(figurepath, dpi=600)
        print(filename2 + " saved")
        #plt.show()
        #plt.figure().clear()
        plt.close("all")
        plt.cla()
        plt.clf()


"""
===========================================================================================
@fn         plot_robustness()
@details
@author     MM
@date       12.01.2023
@param[in]  filepath - path to excel file
@param[in]  mha_array - mah file information matrix
@note  
===========================================================================================
"""     
def plot_robustness(df, _path, _title="Analysis on ..."):
    plt.figure(figsize=(6,6), dpi=300)
    labels = ['poor', 'moderate', 'good', 'excellent']
    _poor = df[df['robustness'] == 'poor']
    _moderate = df[df['robustness'] == 'moderate']
    _good = df[df['robustness'] == 'good']
    _excellent = df[df['robustness'] == 'excellent']
    data = [len(_poor), len(_moderate), len(_good), len(_excellent)]
    #colors = sns.color_palette('Set2')[0:5]
    fig = plt.pie(x=data, labels=labels, colors=glob_robustness_pair_colors, startangle=180, autopct='%.0f%%')
    plt.title(_title)
    plt.savefig(_path, dpi=300)
    print(_path + " saved")
    #plt.show()
    #plt.figure().clear()
    plt.close("all")
    plt.cla()
    plt.clf()
    #plt.show()


"""
    ===========================================================================================
    @fn         bland_altman_parameter()
    @details
    @author     MM
    @date       12.01.2023
    @param[in]  vector1
    @param[in]  vector2
    @note  
    ===========================================================================================
    """     
def bland_altman_parameter(vector1, vector2):
   
    data1 = np.array(vector1).astype(np.float32)
    data2 = np.array(vector2).astype(np.float32)
    diff      = data1 - data2                   # Difference between data1 and data2
    md        = np.mean(diff)                   # Mean of the difference
    bias = md
    # Sample standard deviation
    sd        = np.std(diff, ddof=1)            # Standard deviation of the difference
    upper_loa = bias + 2 * sd
    lower_loa = bias - 2 * sd
    d_min = min(diff)
    d_max = max(diff)
    diff_norm = ((diff-d_min)/(d_max-d_min))*100
    md_norm        = np.mean(diff_norm)                   # Mean of the difference
    # Average difference (aka the bias)
    bias_norm = md_norm
    # Sample standard deviation
    sd_norm = np.std(diff_norm, ddof=1)            # Standard deviation of the difference
    upper_loa_norm = bias_norm + 2 * sd_norm
    lower_loa_norm = bias_norm - 2 * sd_norm
    return [bias_norm, upper_loa_norm, lower_loa_norm]


"""
===========================================================================================
@fn         plot_heatmap()
@details
@author     MM
@date       12.01.2023
@param[in]  df - dataframe
@param[in]  _path - output path
@param[in]  _title - optional title
@note  
===========================================================================================
"""   
def plot_heatmap(df, _path, _title="Analysis on ..."):
    plt.figure(figsize=(20,8), dpi=150)
    sns.heatmap(df.corr(), cmap='hot', annot=True, vmin=-1.0, vmax=1.0)
    #plt.title(_title)
    plt.savefig(_path, dpi=300)
    print(_path + " saved")
    #plt.show()
    #plt.figure().clear()
    plt.close("all")
    plt.cla()
    plt.clf()
    #plt.show()

"""
===========================================================================================
@fn         plot_pairplot()
@details
@author     MM
@date       12.01.2023
@param[in]  df - dataframe
@param[in]  _path - output path
@param[in]  _title - optional title
@note  
===========================================================================================
"""   
def plot_pairplot(df, _path, _title="Analysis on ..."):

    sns.set_palette(sns.color_palette(glob_robustness_colors))
    plt.figure()
    sns.pairplot(df, hue='robustness', hue_order = ['excellent', 'good', 'moderate', 'poor']) ##, palette=[]
    #plt.title(_title)
    plt.savefig(_path, dpi=300)
    print(_path + " saved")
    #plt.show()
    #plt.figure().clear()
    plt.close("all")
    plt.cla()
    plt.clf()
    #plt.show()

"""
===========================================================================================
@fn         pie_class_robustness_plots()
@details
@author     MM
@date       12.01.2023
@param[in]  df - dataframe
@param[in]  _path - output path
@param[in]  _title - optional title
@note  
===========================================================================================
"""   
def pie_class_robustness_plots(df_classes, _path, _title="Robustness of features classes on ..."):
        
    num_classes = len(df_classes)  
    labels = ['excellent', 'good', 'moderate', 'poor']
    fig, axes = plt.subplots(1, num_classes, figsize=(16,3))
    
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid()
    fig.suptitle(_title, fontsize = 20) 
    fig.subplots_adjust(wspace=None)
    sns.set_style("darkgrid")
    
    _textprops = {"fontsize":10} # Font size of text in pie chart
    i = 0
    for _item in df_classes:
        _class = _item[0]
        _df = _item[1]
        _poor = _df[_df['robustness'] == 'poor']
        _moderate = _df[_df['robustness'] == 'moderate']
        _good = _df[_df['robustness'] == 'good']
        _excellent = _df[_df['robustness'] == 'excellent']

        ##_data = [len(_poor), len(_moderate), len(_good), len(_excellent)]
        _data = []
        _mycolors = []
        if(len(_poor)>0):
             _data.append(len(_poor))
             _mycolors.append(col_poor)
       
        if(len(_moderate)>0):
             _data.append(len(_moderate))
             _mycolors.append(col_moderate)

        if(len(_good)>0):
             _data.append(len(_good))
             _mycolors.append(col_good)

        if(len(_excellent)>0):
             _data.append(len(_excellent))
             _mycolors.append(col_excellent)
        
        axes[i].pie(x=_data, colors=_mycolors, autopct='%1.1f%%', radius=1.4, textprops=_textprops)
        axes[i].set_title(_class, pad=10)
        i = i + 1
   
    name_to_color = {name: color for name, color in zip(labels, glob_robustness_colors)}
    handles = [plt.Rectangle((0, 0), 0, 0, color=name_to_color[name], label=name) for name in name_to_color]
    axes[3].legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=False, ncol=4)
   
    plt.savefig(_path, pad_inches = 0.1, bbox_inches='tight', transparent=True, dpi=600)
    print(_path + " saved")
    print(_path)
    plt.close("all")
    plt.cla()
    plt.clf()

"""
===========================================================================================
@fn         stacked_bar_class_robustness_absolute_plots()
@details
@author     MM
@date       12.01.2023
@param[in]  df - dataframe
@param[in]  _path - output path
@param[in]  _title - optional title
@note  
===========================================================================================
"""   
def stacked_bar_class_robustness_absolute_plots(df_classes, _path, _title="Robustness of features classes on ..."):
        
    num_classes = len(df_classes)  
    robustness = ['poor', 'moderate' , 'good', 'excellent']
    results = {
        '2D-Shape': [],
        'First_Order': [],
        'GLCM': [],
        'GLRLM': [],
        'GLSZM': [],
        'GLDM': [],
        'NGTDM': []
    }
    
    i = 0
    for _item in df_classes:
        _class = _item[0]
        _df = _item[1]
        _poor = _df[_df['robustness'] == 'poor']
        _moderate = _df[_df['robustness'] == 'moderate']
        _good = _df[_df['robustness'] == 'good']
        _excellent = _df[_df['robustness'] == 'excellent']

        ##_data = [len(_poor), len(_moderate), len(_good), len(_excellent)]
        _data = []
        if(len(_poor)>0):
             _data.append(len(_poor))
             results[_class].append(len(_poor))
        else:
            results[_class].append(0)

        if(len(_moderate)>0):
            _data.append(len(_moderate))
            results[_class].append(len(_moderate))
        else:
            results[_class].append(0)

        if(len(_good)>0):
            _data.append(len(_good))
            results[_class].append(len(_good))
        else:
            results[_class].append(0)

        if(len(_excellent)>0):
            _data.append(len(_excellent))
            results[_class].append(len(_excellent))
        else:
            results[_class].append(0)

        i = i + 1
       
    _pdres = pd.DataFrame(results)
    colors = ['#e74c3c', '#f1c40f',  "#B3DE69" , 'green']  

    fig, ax = plt.subplots(figsize=(10, 6), gridspec_kw={'wspace': 0, 'hspace': 0})  
    for i, column in enumerate(_pdres.columns):
        for j in range(len(_pdres[column])):
            value = _pdres[column][j]
            if value > 0:
                ax.bar(column, value, color=colors[j], bottom=np.sum(_pdres[column][:j]), edgecolor='none', width=0.9)  

    totals = _pdres.sum(axis=0)
    y_offset = 0.5
    for i, total in enumerate(totals):
        ax.text(totals.index[i], total + y_offset, round(total), ha='center', weight='bold', fontsize=18)

    # negative offset.
    y_offset = -0.85
    for bar in ax.patches:
        if bar.get_facecolor()==(0.0, 0.5019607843137255, 0.0, 1.0):
            textcolor = 'white'
        else:
            textcolor = 'black'

        ax.text(
            # Put the text in the middle of each bar. get_x returns the start
            # so we add half the width to get to the middle.
            bar.get_x() + bar.get_width() / 2,
            # Vertically, add the height of the bar to the start of the bar,
            # along with the offset.
            bar.get_height() + bar.get_y() + y_offset,
            # This is actual value we'll show.
            round(bar.get_height()),
            # Center the labels and style them a bit.
            ha='center',
            color=textcolor,
            weight='bold',
            size=16
        )

    # Entferne die x-Achsenbeschriftung und Ticks
    ax.set_ylabel('Number of features', fontsize=18)
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_xticks(np.arange(len(_pdres.columns[::-1])))
    ax.set_xticklabels(_pdres.columns, fontsize = 16)
    ax.set_title(_title, fontsize = 18, fontweight="bold")
    name_to_color = {name: color for name, color in zip(robustness, colors)}
    handles = [plt.Rectangle((0, 0), 0, 0, color=name_to_color[name], label=name) for name in name_to_color]
 
    plt.tight_layout()
    plt.savefig(_path, pad_inches = 0.1, bbox_inches='tight', transparent=True, dpi=600)
    print(_path + " saved")
    plt.close("all")
    plt.cla()
    plt.clf()

"""
===========================================================================================
@fn         stacked_barh_class_robustness_plots_percent()
@details
@author     MM
@date       12.01.2023
@param[in]  df - dataframe
@param[in]  _path - output path
@param[in]  _title - optional title
@note  
===========================================================================================
"""      
def stacked_barh_class_robustness_plots_percent(df_classes, _path, _title="Robustness of features classes on ..."):
    robustness = ['poor', 'moderate' , 'good', 'excellent']
    results = {
        '2D-Shape': [],
        'First_Order': [],
        'GLCM': [],
        'GLRLM': [],
        'GLSZM': [],
        'GLDM': [],
        'NGTDM': []
    }
    
    i = 0
    for _item in df_classes:
        _class = _item[0]
        _df = _item[1]
        _poor = _df[_df['robustness'] == 'poor']
        _moderate = _df[_df['robustness'] == 'moderate']
        _good = _df[_df['robustness'] == 'good']
        _excellent = _df[_df['robustness'] == 'excellent']

        _data = []
        if(len(_poor)>0):
             _data.append(len(_poor))
             results[_class].append(len(_poor))
        else:
            results[_class].append(0)

        if(len(_moderate)>0):
            _data.append(len(_moderate))
            results[_class].append(len(_moderate))
        else:
            results[_class].append(0)

        if(len(_good)>0):
            _data.append(len(_good))
            results[_class].append(len(_good))
        else:
            results[_class].append(0)

        if(len(_excellent)>0):
            _data.append(len(_excellent))
            results[_class].append(len(_excellent))
        else:
            results[_class].append(0)

        i = i + 1
    
    for _item in df_classes:
        _class = _item[0]
        _sum = np.array(results[_class]).sum()
        results[_class] = results[_class]/_sum*100
        results[_class] = [ round(elem, 1) for elem in results[_class] ]
    
    _pdres = pd.DataFrame(results)
    colors = ['#e74c3c', '#f1c40f',  "#B3DE69", 'green'] 
    fig, ax = plt.subplots(figsize=(10, 6), gridspec_kw={'wspace': 0, 'hspace': 0})  
    
    for i, column in enumerate(_pdres.columns[::-1]):
        for j in range(len(_pdres[column])):
            value = _pdres[column][j]
            if value > 0:
                ax.barh(column, value, color=colors[j], left=np.sum(_pdres[column][:j]), edgecolor='none', height=0.9)  
                ax.text(np.sum(_pdres[column][:j]) + value / 2, i, f'{value:.1f}%', ha='center', va='center', color='black', fontsize = 14)

    ax.set_xlabel('')
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_yticks(np.arange(len(_pdres.columns[::-1])))
    ax.set_yticklabels(_pdres.columns[::-1], fontsize = 14)
    ax.set_title(_title, fontsize = 18, fontweight="bold")
    name_to_color = {name: color for name, color in zip(robustness, colors)}
    handles = [plt.Rectangle((0, 0), 0, 0, color=name_to_color[name], label=name) for name in name_to_color]
    ax.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, -0.1), shadow=False, ncol=4, fontsize=14)

    plt.tight_layout()
    plt.savefig(_path, pad_inches = 0.1, bbox_inches='tight', transparent=True, dpi=600)
    print(_path + " saved")
    #plt.show()
    plt.close("all")
    plt.cla()
    plt.clf()

"""
===========================================================================================
@fn         stacked_barh_class_robustness_plots_absolute()
@details
@author     MM
@date       12.01.2023
@param[in]  df - dataframe
@param[in]  _path - output path
@param[in]  _title - optional title
@note  
===========================================================================================
"""      
def stacked_barh_class_robustness_plots_absolute(df_classes, _path, _title="Robustness of features classes on ..."):
    robustness = ['poor', 'moderate' , 'good', 'excellent']
    results = {
        '2D-Shape': [],
        'First_Order': [],
        'GLCM': [],
        'GLRLM': [],
        'GLSZM': [],
        'GLDM': [],
        'NGTDM': []
    }
    
    i = 0
    for _item in df_classes:
        _class = _item[0]
        _df = _item[1]
        _poor = _df[_df['robustness'] == 'poor']
        _moderate = _df[_df['robustness'] == 'moderate']
        _good = _df[_df['robustness'] == 'good']
        _excellent = _df[_df['robustness'] == 'excellent']

        ##_data = [len(_poor), len(_moderate), len(_good), len(_excellent)]
        _data = []
        if(len(_poor)>0):
             _data.append(len(_poor))
             results[_class].append(len(_poor))
        else:
            results[_class].append(0)

        if(len(_moderate)>0):
            _data.append(len(_moderate))
            results[_class].append(len(_moderate))
        else:
            results[_class].append(0)

        if(len(_good)>0):
            _data.append(len(_good))
            results[_class].append(len(_good))
        else:
            results[_class].append(0)

        if(len(_excellent)>0):
            _data.append(len(_excellent))
            results[_class].append(len(_excellent))
        else:
            results[_class].append(0)

        i = i + 1
    
    _pdres = pd.DataFrame(results)
    colors = ['#e74c3c', '#f1c40f',  "#B3DE69", 'green']  
    
    # Erzeuge das Plot
    fig, ax = plt.subplots(figsize=(10, 6), gridspec_kw={'wspace': 0, 'hspace': 0})  
  
    for i, column in enumerate(_pdres.columns[::-1]):
        for j in range(len(_pdres[column])):
            value = _pdres[column][j]
            if value > 0:
                ax.barh(column, value, color=colors[j], left=np.sum(_pdres[column][:j]), edgecolor='none', height=0.9)  # Kein Rand (edgecolor='none')
                
                if (colors[j]=='green'):
                    textcolor = 'white'
                else:
                    textcolor = 'black'
                            
                ax.text(np.sum(_pdres[column][:j]) + value / 2, i, f'{value}', ha='center', va='center', color=textcolor, fontsize = 16, fontweight = 'bold')

    totals = _pdres.sum(axis=0)
    x_offset = 0.5
    y_offset = -10
    for i, total in enumerate(totals):
        ax.text(total + x_offset,  totals.index[i], round(total), ha='center', va='center', weight='bold', fontsize=18)

    ax.set_xlabel('')
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_yticks(np.arange(len(_pdres.columns[::-1])))
    ax.set_yticklabels(_pdres.columns[::-1], fontsize = 16)
    ax.set_title(_title, fontsize = 18, fontweight="bold")
    name_to_color = {name: color for name, color in zip(robustness, colors)}
    handles = [plt.Rectangle((0, 0), 0, 0, color=name_to_color[name], label=name) for name in name_to_color]
    plt.tight_layout()
    plt.savefig(_path, pad_inches = 0.1, bbox_inches='tight', transparent=True, dpi=600)
    print(_path + " saved")
    plt.close("all")
    plt.cla()
    plt.clf()
    


"""
===========================================================================================
@fn         plot_boxblots()
@details
@author     MM
@date       12.01.2023
@param[in]  vector1, 
@param[in]  vector2, 
@param[in]  minAllowed, 
@param[in]  maxAllowed, 
@param[in]  _path, 
@param[in]  _feature,
@param[in]  _plot_flag=False,
@param[in]  title="Feature difference", 
@param[in]  axLabels=["feature", "|R1-R2|"],  
@param[in]  _labelfontsize=22
@note  
===========================================================================================
"""  
def bland_altman_plot_norm(vector1, 
                           vector2, 
                           minAllowed, 
                           maxAllowed, 
                           _path, 
                           _feature,
                           _plot_flag=False,
                           title="Feature difference", 
                           axLabels=["feature", "|R1-R2|"],  
                           _labelfontsize=22):
   
        data1 = np.array(vector1).astype(np.float32)
        data2 = np.array(vector2).astype(np.float32)
         ### NORMALIZATION
        glob_min = min([min(data1), min(data2)])
        glob_max = max([max(data1), max(data2)])
        assert len(data1) == len(data2)
        data1_norm = normalize_relative_percent(data1, glob_min, glob_max)
        data2_norm = normalize_relative_percent(data2, glob_min, glob_max)
        mean      = np.mean([data1_norm, data2_norm], axis=0)
        diff      = data1_norm - data2_norm                   # Difference between data1 and data2
        md        = np.mean(diff)                   # Mean of the difference

        # Average difference (aka the bias)
        bias = md
        # Sample standard deviation
        sd        = np.std(diff, ddof=1)            # Standard deviation of the difference
        upper_loa = bias + 2 * sd
        lower_loa = bias - 2 * sd

        n = len(data1)
        # Variance
        var = sd**2
        # Standard error of the bias
        se_bias = np.sqrt(var / n)
        # Standard error of the limits of agreement
        se_loas = np.sqrt(3 * var / n)
        # Endpoints of the range that contains 95% of the Student’s t distribution
        t_interval = stats.t.interval(confidence=0.95, df=n - 1)
        # Confidence intervals
        ci_bias = bias + np.array(t_interval) * se_bias
        ci_upperloa = upper_loa + np.array(t_interval) * se_loas
        ci_lowerloa = lower_loa + np.array(t_interval) * se_loas

        if (_plot_flag==True):
            fig, ax = plt.subplots(1, 4, figsize=(16,5))
            ax[0].plot(data1, '--o', color='red', label="R1")
            ax[0].plot(data2, '--o', color='blue', label="R2")
            ax[0].grid()
            ax[0].set_title(_feature, fontsize = 18)
            ax[0].legend()

            ax[1].plot(data1_norm, '--o', color='red', label="R1")
            ax[1].plot(data2_norm, '--o', color='blue', label="R2")
            ax[1].grid()
            ax[1].set_title("normalized", fontsize = 18)
            ax[1].legend()

            ax[2].scatter(data1, data2, c='k', s=20, alpha=0.6, marker='o')
            ax[2].grid()
            ax[2].set_title(_feature, fontsize = 18)
            ax[2].set_xlabel('R1')
            ax[2].set_ylabel('R2')
            ax[2].set_aspect('equal')

            ax[3].set_title(title, fontsize = 18)
            ax[3].scatter(mean, diff, color="royalblue", alpha=0.6, s=80) # cornflowerblue
            ax[3].axhline(md,           color='blue', linestyle='--')
            ax[3].axhline(md + 1.96*sd, color='red', linestyle='--')
            ax[3].axhline(md - 1.96*sd, color='red', linestyle='--')
            ax[3].grid()
            
            # Get axis limits
            left, right = ax[3].get_xlim()
            bottom, top = ax[3].get_ylim()
            # Set y-axis limits
            max_y = max(abs(bottom), abs(top))
            ax[3].set_ylim(-max_y * 1.1, max_y * 1.1)
            ax[3].yaxis.get_label().set(fontsize=_labelfontsize)  
        
            # Set x-axis limits
            domain = right - left
            ax[3].set_xlim(left, left + domain * 1.1)
            ax[3].xaxis.get_label().set(fontsize=_labelfontsize)
            # Add the annotations
            ax[3].annotate('+1.96×SD', (right, upper_loa), (0, 35), textcoords='offset pixels', fontsize=15)
            ax[3].annotate(f'{upper_loa:+4.2f}', (right, upper_loa), (0, -115), textcoords='offset pixels', fontsize=14)
            ax[3].annotate('Bias', (right, bias), (0, 35), textcoords='offset pixels', fontsize=15)
            ax[3].annotate(f'{bias:+4.2f}', (right, bias), (0, -115), textcoords='offset pixels', fontsize=15)
            ax[3].annotate('-1.96×SD', (right, lower_loa), (0, 35), textcoords='offset pixels', fontsize=15)
            ax[3].annotate(f'{lower_loa:+4.2f}', (right, lower_loa), (0, -115), textcoords='offset pixels', fontsize=14)
            
            plt.savefig(_path, pad_inches = 0.1, bbox_inches='tight', transparent=True, dpi=300)
            print(_path + " saved")
            print(_path)
            #plt.show()
            plt.close("all")
            plt.cla()
            plt.clf()

        ret = [bias, lower_loa, upper_loa]
        print(ret)
        return ret
