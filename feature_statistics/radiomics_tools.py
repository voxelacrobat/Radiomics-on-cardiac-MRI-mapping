import os
import pandas as pd
import numpy as np
import pingouin as pg
from scipy.stats import pearsonr

"""
===========================================================================================
@brief      direct_icc_func()
@details    Direct calculation of intra class correllation
@author     https://github.com/cosanlab/nltools/blob/master/nltools/data/brain_data.py
@date       
@param[in]  v1
@param[in]  v2
@param[in]  icc_type
@return     ICC
@note  
===========================================================================================
"""     
def direct_icc_func(v1, v2, icc_type='ICC(2,1)'):
    ''' Calculate intraclass correlation coefficient

    ICC Formulas are based on:
    Shrout, P. E., & Fleiss, J. L. (1979). Intraclass correlations: uses in
    assessing rater reliability. Psychological bulletin, 86(2), 420.
    icc1:  x_ij = mu + beta_j + w_ij
    icc2/3:  x_ij = mu + alpha_i + beta_j + (ab)_ij + epsilon_ij
    Code modifed from nipype algorithms.icc
    https://github.com/nipy/nipype/blob/master/nipype/algorithms/icc.py

    Args:
        Y: The data Y are entered as a 'table' ie. subjects are in rows and repeated
            measures in columns
        icc_type: type of ICC to calculate. (ICC(2,1), ICC(2,k), ICC(3,1), ICC(3,k)) 
    Returns:
        ICC: (np.array) intraclass correlation coefficient
    '''

    '''
        msw = (aov.at[1, "SS"] + aov.at[2, "SS"]) / (aov.at[1, "DF"] + aov.at[2, "DF"])
        icc1 = (msb - msw) / (msb + (k - 1) * msw)
        icc2 = (msb - mse) / (msb + (k - 1) * mse + k * (msj - mse) / n)
        -> correct ICC = (MSR - MSE) / (MSR + (k-1) * MSE + k * (MSC - MSE) / n)
        icc3 = (msb - mse) / (msb + (k - 1) * mse)
        icc1k = (msb - msw) / msb
        icc2k = (msb - mse) / (msb + (msj - mse) / n)
        icc3k = (msb - mse) / msb

        # Calculate F, df, and p-values
        f1k = msb / msw
        df1 = n - 1
        df1kd = n * (k - 1)
        p1k = f.sf(f1k, df1, df1kd)

        f2k = f3k = msb / mse
        df2kd = (n - 1) * (k - 1)
        p2k = f.sf(f2k, df1, df2kd)
    '''
    Y = np.c_[v1,v2] #np.concatenate((y1,y2), axis=0)
    [n, k] = Y.shape

    # Degrees of Freedom
    dfc = k - 1
    dfe = (n - 1) * (k-1)
    dfr = n - 1
    dfinn = k*(n-1) ## TODO

    # Sum Square Total
    mean_Y = np.mean(Y)
    SST = ((Y - mean_Y) ** 2).sum()

    m1=np.mean(Y[:,0])
    m2=np.mean(Y[:,1])
    m=(m1+m2)/2
    err_a=list(Y[:,0]-m1)
    err_b=list(Y[:,1]-m2)
    err=err_a+err_b
    ssw=[]
    for i in err:
        ssw.append(i**2)
    
    SSW=np.sum(ssw)

    # create the design matrix for the different levels
    x = np.kron(np.eye(k), np.ones((n, 1)))  # sessions
    x0 = np.tile(np.eye(n), (k, 1))  # subjects
    X = np.hstack([x, x0])

    # Sum Square Error
    predicted_Y = np.dot(np.dot(np.dot(X, np.linalg.pinv(np.dot(X.T, X))), X.T), Y.flatten('F'))
    residuals = Y.flatten('F') - predicted_Y
    SSE = (residuals ** 2).sum()

    MSE = SSE / dfe
    # Sum square column effect - between colums
    SSC = ((np.mean(Y, 0) - mean_Y) ** 2).sum() * n
    MSC = SSC / dfc  # / n (without n in SPSS results)

    # Sum Square subject effect - between rows/subjects
    SSR = SST - SSC - SSE
    MSR = SSR / dfr

    MSW = SSW / dfinn

    ## https://datasciencechalktalk.wordpress.com/2019/09/04/one-way-analysis-of-variance-anova-with-python/
    ## TODO MSW

    if icc_type == 'ICC(1,1)' or icc_type == 'ICC(1,k)':
        # ICC(2,1) = (mean square subject - mean square error) /
        # (mean square subject + (k-1)*mean square error +
        # k*(mean square columns - mean square error)/n)
        # ICC = (MSR - MSRW) / (MSR + (k-1) * MSRW)

        if icc_type == 'ICC(1,k)':
            k = 1
        ICC = (MSR - MSW) / (MSR + (k-1) * MSW) 
        ##NotImplementedError("This method isn't implemented yet.")

    elif icc_type == 'ICC(2,1)' or icc_type == 'ICC(2,k)':
        # ICC(2,1) = (mean square subject - mean square error) /
        # (mean square subject + (k-1)*mean square error +
        # k*(mean square columns - mean square error)/n)
        if icc_type == 'ICC(2,k)':
            k = 1
        ICC = (MSR - MSE) / (MSR + (k-1) * MSE + k * (MSC - MSE) / n)

    elif icc_type == 'ICC(3,1)' or icc_type == 'ICC(3,k)':
        # ICC(3,1) = (mean square subject - mean square error) /
        # (mean square subject + (k-1)*mean square error)
        if icc_type == 'ICC(3,k)':
            k = 1
        ICC = (MSR - MSE) / (MSR + (k-1) * MSE)

    return ICC


"""
===========================================================================================
@brief      calc_feature_icc()
@details
@author     MM
@date       24.04.2024
@param[in]  df
@param[in]  _feature
@return     icc_results
@note  
===========================================================================================
"""       
def calc_feature_icc(df, _feature):
    data = df[df['repeat'].isin([1, 2])]
    results = pg.intraclass_corr(data, 'patient', 'repeat', _feature)
    results = results.set_index('Type')
    icc_results = results.set_index('Description')
    return icc_results


"""
===========================================================================================
@brief      interpret_koo_and_li(df)
@details
@author     MM
@date       24.04.2024
@param[in]  icc
@return     roobustness
@note  
===========================================================================================
"""   
def interpret_koo_and_li(icc):
    """Interpret the inter-rater agreement."""
    if icc < 0.50:
        return 'poor'
    elif (icc >= 0.5 and icc <= 0.75):
        return 'moderate'
    elif (icc > 0.75 and icc <= 0.9):
        return 'good'
    elif (icc > 0.9 and icc <= 1.0):
        return 'excellent'
    else:
        raise ValueError(f'Invalid value for the ICC: {icc}')
    

"""
===========================================================================================
@brief      create_folder()
@details
@author     MM
@date       24.04.2024
@param[in]  _folderpath
@return     True: folder created / False: creation failed
@note  
===========================================================================================
"""   
def create_folder(_folderpath):
    try: 
        os.mkdir(_folderpath) 
        return True
    except OSError as error: 
        print(error)
        return False

"""
===========================================================================================
@brief      normalize_between()
@details
@author     MM
@date       24.04.2024
@param[in]  array
@param[in]  ref_min
@param[in]  ref_max
@param[in]  minAllowed
@param[in]  maxAllowed
@return     normalized array
@note  
===========================================================================================
"""   
def normalize_between(array, ref_min, ref_max, minAllowed, maxAllowed):
     normalized = []
     
     for val in array:
          norm_val = (maxAllowed - minAllowed) * (val - ref_min) / (ref_max - ref_min) + minAllowed
          norm_val = norm_val.astype(np.float32)
          normalized.append(norm_val) 

     return np.array(normalized).astype(np.float32)

"""
===========================================================================================
@brief      normalize_relative_percent()
@details
@author     MM
@date       24.04.2024
@param[in]  X
@param[in]  x_min
@param[in]  x_max
@return     normalized array
@note  
===========================================================================================
""" 
def normalize_relative_percent(X, x_min, x_max):
    maxAllowed = 100.0
    minAllowed = 0.0

    if(0.0<=x_min) and (0.0<=x_max) and (x_min<x_max):
        x_min = 0.0
    elif(x_min<=0.0) and (x_max<=0.0) and (x_min<x_max):
        x_max = 0.0

    normalized = []   
    for x in X:
        norm_val = (maxAllowed - minAllowed) * (x - x_min) / (x_max - x_min) + minAllowed
        norm_val = norm_val.astype(np.float32)
        normalized.append(norm_val) 

    return np.array(normalized).astype(np.float32)

"""
===========================================================================================
@brief      remove_substrings_in_list()
@details
@author     MM
@date       24.04.2024
@param[in]  _string_list
@param[in]  _to_remove_strings
@return     _new_string_list
@note  
===========================================================================================
""" 
def remove_substrings_in_list(_string_list, _to_remove_strings):
    i = 0
    #labels_outer = df_feature.index.get_level_values(1).to_list()
    _new_string_list = _string_list.copy()
    for _text in _new_string_list:
        print(_text)
        for sub in _to_remove_strings:
            _text = _text.replace(sub, '')
            _new_string_list[i] = _text

        i = i + 1
    return _new_string_list

"""
===========================================================================================
@brief      concordance_correlation_coefficient()
@details    Concordance correlation coefficient
@author     MM
@date       24.04.2024
@param[in]  y1
@param[in]  y2
@return     ccc
@note  
===========================================================================================
"""   
def concordance_correlation_coefficient(y1, y2):
    mean1 = np.mean(y1)
    mean2 = np.mean(y2)
    var1 = np.var(y1)
    var2 = np.var(y2)
    covariance = np.mean((y1 - mean1) * (y2 - mean2))
    ccc = (2 * covariance) / (var1 + var2 + (mean1 - mean2)**2)
    return ccc

"""
===========================================================================================
@brief      mean_squared_error()
@details    Mean square error
@author     MM
@date       24.04.2024
@param[in]  y1
@param[in]  y2
@return     mse
@note  
===========================================================================================
"""  
def mean_squared_error(y1, y2):
    differences = y1 - y2
    squared_differences = differences ** 2
    mse = np.mean(squared_differences)
    return mse

"""
===========================================================================================
@brief      pearson_corr()
@details    Pearson correlation
@author     MM
@date       24.04.2024
@param[in]  y1
@param[in]  y2
@return     pearson_correlation, _p_val
@note  
===========================================================================================
"""  
def pearson_corr(y1, y2):
    pearson_correlation, _p_val = pearsonr(y1, y2)
    return pearson_correlation, _p_val

"""
===========================================================================================
@brief      calc_mrd()
@details    Mean relative difference
@author     MM
@date       24.04.2024
@param[in]  y1
@param[in]  y2
@return     [MRDleft, MRDright]
@note  
===========================================================================================
"""     
def calc_mrd(v1, v2):
    ## https://stats.stackexchange.com/questions/21587/how-to-find-mean-relative-differences
    assert len(v1) == len(v2)
    N = len(v1)

    sum_v1 = 0
    sum_v2 = 0
    sum_diff_left = 0
    sum_diff_right = 0
    for i in range(0,N):
        sum_v1 = sum_v1 + np.abs(v1[i])
        sum_diff_left = sum_diff_left + np.abs(v1[i] - v2[i])
        sum_v2 = sum_v2 + np.abs(v2[i])
        sum_diff_right = sum_diff_right + np.abs(v2[i] - v1[i])

    scale_left = sum_v1/N
    scale_right = sum_v2/N
    MRDleft = sum_diff_left/(N*scale_left)*100 
    MRDright = sum_diff_right/(N*scale_right)*100 

    return [MRDleft, MRDright]

"""
===========================================================================================
@brief      calc_CVs(df)
@details    Concordance correlation coefficient
@author     MM
@date       24.04.2024
@param[in]  v1
@param[in]  v2
@return     [_CV, _CVerr, _CVerr_abs]
@note  
===========================================================================================
"""   
def calc_CVs(v1,v2):
    Y = np.c_[v1,v2] 
    _CV = np.std(Y) / np.mean(Y) * 100   
    _err      = v1 - v2                   
    _err_abs  = np.abs(v1 - v2)           
    _CVerr = np.std(_err, ddof=1) / np.mean(_err) * 100   
    _CVerr_abs = np.std(_err_abs, ddof=1) / np.mean(_err_abs) * 100
    return [_CV, _CVerr, _CVerr_abs]