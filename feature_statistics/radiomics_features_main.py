import os
import pandas as pd
import numpy as np
#import seaborn as sns
#import matplotlib.pyplot as plt
from var_radiomics_selected_features import selecteted_features, class_selecteted_features
from radiomics_features_plots import bland_altman_plot, plot_pairplot,  \
stacked_barh_class_robustness_plots_percent, stacked_bar_class_robustness_absolute_plots, stacked_barh_class_robustness_plots_absolute, \
bland_altman_plot_norm
from radiomics_tools import concordance_correlation_coefficient, calc_mrd, calc_CVs, pearson_corr, mean_squared_error, interpret_koo_and_li, direct_icc_func, calc_feature_icc, create_folder 
from radiomics_robustness_to_heatmap import create_heatmaps_for_all_T1_robustnesses_multi_plot, create_heatmaps_for_all_T2_robustnesses_multi_plot
from radiomics_best_features_to_heatmap import create_T1_best_heatmap_from_df, create_T2_best_heatmap_from_df, create_ICC_CCC_Rcorr_T1andT2_from_df

## Coefficient of variation
cv = lambda x: np.std(x, ddof=1) / np.mean(x) * 100 

"""
===========================================================================================
@brief      result_dataframe_split_to_classes(df)
@details
@author     MM
@date       24.04.2024
@param[in]  df_results
@param[in]  _result_attribute
@param[in]  _moco
@param[in]  _res
@param[in]  _view
@param[in]  _result_root_path
@return     void
@note  
===========================================================================================
"""   
def result_dataframe_split_to_classes(df_results, _result_attribute, _moco, _res, _view, _result_root_path):
    _feature_class_dfs = []
    _feature_class_dfs.append(["2D-Shape",     df_results[(df_results['class'] == "2D-Shape")]])
    _feature_class_dfs.append(["First_Order", df_results[(df_results['class'] == "First_Order")]])
    _feature_class_dfs.append(["GLCM",      df_results[(df_results['class'] == "GLCM")]])
    _feature_class_dfs.append(["GLRLM",     df_results[(df_results['class'] == "GLRLM")]])
    _feature_class_dfs.append(["GLSZM",     df_results[(df_results['class'] == "GLSZM")]])
    _feature_class_dfs.append(["GLDM",      df_results[(df_results['class'] == "GLDM")], ])
    _feature_class_dfs.append(["NGTDM",     df_results[(df_results['class'] == "NGTDM")]])
    T_result_path = os.path.join(_result_root_path, _result_attribute)

    if (_res == "nd") or (_result_attribute == "T2_selected_features"):
        title = "T2/%s"%(_view)
        _mseq = "T2"
    else:
        if _res == "LR":
            title = "T1/SR/%s"%(_view)
        else:
            title = "T1/%s/%s"%(_res, _view)
        _mseq = "T1"

    if (_res=="HR") and (_view=="4Ch"):
        debug = 1

    _robust_pies_filepath = os.path.join(T_result_path, "pie_class_%s_%s_%s_Robustnesses.png"%(_moco, _res, _view ))
    #pie_class_robustness_plots(_feature_class_dfs, T_result_path, title)     
    _bar_stacked_absolute = os.path.join(T_result_path, "%s_bar_stacked_absolute_%s_%s_%s_Robustnesses.png"%(_mseq, _moco, _res, _view ))
    #stacked_bar_class_robustness_absolute_plots(_feature_class_dfs, _bar_stacked_absolute, title) 
    _barh_stacked_percent = os.path.join(T_result_path, "%s_barh_stacked_percent_%s_%s_%s_Robustnesses.png"%(_mseq, _moco, _res, _view ))
    stacked_barh_class_robustness_plots_percent(_feature_class_dfs, _barh_stacked_percent, title) 
    _barh_stacked_absolute = os.path.join(T_result_path, "%s_barh_stacked_absolute_%s_%s_%s_Robustnesses.png"%(_mseq, _moco, _res, _view ))
    stacked_barh_class_robustness_plots_absolute(_feature_class_dfs, _barh_stacked_absolute, title) 

"""
===========================================================================================
@brief      data_frame_pre_analysis_all_filters(df)
@details
@author     MM
@date       24.05.2024
@param[in]  _src_path
@param[in]  _result_root_path
@param[in]  _result_attribute
@param[in]  _enable_bland_altman
@return     df_all_results
@note  
===========================================================================================
"""   
def data_frame_pre_analysis_all_filters(_src_path, 
                                        _result_root_path, 
                                        _result_attribute):
    
    if (_result_attribute == "T1_selected_features"):
        _filt_resolutions = ["HR", "LR"]
        _sheet_name = "features_T1"
    elif (_result_attribute == "T2_selected_features"):
        _filt_resolutions = ["nd"]
        _sheet_name = "features_T2"

    _filt_views = ["4Ch", "SAX"]
    _filt_moco = ["ORG"] # ["MOCO", "ORG"]
    sheet_T1 = pd.read_excel(_src_path, sheet_name=_sheet_name)
    acquistion_info = ["patient", "repeat", "MOCO", "Resolution", "View"]
    mymatrix = np.array(class_selecteted_features)
    relevant_columns_T1 = acquistion_info + mymatrix[:,1].tolist() 
    df_T1 = pd.DataFrame(sheet_T1, columns=relevant_columns_T1)
    print("Num Columns: %d"%(len(relevant_columns_T1)))
    print("Num features: %d"%(len(class_selecteted_features)))
    print("Given Dataframe :\n", df_T1)

    _all_ICC_list = []
    idx = 0
    for res in _filt_resolutions:
        for view in _filt_views:
            for moco in _filt_moco:
                temp_path = os.path.join(_result_root_path, _result_attribute)
                _result_list = [] 
                new_idx = 0
                for _item in class_selecteted_features: 
                    _feature_class = _item[0]
                    _feature = _item[1]
                    ## Filter entire Datenframe for (Res, View, MOCO)
                    df_filtered = df_T1[(df_T1['Resolution'] == res) & (df_T1['View'] == view) & (df_T1['MOCO'] == moco)] 
                    df_pairwise_d1d2_feature =  df_filtered[['patient', 'repeat', _feature]]
                    pairwise_filepath = os.path.join(_result_root_path, "pairwise_%s_%s_%s_%s_%s.xlsx"%(_result_attribute, res, view, moco, _feature)) 
                    ### nur zum Test
                    df_d1_data = df_pairwise_d1d2_feature[(df_pairwise_d1d2_feature['repeat'] == 1)]
                    df_d2_data = df_pairwise_d1d2_feature[(df_pairwise_d1d2_feature['repeat'] == 2)]
                    y1 = np.array(df_d1_data[_feature]).astype(np.float32)
                    y2 = np.array(df_d2_data[_feature]).astype(np.float32)

                    if (y1.size > 0 and y2.size > 0):
                        if (max(y1) != min(y1)) and (max(y2) != min(y2)):
                            [CV, CVerr, CVerr_abs] = calc_CVs(y1,y2)
                            [MRDleft, MRDright] = calc_mrd(y1, y2)
                            T_result_path = os.path.join(_result_root_path, _result_attribute)
                            ccc_result = concordance_correlation_coefficient(y1, y2)
                            _MSE = mean_squared_error(y1, y2)
                            _PersonCoeff, _p_val = pearson_corr(y1, y2)
                             ## Calc coefficent of variation
                            _icc_type='ICC(2,1)'
                            direct_icc = direct_icc_func(y1, y2, icc_type=_icc_type)
                            icc_results = calc_feature_icc(df_pairwise_d1d2_feature, _feature)
                            # ICC = icc_results.loc['Single raters absolute', 'ICC']
                            # lower_ci = icc_results.loc['Single raters absolute', 'CI95%'][0]
                            # upper_ci = icc_results.loc['Single raters absolute', 'CI95%'][1]  
                            ## ICC(2,1)
                            ICC = icc_results.loc['Single random raters',"ICC"]
                            lower_ci = icc_results.loc['Single random raters', 'CI95%'][0]
                            upper_ci = icc_results.loc['Single random raters', 'CI95%'][1]
                            # ICC = icc_results.loc['Single fixed raters', 'ICC']
                            # lower_ci = icc_results.loc['Single fixed raters', 'CI95%'][0]
                            # upper_ci = icc_results.loc['Single fixed raters', 'CI95%'][1]
                            robustness = interpret_koo_and_li(ICC)
                            temp_path = os.path.join(_result_root_path, _result_attribute)
                            _folderpath = os.path.join(temp_path, "%s_%s_%s"%(res, view, moco))
                        else:
                            icc_results = 0
                            ICC = 0
                    else:
                        icc_results = 0
                        ICC = 0
                    
                    print("%3d | %4s | %4s | %10s | %10s | %40s  |  %10s | %.3f | %.3f  95%% CI [%.3f, %.3f] | %5.3f | %5.3f | %5.3f"%(idx, res, view, moco, _feature_class, _feature, robustness, direct_icc, ICC, ccc_result, lower_ci, upper_ci, CVerr_abs, MRDleft))
                    icc_str = "%.3f  (%.3f, %.3f)"%(ICC, lower_ci, upper_ci) 
                    my_results = [new_idx, res, view, moco, _feature_class, _feature, robustness, direct_icc, icc_str, ICC, ccc_result, _PersonCoeff, _MSE, CVerr_abs, MRDleft]
                    _result_list.append(my_results)
                    _all_ICC_list.append(my_results)
                    idx = idx + 1 
                    new_idx = new_idx + 1 
                
                df_results = pd.DataFrame(_result_list, columns =['Idx', 'Res', 'View', 'MOCO', 'class', 'feature', 'robustness', 'myICC(2,1)', 'ICC_confi(2,1)', 'ICC(2,1)', 'CCC', 'R_pearson', 'MSE', 'CV|err|', 'MRD'])
                T_result_path = os.path.join(_result_root_path, _result_attribute)
                pairwise_filepath = os.path.join(T_result_path, "all_%s_%s_%s_ICC.xlsx"%(moco, res, view ))
                result_dataframe_split_to_classes(df_results, _result_attribute, moco, res, view, _result_root_path )

                if (moco not in ["MOCO"]):
                    if (_result_attribute == "T2_selected_features"):
                        res = "nd"
           
                df_pairplot = df_results[['robustness', 'myICC(2,1)', 'CV|err|', 'MRD']] 
                _pairplot_filepath = os.path.join(T_result_path, "%s_%s_%s_Pairplot.png"%(moco, res, view))
                plot_pairplot(df_pairplot, _pairplot_filepath, _title="Pairplot (%s/%s/%s)"%(moco, res, view))

    df_all_results = pd.DataFrame(_all_ICC_list, columns =['Idx', 'Res', 'View', 'MOCO', 'class', 'feature', 'robustness', 'myICC(2,1)', 'ICC_confi(2,1)', 'ICC(2,1)', 'CCC', 'R_pearson', 'MSE', 'CV|err|', 'MRD'])
    T_result_path = os.path.join(_result_root_path, _result_attribute)
    pairwise_filepath = os.path.join(T_result_path, "all_ICC_%s.xlsx"%(_result_attribute))
    df_all_results.to_excel(pairwise_filepath)

    return df_all_results
                 
"""
===========================================================================================
@fn         main()    
@author     MM
@date       24.04.2024
@return     void
@note  
===========================================================================================
"""     
def main():
    result_root_path = "..\\results\\"
    t1_src_path = '..\\features\\features_T1.xlsx'
    t2_src_path = '..\\features\\features_T2.xlsx'

    _resdf_T1 = data_frame_pre_analysis_all_filters(t1_src_path, result_root_path, "T1_selected_features")
    create_heatmaps_for_all_T1_robustnesses_multi_plot(_resdf_T1, result_root_path, "T1_selected_features")
    _resdf_T2 = data_frame_pre_analysis_all_filters(t2_src_path, result_root_path, "T2_selected_features")   
    create_heatmaps_for_all_T2_robustnesses_multi_plot(_resdf_T2, result_root_path, "T2_selected_features")

    create_T1_best_heatmap_from_df(_resdf_T1, result_root_path, "T1_selected_features", True)
    create_T1_best_heatmap_from_df(_resdf_T1, result_root_path, "T1_selected_features", False)
    create_T2_best_heatmap_from_df(_resdf_T2, result_root_path, "T2_selected_features", True)
    create_T2_best_heatmap_from_df(_resdf_T2, result_root_path, "T2_selected_features", False)

    create_ICC_CCC_Rcorr_T1andT2_from_df(_resdf_T1, _resdf_T2, result_root_path)


if __name__ == "__main__":
    main()