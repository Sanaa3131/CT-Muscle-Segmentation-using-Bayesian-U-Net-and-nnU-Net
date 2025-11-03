# CT-Muscle-Segmentation-using-Bayesian-U-Net-and-nnU-Net
This repository documents the workflow and visualization process from my research on **automated musculoskeletal segmentation** in torso CT images at the **Nara Institute of Science and Technology (NAIST), Japan**.  
> ⚠️ Due to institutional research policies and ongoing publication, model code and real experimental results are not included.  
## 🎯 Project Overview

The research investigates automated segmentation of torso muscles using **deep learning models (Bayesian U-Net and nnU-Net)** and explores **uncertainty quantification** and investigates uncertainty use as a surrogate for segmentation accuracy.

Key contributions:
- Evaluated predictive uncertainty as a surrogate for segmentation accuracy.
- Conducted large-scale biomarker analysis of torso muscles (volume, density, fat ratio).

---

## ⚙️ Tools & Frameworks

| Category | Tools |
|-----------|-------|
| Programming | Python |
| Deep Learning | PyTorch |
| Image Processing | SimpleITK, scikit-image |
| Visualization | matplotlib, seaborn |
| Data Handling & Analysis | Numpy, Pandas |
| Statistical analysis | scipy.stats |
| Statistical visualization enhancement | statannotations |

---

## 📊 Visualization Example

Although real results cannot be shared, the following script demonstrates the **boxplot generation process** used to compare segmentation metrics (e.g., Dice coefficient, ASD, AVE) across models.

### Example Code (`Metric_Evaluation_Results.py`)
```python
import os
import pandas as pd 
import numpy as np 
import glob
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy.stats import wilcoxon,ttest_rel,shapiro,mannwhitneyu
from scipy import stats
from statannotations.Annotator import Annotator
import re
import matplotlib.gridspec as gridspec

rc('font', family='Times New Roman')
Structures = ['Rectus abdominis',
              'Psoas major', 
              'Intercostal muscles', 
              'Erector spinae', 
              'Quadratus lumborum', 
              'Internal oblique', 
              'Transversus abdominis', 
              'External oblique', 
              'Pectoralis major',
              'Pectoralis minor',
              'Trapezius',
              'Serratus Anterior',
              'Latissimus Dorsi',
              'Supraspinatus',
              'Infraspinatus',
              'Subscapularis',
              'Teres major',
              'Teres minor',
              'Other muscles',
              'Visceral fat',
              'Subcutaneous fat']

case_IDs = ['case1', 'case2', 'case3', 'case4', 'case5', 'case6', 'case7', 'case8', 'case9',
            'case10', 'case11', 'case12', 'case13', 'case14', 'case15', 'case16', 
            'case17', 'case18', 'case19', 'case20']

metric_list = ['DC','ASD','AbsHU','AbsVol']

for metric in metric_list:
    data1=pd.read_csv(f'path/Model_A/Evaluation/{metric}.csv',names=Structures)
    data1 = data1.drop(data1.columns[19], axis=1) #Drop other muscles 
    data2=pd.read_csv(f'path/Model_B/Evaluation/{metric}.csv',names=Structures)
    data2 = data2.drop(['Other muscles'], axis=1) #Drop 'Other muscles'
    
    # Compute the average DC value for each case
    data1[f'Average_{metric}'] = data1.mean(axis=1)
    data2[f'Average_{metric}'] = data2.mean(axis=1)
    
    # Compute the STD sample value for each case
    data1[f'STD_{metric}'] = data1.std(axis=1)
    data2[f'STD_{metric}'] = data2.std(axis=1)
    
    # Add a column to indicate the dataset
    data1['Model'] = 'Model A'
    data2['Model'] = 'Model B'
    
    # Combine the datasets
    combined_df = pd.concat([data1[[f'Average_{metric}', 'Model']],
                             data2[[f'Average_{metric}', 'Model']]])
    
    # Create boxplots
    fig, ax1 = plt.subplots(figsize=(2, 6))
    
    # Create boxplot
    sns.boxplot(data=combined_df, x='Model', y=f'Average_{metric}', palette='pastel', showmeans=True, meanline=False, width=0.5, ax=ax1)
    
    # Overlay data points
    sns.stripplot(data=combined_df, x='Model', y=f'Average_{metric}', color='red', alpha=0.6, jitter=True, ax=ax1)
    
    # Define pairwise comparisons for Models
    pairwise_comparisons = [('Model A', 'Model B')]
    
    data11 = combined_df[combined_df['Model'] == 'Model A'][f'Average_{metric}']
    data22 = combined_df[combined_df['Model'] == 'Model B'][f'Average_{metric}']
    
    custom_annotations = []
    
    # Annotator setup
    annotator = Annotator(ax1, pairwise_comparisons, data=combined_df, x='Model', y=f'Average_{metric}')
    
    if len(data1) == len(data2):
        # Test for normality using Shapiro-Wilk test
        stat1, p1 = shapiro(data11)
        stat2, p2 = shapiro(data22)
        
    # If all datasets are normally distributed, perform paired t-tests
        if p1 > 0.05 and p2 > 0.05:
            print('Performing paired t-tests')
            annotator.configure(test="t-test_paired")
            annotator.apply_and_annotate()
        else:
            # If not normally distributed, perform Wilcoxon signed-rank tests
            print('Performing Wilcoxon signed-rank tests')
            annotator.configure(test="Wilcoxon")
            annotator.apply_and_annotate()
    else:
        # If datasets have different lengths, perform Mann-Whitney U tests
        print('Performing Mann-Whitney U tests')
        annotator.configure(test="Mann-Whitney")
        annotator.apply_and_annotate()
        
    # Calculate overall average and standard deviation for each dataset
    dataset_stats = combined_df.groupby('Model')[f'Average_{metric}'].agg(['mean', 'std'])

    
    # Format the table data as average ± standard deviation
    table_data = [
        dataset_stats.index.tolist(),
        [f"{row['mean']:.2f} ± {row['std']:.2f}" for _, row in dataset_stats.iterrows()]
        ]
          
    # Add the table directly below the boxplot
    table = ax1.table(
        cellText=table_data,
        rowLabels=None,
        loc='bottom',
        cellLoc='center',
        bbox=[0, -0.15, 1, 0.15] # [0, -0.15, 1, 0.15]
        )
    
    # Customize the table font
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    
    # Apply bold styling manually 
    for (row_idx, col_idx), cell in table.get_celld().items():
        if row_idx > 0 and col_idx == 1:  
            text = cell.get_text().get_text()
            print(text)
            cell.set_text_props(weight="bold")

    # Adjust the position of the boxplot
    ax1.set_position([0.1, 0.15, 1.2, 0.75])
    
    # Add labels and grid
    ax1.set_ylabel(f'{metric}', fontsize=12)
    ax1.set_xlabel('')
    ax1.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    plt.tight_layout()
    plt.show()

