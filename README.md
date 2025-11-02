# CT-Muscle-Segmentation-using-Bayesian-U-Net-and-nnU-Net
This repository documents the workflow and visualization process from my research on **automated musculoskeletal segmentation** in torso CT images at the **Nara Institute of Science and Technology (NAIST), Japan**.  > ⚠️ Due to institutional research policies and ongoing publication, model code and real experimental results are not included.  
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

---

## 📊 Visualization Example

Although real results cannot be shared, the following script demonstrates the **boxplot generation process** used to compare segmentation metrics (e.g., Dice coefficient, ASD, AVE) across models.

### Example Code (`boxplot_results.py`)
```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Example synthetic data
data = pd.DataFrame({
    "Model": ["Bayesian U-Net"] * 5 + ["nnU-Net"] * 5,
    "Dice": [0.87, 0.88, 0.86, 0.89, 0.88, 0.90, 0.91, 0.89, 0.90, 0.88]
})

sns.boxplot(x="Model", y="Dice", data=data, palette="Set2")
plt.title("Comparison of Segmentation Accuracy (Synthetic Data)")
plt.xlabel("Model")
plt.ylabel("Dice Coefficient")
plt.show()
