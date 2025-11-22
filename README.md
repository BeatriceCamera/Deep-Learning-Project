# Deep-Learning-Project
Deep Learning project integrating **multimodal learning** with **textual and numerical data**, using **LSTM** and **MLP** branches to perform **joint classification and regression** on hotel reviews.


## Multimodal Deep Learning for Sentiment and Rating Prediction
- **Language:** Python  
- **Libraries:**  
  - **Deep Learning:** TensorFlow, Keras  
  - **Preprocessing & Analysis:** pandas, numpy, scikit-learn  
  - **Visualization:** matplotlib, seaborn  
  - **Utilities:** itertools, random  


## Overview  
This project implements a **multimodal neural network** designed to process both **numerical** and **textual** features extracted from hotel reviews.  
The goal is twofold:
1. **Classify** whether a review is positive or negative.  
2. **Predict** the associated **numerical score (regression)**.  

The dataset includes customer feedback, ratings, and textual descriptions, enabling the model to learn how linguistic sentiment correlates with quantitative evaluations.


## Architecture  
The model follows a **dual-branch architecture**:  
- **MLP branch:** processes tabular/numerical features (e.g., review length, sentiment metrics).  
- **LSTM branch:** processes tokenized and embedded textual reviews.  

After independent processing, the two representations are **concatenated** and passed to shared dense layers that produce:  
- an **output** for binary classification,  
- an **output** for score regression.  

Training uses a **custom multi-output loss function** combining *binary cross-entropy* and *mean squared error*, balanced with weighting factors *(λ₁, λ₂)*.


## Methodology  
- **Text Preprocessing:** tokenization, embedding, and sequence padding.  
- **Regularization:** dropout layers and L2 penalties to reduce overfitting.  
- **Optimization:** Adam optimizer with early stopping on validation loss.  
- **Evaluation:** Accuracy, F1-score, and Mean Absolute Error (MAE) used to assess both tasks.  


## Results and Insights  
- The **LSTM–MLP hybrid** model achieved stable convergence and balanced performance across both outputs.  
- The **text branch** captured sentiment nuances, while the **numeric branch** improved consistency of score prediction.  
- The joint architecture demonstrated **good generalization** and highlights the value of shared feature learning across tasks.  


### View the Full Notebook  
You can explore the complete notebook interactively on **nbviewer**:  
[Open in nbviewer](https://nbviewer.org/github/BeatriceCamera/Deep-Learning-Project/blob/main/DLProject.ipynb)


**Author:** Beatrice Camera  
B.Sc. Artificial Intelligence @ University of Pavia, University of Milan, University of Milano-Bicocca  
