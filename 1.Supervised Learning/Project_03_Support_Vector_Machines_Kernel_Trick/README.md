# Support Vector Machines: Kernel Comparison and Optimization

## Project Overview
This project focuses on exploring and comparing **Support Vector Machines (SVM)** with different kernel functions for binary classification tasks.  
The main objectives are:  

- Understand how SVM works and the role of kernel functions.  
- Implement and compare different kernels: Linear, Polynomial, RBF, and Sigmoid.  
- Optimize hyperparameters using GridSearchCV.  
- Experiment with optimization techniques (SGD) for faster training.  
- Analyze model performance using accuracy, precision, recall, and F1-score.  

---

## Dataset Used

**Pima Indians Diabetes Database** (Kaggle)  
- Features: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age (8 numerical features).  
- Target: Binary classification (`Outcome`: 0 = No Diabetes, 1 = Diabetes)  
- Size: 768 samples with medical measurements from female patients.  
- Challenge: Class imbalance and missing values encoded as zeros.  
- Dataset link: [Kaggle - Pima Indians Diabetes](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)  


---

## Data Preprocessing
- **Handling invalid zeros**: Medical impossibilities (Glucose=0, BMI=0) replaced with NaN.  
- **Imputation**: Missing values filled with median of each column to preserve distribution.  
- **Duplicate removal**: Verification and removal of duplicate entries.  
- **Standardization**: Features scaled using StandardScaler for optimal SVM performance.  

---

## SVM Mathematical Foundation

Support Vector Machines find an optimal hyperplane that separates classes while maximizing the margin:

**Optimization Problem:**  
```
min(w,b) 1/2 ||w||²
Subject to: y⁽ⁱ⁾(w^T x⁽ⁱ⁾ + b) ≥ 1 - ξᵢ
```

**Kernel Functions** allow implicit transformation to higher dimensions:

1. **Linear Kernel:**  
   ```
   k(xⁱ, xʲ) = xⁱ^T xʲ
   ```
   Best for linearly separable data.

2. **Polynomial Kernel:**  
   ```
   k(xⁱ, xʲ) = (1 + xⁱ^T xʲ)^m
   ```
   Captures polynomial relationships between features.

3. **RBF Kernel (Radial Basis Function):**  
   ```
   k(xⁱ, xʲ) = exp(-||xⁱ - xʲ||² / 2σ²)
   ```
   Ideal for complex, non-linear separations.

4. **Sigmoid Kernel:**  
   ```
   k(xⁱ, xʲ) = tanh(β xⁱ^T xʲ + θ)
   ```
   Similar to neural network activation.

---

## SVM Implementations

### 1. Linear Kernel SVM

**Hyperparameters optimized:**  
- C: [0.1, 1, 10] - Regularization parameter  
- gamma: ['scale', 0.1, 0.01] - Kernel coefficient  

![Linear Confusion Matrix](images/Confusion%20Matrix%20-%20Linear%20Kernel.png)

**Results:**  
- Best parameters: C=1, gamma='scale'  
- Accuracy: 75.3%  
- Good baseline for linearly separable regions  

---

### 2. Polynomial Kernel SVM

**Hyperparameters optimized:**  
- C: [1, 10]  
- degree: [2, 3] - Polynomial degree  
- gamma: ['scale']  

 
![Polynomial Confusion Matrix](images/Confusion%20Matrix%20-%20Polynomial%20Kernel.png)

**Results:**  
- Best parameters: C=1, degree=3, gamma='scale'  
- Accuracy: 74.0%  
- Captures polynomial relationships but slightly lower performance  

---

### 3. RBF Kernel SVM

**Hyperparameters optimized:**  
- C: [0.1, 1, 10]  
- gamma: ['scale', 0.01, 0.1]  


![RBF Confusion Matrix](images/Confusion%20Matrix%20-%20RBF%20Kernel.png

**Results:**  
- Best parameters: C=1, gamma=0.01  
- Accuracy: **77.9%** ⭐ (Best overall)  
- Superior for complex non-linear boundaries  
- Improved recall on minority class  

---

### 4. Sigmoid Kernel SVM

**Hyperparameters optimized:**  
- C: [0.1, 1, 10]  
- gamma: ['scale', 0.01, 0.1]  


![Sigmoid Confusion Matrix](images/Confusion%20Matrix%20-%20Sigmoid%20Kernel.png)

**Results:**  
- Best parameters: C=10, gamma=0.01  
- Accuracy: 75.3%  
- Less stable, sensitive to extreme values  
- Performance similar to linear kernel  

---

## Correlation Analysis and PCA Visualization

### Correlation Matrix

Analysis of feature correlations with the target variable to identify the most predictive features.


![Correlation Matrix](images/Correlation%20Matrix%20of%20Diabetes%20Dataset.png)

**Key findings:**  
- Glucose shows highest correlation with Diabetes outcome  
- BMI and Age are also strong predictors  
- Used to select top 2 features for 2D visualization  

### PCA (Principal Component Analysis)

Dimensionality reduction from 8 features to 2 principal components for visualization:


![PCA Results](images/SVM%20Decision%20Boundaries%20in%20PCA%20Space.png)

**PCA Results:**  
- Variance explained: ~48% by first 2 components  
- Main contributors: Glucose (0.45), BMI (0.42), Age (0.38), Insulin (0.35)  
- Enables visualization of decision boundaries in 2D space  

**Decision Boundary Analysis:**  
- **Linear SVM**: Straight line boundary, limited separation due to class overlap  
- **Polynomial SVM**: Curved boundary attempting to follow data contours  
- **RBF SVM**: Best adaptation to data structure, captures complex patterns  
- **Sigmoid SVM**: Smooth transitions, similar to linear but more flexible  

**Key Observation**: Significant class overlap in 2D projection indicates the complexity of the classification problem. RBF kernel best handles this non-linear separation.

---

## Optimization Techniques

### SGDClassifier (Stochastic Gradient Descent)

Alternative linear approximation to SVM using gradient descent optimization.

**Why SGD?**  
- Faster training on large datasets  
- Approximate linear SVM behavior  
- Supports various loss functions and regularization  

**Hyperparameters optimized:**  
- loss: ['hinge', 'log'] - SVM or logistic loss  
- alpha: [0.0001, 0.001, 0.01] - Regularization strength  
- max_iter: [1000, 2000] - Maximum iterations  
- penalty: ['l2', 'l1', 'elasticnet'] - Regularization type  
- learning_rate: ['optimal', 'invscaling'] - Learning rate schedule  

![SGD Confusion Matrix](images/Confusion%20Matrix%20-%20SGDClassifier.png)

**Results:**  
- Best parameters: loss='hinge', penalty='elasticnet', learning_rate='optimal'  
- Accuracy: 77.3%  
- Fast alternative to kernel SVM  
- Good balance between speed and performance  

**Note on Adam and RMSprop:**  
These optimizers are designed for neural networks with backpropagation and adaptive learning rates. Scikit-learn's SVM implementations use analytical formulas, making Adam/RMSprop unavailable. For these optimizers, frameworks like TensorFlow or PyTorch are required.

---

## Performance Comparison

| Model | Accuracy | Precision (0/1) | Recall (0/1) | F1-Score (0/1) |
|-------|----------|-----------------|--------------|----------------|
| **SVM Linear** | 75.3% | 0.80 / 0.70 | 0.83 / 0.64 | 0.82 / 0.67 |
| **SVM Polynomial** | 74.0% | 0.79 / 0.68 | 0.81 / 0.63 | 0.80 / 0.65 |
| **SVM RBF** | **77.9%** ⭐ | 0.81 / 0.69 | 0.84 / 0.65 | 0.82 / 0.67 |
| **SVM Sigmoid** | 75.3% | 0.78 / 0.67 | 0.80 / 0.62 | 0.79 / 0.64 |
| **SGDClassifier** | 77.3% | 0.81 / 0.70 | 0.85 / 0.64 | 0.83 / 0.67 |

---

## Key Findings

1. **RBF Kernel achieves best performance** (77.9% accuracy) by effectively capturing non-linear patterns in the data.

2. **SGDClassifier offers competitive results** (77.3%) with significantly faster training time, making it suitable for large datasets.

3. **Class imbalance challenge**: All models predict the majority class (No Diabetes) better than the minority class (Diabetes), as shown by higher precision and recall for class 0.

4. **Linear vs Non-linear**:  
   - Linear and Sigmoid kernels: ~75% accuracy  
   - Polynomial kernel: 74% (may overfit with higher degrees)  
   - RBF kernel: 78% (best for this non-linear problem)  

5. **Hyperparameter tuning impact**: GridSearchCV successfully identifies optimal parameters, though improvements are modest due to inherent data complexity.

6. **PCA visualization**: Reveals significant class overlap in 2D, explaining why perfect separation is impossible. RBF's flexible boundary adapts best to this challenge.

7. **Practical considerations**:  
   - Use RBF for maximum accuracy on moderate datasets  
   - Use SGDClassifier for faster training and scalability  
   - Consider class weighting or SMOTE for imbalanced data  

---

## Technologies Used
- **Python 3.8+**  
- **scikit-learn**: SVM implementations, GridSearchCV, metrics  
- **pandas**: Data manipulation  
- **numpy**: Numerical operations  
- **matplotlib & seaborn**: Visualizations  

---

## Project Conclusion

This project demonstrates the power and versatility of Support Vector Machines for binary classification. Through systematic comparison of four kernel functions and optimization techniques, we showed that:

- **Kernel selection matters**: RBF kernel's flexibility makes it superior for non-linearly separable data.  
- **Data preprocessing is crucial**: Proper handling of missing values and feature scaling significantly impacts performance.  
- **Hyperparameter tuning pays off**: GridSearchCV with cross-validation ensures robust model selection.  
- **Trade-offs exist**: RBF offers best accuracy but slower training; SGD provides speed with competitive results.  

The analysis of correlation matrices, PCA projections, and decision boundaries provides deeper insights into the classification problem. While no model achieves perfect separation due to inherent class overlap, RBF SVM and SGDClassifier emerge as the most effective approaches for this diabetes prediction task.

Future improvements could include:
- Addressing class imbalance with SMOTE or class weighting  
- Feature engineering to create more discriminative variables  
- Ensemble methods combining multiple kernels  
- Deep learning approaches with Adam/RMSprop optimizers  

---

## References
- [Pima Indians Diabetes Database - Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)  
- [Scikit-learn: Support Vector Machines](https://scikit-learn.org/stable/modules/svm.html)  
- [Scikit-learn: SGDClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html)  
