# Explainable AI (XAI) Iris Flower Classification Dashboard

An interactive, production-ready Explainable AI (XAI) application that classifies Iris flower species using a standardized Weighted $K$-Nearest Neighbors (KNN) pipeline, and provides local visual interpretations of predictions using LIME (Local Interpretable Model-agnostic Explanations).

---

## 1. Project Title

**Dynamic Iris Classification and Local Model Interpretation Dashboard**

---

## 2. Project Overview

This project delivers a web dashboard to predict Iris flower species and explain individual model predictions in real time. It features a scikit-learn standard evaluation pipeline (containing standard scaling and distance-weighted KNN), a dynamic rule-extraction engine, and an embedded LIME explanation panel. The frontend utilizes a premium two-column layouts separating prediction details from explainability outputs, offering a clear interpretation of classifications for both data scientists and end users.

---

## 3. Problem Statement

Many machine learning models operate as "black boxes," making decisions without explaining their reasoning. In critical sectors (such as biology, medicine, and automated security), relying on raw outputs without understanding feature contributions can lead to distrust or a failure to detect model bias. This project addresses this issue of explainability by creating a clear interpretation layer for a classification model.

---

## 4. Objectives

- **Build a Robust ML Pipeline:** Implement feature standardization and an optimized Weighted $K$-Nearest Neighbors classification model.
- **Establish a Web Interface:** Design a responsive Flask web application that accepts user parameter inputs, displays predictions, and serves live visual explanations.
- **Implement Local Explainability:** Integrate the LIME framework to explain predictions locally on a sample-by-sample basis.
- **Deliver Human-Readable Explanations:** Build a rule extraction compiler that translates statistical feature attributions into natural language narratives and color-coded status indices.
- **Ensure Production Quality:** Create modular, version-controlled scripts with clean structures and comprehensive documentation.

---

## 5. Why Explainable AI (XAI) is Important

Explainable AI (XAI) bridges the gap between predictive performance and human understanding. It is important for:

- **Building Trust:** Stakeholders must understand _why_ a model reached a specific conclusion before acting on its results.
- **Model Debugging:** Allows developers to verify that the model is learning meaningful biological features rather than noise or dataset artifacts (e.g. data leakage).
- **Regulatory Compliance:** Meets legal standards (such as the GDPR's "right to explanation") by explaining decisions that impact users.
- **Bias Detection:** Highlights features that are driving unfair or skewed classifications.

---

## 6. Dataset Description

The model is trained on the classic **Iris Flower Dataset** (originally introduced by Ronald Fisher in 1936), which consists of 150 samples from three distinct species:

- _Iris setosa_
- _Iris versicolor_
- _Iris virginica_

Each sample contains four measurements (features) in centimeters:

1.  **Sepal Length:** Length of the flower's sepal.
2.  **Sepal Width:** Width of the flower's sepal.
3.  **Petal Length:** Length of the flower's petal.
4.  **Petal Width:** Width of the flower's petal.

---

## 7. Machine Learning Algorithm

The core classification model uses **$K$-Nearest Neighbors (KNN)**. KNN classifies a test sample by identifying the $K$ points in the training dataset that are closest to it (using a distance metric) and assigning the class label that is most common among those neighbors.

---

## 8. Why Weighted KNN Was Selected

For this implementation, **Weighted KNN** is utilized with parameters `n_neighbors=5` and `weights='distance'`.

- **Distance-Weighted Voting:** In standard KNN, all $K$ neighbors have equal votes. In Weighted KNN, votes are weighted by the inverse of their distance to the query point:
  $$w_i = \frac{1}{d(x, x_i)}$$
  This ensures that neighbors closer to the query point exert a stronger influence on the classification output than those further away, resolving ties and improving accuracy in clustered overlap regions.
- **Non-linear decision space:** Dynamic classification boundaries are handled well by KNN without assuming linear data separation structures.

---

## 9. Why StandardScaler is Required

KNN classification uses distance metrics (such as Euclidean distance) to identify neighbors:
$$d(\mathbf{p}, \mathbf{q}) = \sqrt{\sum_{i=1}^{n} (p_i - q_i)^2}$$
If one feature has a much larger scale (e.g. millimeters vs centimeters), it will dominate the distance calculations. **`StandardScaler`** standardizes features by centering them to a mean of 0 and scaling them to a variance of 1:
$$z = \frac{x - \mu}{\sigma}$$
This ensures all dimensions contribute equally to the distance measurements, preventing scale bias.

---

## 10. Explainable AI using LIME

**LIME (Local Interpretable Model-agnostic Explanations)** explains machine learning predictions by treating the model as a black box. For any individual test instance, LIME generates local explanations through a four-step process:

1.  **Perturbing the Input:** Generates replicas of the test instance by adding random changes (perturbations) to its feature values.
2.  **Generating Predictions:** Runs the perturbed samples through the pipeline to obtain prediction confidence scores.
3.  **Distance Weighting:** Calculates the distance between each perturbed sample and the original test instance. Replicas that are closer to the input receive higher similarity weights.
4.  **Fitting a Local Surrogate:** Fits a simple, interpretable linear model (such as Ridge Regression) on the weighted perturbations. The surrogate model's coefficients are displayed as the feature contributions for that prediction.

---

## 11. Project Architecture Diagram

The diagram below shows the flow of data through the prediction and explanation pipeline:

```text
  [ User Web Interface ] --- (Sepal/Petal Inputs) ---> [ Flask Web Controller ]
                                                                |
  +-------------------------------------------------------------+
  |
  v
[ Loaded model.pkl Pipeline ]
  |
  +------> [ 1. StandardScaler ] ➔ Scales raw inputs using training statistics.
  |
  +------> [ 2. Weighted KNN ]   ➔ Generates prediction and class probabilities.
  |
  v
[ LIME Tabular Explainer ]
  |
  +------> Generates perturbed samples and fits a local surrogate model.
  +------> Generates interactive visualizations and saves them to 'static/lime/explanation.html'.
  |
  v
[ Narrative Rule Compiler ]
  |
  +------> Identifies strongest/weakest features and compiles the text description.
  +------> Computes the confidence-based trust score.
  |
  v
[ HTML UI Rendering ] <--- (Displays metrics, images, and explanations) --- [ index.html ]
```

---

## 12. Project Folder Structure

```text
iris_flask_app/
├── iris_flask_app/
│   ├── static/
│   │   └── lime/
│   │       └── explanation.html    # Generated dynamic interactive LIME report
│   ├── templates/
│   │   └── index.html              # Dynamic Jinja2 dashboard template
│   ├── app.py                      # Flask Application Controller & XAI Engine
│   ├── model.pkl                   # Pickled StandardScaler + Weighted KNN Pipeline
│   └── train_model.py              # Offline model training & validation pipeline script
├── .gitignore                      # Git exclusion rules
└── README.md                       # Comprehensive project documentation (this file)
```

---

## 13. Technologies Used

- **Python (v3.13.7 / v3.14.x):** Base runtime.
- **Flask (v3.x):** Web backend framework.
- **Scikit-Learn (v1.6+ / v1.17+):** ML preprocessing, models, and evaluation package.
- **NumPy:** Fast multi-dimensional array operations.
- **SciPy:** Scientific computing backend for scikit-learn.
- **LIME (v0.2.x):** Explainable AI local interpretations.
- **Jinja2 & CSS3:** Responsive UI rendering and dashboard layout.

---

## 14. Installation Guide

### Prerequisites

- Python 3.13+ installed on your system.

### Steps

1.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd iris_flask_app
    ```
2.  **Set Up a Virtual Environment:**
    ```bash
    python -m venv venv
    ```
3.  **Activate the Virtual Environment:**
    - **Windows (PowerShell):**
      ```powershell
      .\venv\Scripts\Activate.ps1
      ```
    - **Windows (CMD):**
      ```cmd
      .\venv\Scripts\activate.bat
      ```
    - **macOS / Linux:**
      ```bash
      source venv/bin/activate
      ```
4.  **Install Required Packages:**
    ```bash
    pip install flask numpy scikit-learn scipy lime
    ```

---

## 15. Running the Project

### Phase 1: Train the Machine Learning Pipeline

Train the model, evaluate its performance, and export the standardized pipeline by running:

```bash
python train_model.py
```

This script computes validation metrics and exports the StandardScaler + Weighted KNN model to `model.pkl`.

### Phase 2: Start the web application

Start the Flask web server by running:

```bash
python app.py
```

The server will start in debug mode. Open your browser and navigate to `http://127.0.0.1:5000/`.

---

## 16. Example Prediction Workflow

1.  **Dashboard Load:** Open browser to `http://127.0.0.1:5000/`.
2.  **Input Parameters:** Enter measurements (e.g. Sepal Length: `6.0`, Sepal Width: `3.0`, Petal Length: `4.8`, Petal Width: `1.8`).
3.  **Submit Request:** Click **Predict Flower**.
4.  **Web Processing:** The app parses values, scales features, predicts the class (and probability), generates a local explanation, and saves the LIME HTML report.
5.  **Dashboard Display:** The page updates to show the prediction results next to the explainability panels.

---

## 17. Evaluation Metrics

The model is evaluated on a stratified 20% test split. Running `train_model.py` outputs the following metrics:

- **Accuracy:** The percentage of correct classifications.
- **Precision:** The proportion of positive predictions that are correct.
- **Recall:** The proportion of actual positive samples that are identified.
- **F1-Score:** The harmonic mean of precision and recall.
- **Confusion Matrix:** A grid showing predicted vs. true labels.

### Example Console Report:

```text
================ MODEL EVALUATION METRICS ================
Accuracy:  0.9667
Precision: 0.9697 (weighted)
Recall:    0.9667 (weighted)
F1 Score:  0.9663 (weighted)

Confusion Matrix:
[[10  0  0]
 [ 0  9  1]
 [ 0  0 10]]
==========================================================
```

---

## 18. LIME Explanation Workflow

When a user submits features:

1.  Flask calls `explainer.explain_instance` on the input features.
2.  The resulting LIME object generates feature rule contributions (e.g., `petal length (cm) <= 1.60` with a weight of `+0.3928`).
3.  LIME writes the interactive graphical summary charts to `static/lime/explanation.html`.
4.  The Flask app identifies supporting and opposing rules and calculates a trustworthiness score based on prediction confidence.
5.  `index.html` renders the dynamic text narrative, color-coded rule cards, and the interactive LIME iframe.

---

## 19. Screenshots Section

_(Screen captures demonstrating the visual results of the application)_

### Dashboard Interface (Initial Form State)

![Initial Application State](docs/images/empty_form.png)
_Placeholder: Interface featuring empty inputs waiting for user parameter coordinates._

### Dashboard Interface (Classified Result & Explanation)

![Virginica Prediction and Explanation](docs/images/prediction_result_interpretation.png)
_Placeholder: The two-column dashboard displaying prediction results on the left and the LIME explanation panel on the right._

---

## 20. Future Improvements

- **Add global explainability:** Integrate SHAP to display global feature importance plots across the entire dataset.
- **Input Validation:** Add error-handling parameters to warn users about invalid entries (e.g., letters, negative numbers, empty values) without crashing the Flask thread.
- **Dynamic K Selection:** Allow users to adjust the parameter $K$ via the UI.
- **Model Comparison:** Allow users to choose between other classification algorithms (such as Decision Trees or SVMs) and compare their predictions and explanations side-by-side.

---

## 21. Learning Outcomes

- **Pipeline Engineering:** Packaged scaling steps and estimator weights inside a single, deployable pipeline object.
- **Local Proxy Interpretation:** Learned how local surrogates approximate complex decision boundaries.
- **Web Integration:** Learned how to parse model outputs and dynamically build natural-language interpretations.
- **Responsive UI Design:** Built a responsive, split-screen web application that serves dynamic data and static assets.

---

## 22. References

- Fisher, R. A. (1936). _The use of multiple measurements in taxonomic problems_. Annals of Eugenics, 7(2), 179-188.
- Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). _"Why should I trust you?": Explaining the predictions of any classifier_. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining.
- Scikit-Learn documentation on KNeighborsClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
- LIME Project Repository: https://github.com/marcotcr/lime

---

## 23. License

Delivered under the **MIT License**. You are free to modify, distribute, and execute this codebase for private, commercial, and educational activities.

---

## 24. Author

**Explainable AI Engineering Student & Mentee**  
_M.Sc. Artificial Intelligence and Explainable Machine Learning Assignment_
