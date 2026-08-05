from flask import Flask, render_template, request
import pickle
import numpy as np
import os
import time
from sklearn.datasets import load_iris
from lime.lime_tabular import LimeTabularExplainer

app = Flask(__name__)

# Load the trained machine learning pipeline
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.pkl")
model = pickle.load(open(model_path, "rb"))

# Load Iris metadata to construct the LIME explainer
iris = load_iris()
feature_names = iris.feature_names
class_names = list(iris.target_names)
training_data = iris.data

# Initialize the LIME Tabular Explainer
explainer = LimeTabularExplainer(
    training_data=training_data,
    feature_names=feature_names,
    class_names=class_names,
    mode="classification",
    random_state=42
)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Read text variables from form client
    f1 = request.form["f1"]
    f2 = request.form["f2"]
    f3 = request.form["f3"]
    f4 = request.form["f4"]

    # Parse and structure input features to match the pipeline shape
    features = [float(f1), float(f2), float(f3), float(f4)]
    final_features = np.array(features).reshape(1, -1)

    # 1. Run classifier prediction
    prediction = model.predict(final_features)
    pred_class_idx = int(prediction[0])
    
    flower_types = ["Setosa", "Versicolor", "Virginica"]
    result = flower_types[pred_class_idx]

    # 2. Extract prediction probabilities
    probabilities = model.predict_proba(final_features)[0]
    pred_proba = probabilities[pred_class_idx]

    # 3. Generate LIME local explanation for the specific prediction instance
    exp = explainer.explain_instance(
        data_row=final_features[0],
        predict_fn=model.predict_proba,
        num_features=4,
        labels=(pred_class_idx,)
    )

    # Verify and establish static directory inside Flask framework paths
    static_lime_dir = os.path.join(app.root_path, "static", "lime")
    os.makedirs(static_lime_dir, exist_ok=True)
    
    # Save the LIME interactive HTML report output
    report_path = os.path.join(static_lime_dir, "explanation.html")
    exp.save_to_file(report_path)

    # Format the positive and negative feature contributions for direct UI render
    explanation_list = exp.as_list(label=pred_class_idx)
    features_contrib = []
    supporting_rules = []
    opposing_rules = []

    for rule, weight in explanation_list:
        is_positive = weight > 0
        features_contrib.append({
            'rule': rule,
            'weight': round(weight, 4),
            'is_positive': is_positive
        })
        if is_positive:
            supporting_rules.append(rule)
        else:
            opposing_rules.append(rule)

    # Identify strongest and weakest contributors based on absolute weights
    sorted_by_abs = sorted(explanation_list, key=lambda x: abs(x[1]), reverse=True)
    strongest_rule = sorted_by_abs[0][0] if sorted_by_abs else "N/A"
    weakest_rule = sorted_by_abs[-1][0] if sorted_by_abs else "N/A"

    # Evaluate dynamic trust metrics
    if pred_proba >= 0.85:
        trust_level = "High"
        trust_reason = f"The model exhibits high confidence ({pred_proba * 100:.1f}%), backed by strong positive indicators such as {', and '.join(supporting_rules[:2]) if supporting_rules else 'features'} without significant negative features."
    elif pred_proba >= 0.60:
        trust_level = "Moderate"
        trust_reason = f"The model shows moderate confidence ({pred_proba * 100:.1f}%). The feature requirements are generally satisfied, but minor opposing features exist or the nearest neighbor votes are divided."
    else:
        trust_level = "Low"
        trust_reason = f"The model presents low confidence ({pred_proba * 100:.1f}%). There is high ambiguity in the local feature space, indicating potential misclassification."

    # Build dynamic interpretation narrative dict
    why_predicted = f"The sample matches parameters for species '{result}' with a confidence of {pred_proba * 100:.2f}%. This decision was primarily driven by features satisfying the criteria: {', '.join(supporting_rules[:2]) if supporting_rules else 'N/A'}"

    narrative = {
        'why_predicted': why_predicted,
        'most_contributed': strongest_rule,
        'least_contributed': weakest_rule,
        'supporting': supporting_rules,
        'opposing': opposing_rules,
        'trustworthiness_level': trust_level,
        'trustworthiness_reason': trust_reason,
        'explanation_scope': 'Local'
    }

    image_map = {
        "Setosa": "https://thumbs.dreamstime.com/b/red-yellow-flowers-garden-echeveria-setosa-echeveria-setosa-flowers-236287858.jpg",
        "Versicolor": "https://cdn.pixabay.com/photo/2015/05/08/01/15/iris-versicolor-757440_960_720.jpg",
        "Virginica": "https://tse3.mm.bing.net/th/id/OIP.d_3k01dQBa-icusU2tcjOgHaFj?pid=Api&P=0&h=180"
    }

    flower_img = image_map[result]

    return render_template(
        "index.html",
        prediction_text="Predicted Flower: " + result,
        prediction_proba=f"{pred_proba * 100:.2f}%",
        flower_image=flower_img,
        features_contrib=features_contrib,
        narrative=narrative,
        f1=f1,
        f2=f2,
        f3=f3,
        f4=f4,
        t=int(time.time())
    )

if __name__ == "__main__":
    app.run(debug=True)
