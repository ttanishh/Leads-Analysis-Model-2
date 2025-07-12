from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os

# Load model
model_path = os.path.join("model", "lead_scoring_model.pkl")
model = joblib.load(model_path)

# Define binary map (same as used during training)
binary_map = {'Yes': 1, 'No': 0}
binary_cols = ['Took_Exam', 'Visa_Knowledge', 'Language_Test_Willing', 'Follows_Intl_Edu_Content']

# Flask app
app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return "🎯 Lead Scoring Model API is running!"

@app.route('/predict', methods=['POST'])
def predict_batch():
    try:
        data = request.get_json()
        df = pd.DataFrame(data)

        # Apply any necessary preprocessing if not inside the pipeline
        for col in ['Took_Exam', 'Visa_Knowledge', 'Language_Test_Willing', 'Follows_Intl_Edu_Content']:
            if col in df.columns:
                df[col] = df[col].map({'Yes': 1, 'No': 0}).fillna(0)

        # Predict
        scores = model.predict_proba(df)[:, 1]
        preds = model.predict(df)

        response = []
        for i in range(len(df)):
            response.append({
                "lead_score_percent": round(scores[i] * 100, 2),
                "predicted_interest": int(preds[i]),
                "lead_category": "High" if scores[i] > 0.7 else "Medium" if scores[i] > 0.4 else "Low"
            })

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)
