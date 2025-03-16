from flask import Flask, render_template, request
import joblib
import pandas as pd
import os

app = Flask(__name__)

# Load trained models
models_dir = "models"
dt_model = joblib.load(os.path.join(models_dir, "decision_tree.pkl"))
rf_model = joblib.load(os.path.join(models_dir, "random_forest.pkl"))
nn_model = joblib.load(os.path.join(models_dir, "neural_network.pkl"))
scaler = joblib.load(os.path.join(models_dir, "scaler.pkl"))
label_encoder = joblib.load(os.path.join(models_dir, "label_encoder.pkl"))

# Load feature names from dataset
features = [col for col in pd.read_csv("CareerMap- Mapping Tech Roles With Personality & Skills.csv").columns if col != "Role"]

# Explanations for skills & personality traits
feature_descriptions = {
    "Database Fundamentals": "Knowledge of databases, SQL, and data management.",
    "Computer Architecture": "Understanding of system design, hardware, and performance.",
    "Distributed Computing Systems": "Working with cloud computing, networking, and clusters.",
    "Cyber Security": "Ability to protect systems and networks from cyber threats.",
    "Networking": "Understanding network protocols, routing, and connectivity.",
    "Software Development": "Building applications using programming languages.",
    "Programming Skills": "Proficiency in languages like Python, Java, or C++.",
    "Project Management": "Planning and managing software projects efficiently.",
    "Computer Forensics Fundamentals": "Investigating and analyzing cyber threats.",
    "Technical Communication": "Writing and explaining technical concepts clearly.",
    "AI ML": "Understanding Artificial Intelligence and Machine Learning concepts.",
    "Software Engineering": "Applying engineering principles to software development.",
    "Business Analysis": "Understanding business requirements and translating them into technical solutions.",
    "Communication skills": "Effectively conveying ideas and collaborating with teams.",
    "Data Science": "Analyzing and interpreting complex data to extract insights.",
    "Troubleshooting skills": "Solving technical issues effectively and efficiently.",
    "Graphics Designing": "Creating digital art, UI/UX design, and branding.",
    "Openness": "Willingness to try new experiences and think creatively.",
    "Conscientousness": "Being organized and responsible in work.",
    "Extraversion": "Enjoying teamwork, collaboration, and social interactions.",
    "Agreeableness": "Being cooperative and empathetic in work settings.",
    "Emotional_Range": "Ability to handle stress and work under pressure.",
    "Conversation": "Engaging in meaningful discussions with clarity.",
    "Openness to Change": "Ability to adapt to new technologies and workflows.",
    "Hedonism": "Enjoying challenges and creative problem-solving.",
    "Self-enhancement": "Striving for personal and professional growth.",
    "Self-transcendence": "Working for the greater good and innovation."
}

# Career descriptions
career_descriptions = {
    "Software Developer": "Develops and maintains software applications. Requires strong programming and problem-solving skills.",
    "Data Scientist": "Analyzes large data sets to find insights. Requires knowledge of statistics, ML, and data visualization.",
    "Cyber Security Specialist": "Protects systems from cyber threats. Requires expertise in security tools and risk management.",
    "AI/ML Engineer": "Designs AI-driven applications. Requires deep knowledge of ML models and deployment techniques.",
    "Customer Service Executive": "Handles customer queries, resolves complaints, and improves user experience.",
}

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/form")
def form():
    return render_template("form.html", features=features, descriptions=feature_descriptions)

@app.route("/predict", methods=["POST"])
def predict():
    # Collect user input
    user_input = [float(request.form[feature]) for feature in features]
    user_df = pd.DataFrame([user_input], columns=features)
    user_scaled = scaler.transform(user_df)

    # Get predictions with confidence scores
    dt_pred = dt_model.predict_proba(user_scaled)[0]
    rf_pred = rf_model.predict_proba(user_scaled)[0]
    nn_pred = nn_model.predict_proba(user_scaled)[0]

    # Convert predictions to career labels
    dt_career = label_encoder.inverse_transform([dt_model.predict(user_scaled)[0]])[0]
    rf_career = label_encoder.inverse_transform([rf_model.predict(user_scaled)[0]])[0]
    nn_career = label_encoder.inverse_transform([nn_model.predict(user_scaled)[0]])[0]

    # Get confidence scores
    dt_confidence = round(dt_pred.max() * 100, 2)
    rf_confidence = round(rf_pred.max() * 100, 2)
    nn_confidence = round(nn_pred.max() * 100, 2)

    # Find the best career match
    career_predictions = {
        dt_career: dt_confidence,
        rf_career: rf_confidence,
        nn_career: nn_confidence
    }
    best_career = max(career_predictions, key=career_predictions.get)
    best_confidence = career_predictions[best_career]

    # Ensure a career description exists
    career_description = career_descriptions.get(best_career, "This is an exciting career! Have a great journey.")

    # Generate learning resources
    resources = generate_resources(best_career)

    return render_template("results.html", best_career=best_career, best_confidence=best_confidence, career_description=career_description, resources=resources)

def generate_resources(career):
    """Return relevant learning resources based on the predicted career."""
    learning_paths = {
        "Software Developer": [
            "https://www.freecodecamp.org/",
            "https://www.udemy.com/course/python-for-beginners/",
            "https://leetcode.com/"
        ],
        "Data Scientist": [
            "https://www.kaggle.com/",
            "https://www.coursera.org/learn/machine-learning",
            "https://www.udacity.com/course/data-scientist-nanodegree--nd025"
        ],
        "Cyber Security Specialist": [
            "https://www.cybrary.it/",
            "https://www.udemy.com/course/the-complete-cyber-security-course/",
            "https://www.comptia.org/certifications/security"
        ],
        "AI/ML Engineer": [
            "https://www.deeplearning.ai/",
            "https://www.udacity.com/course/intro-to-machine-learning--ud120",
            "https://developers.google.com/machine-learning"
        ]
    }
    return learning_paths.get(career, ["https://www.coursera.org/", "https://www.udemy.com/", "https://www.linkedin.com/learning/"])

if __name__ == "__main__":
    app.run(debug=True)
