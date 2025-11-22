# 🌟 Nirmaan AI Communication Skills Scorer

## AI Intern Case Study Submission

This repository contains the end-to-end solution for the Nirmaan AI Intern Case Study: a tool designed to automatically analyze and score a student's spoken communication skills based on a text transcript and a complex, weighted rubric.

The solution uses a hybrid approach, combining traditional Rule-Based programming with modern Natural Language Processing (NLP) techniques to provide accurate, objective, and transparent scoring.

---

## 🎯 The Problem and Our Solution

**The Task:** Develop a system to evaluate a student's self-introduction transcript against a provided rubric, producing a final, normalized score (0-100) and criterion-specific feedback. The challenge was integrating both qualitative (flow, content relevance) and quantitative (WPM, grammar count) scoring criteria.

**Our Solution (The Hybrid Scorer):** We built a Python-based application using Flask to create a robust backend API (`app.py`) and a simple, intuitive web interface (`index.html`).

Our scoring function utilizes a **Hybrid Logic Model**:

1.  **Rule-Based Scoring:** Used for discrete, measurable elements like **Speech Rate (WPM)**, **Filler Word Rate**, and **Keyword Presence** (e.g., did they mention their name, school, and hobbies?).
2.  **NLP-Based Scoring:** Used for subjective criteria like **Content Flow** and **Relevance**. We use **Sentence Embeddings** (e.g., from Sentence Transformers) and **Cosine Similarity** to compare the transcript's semantic meaning against the rubric's "ideal response" description.

### Final Scoring Formula

The Overall Score is calculated as a **Weighted Average**, ensuring that criteria with higher importance (like Content) contribute proportionally more to the final grade:

$$\text{Overall Score} = \frac{\sum_{i=1}^{N} (\text{Normalized Score}_i \times \text{Criterion Weight}_i)}{\sum_{i=1}^{N} \text{Criterion Weight}_i} \times 100$$


[Image of a formula illustrating weighted average calculation]


---

## 💻 Project Files Explained

The solution is contained in three main files:

| File | Role | Description |
| :--- | :--- | :--- |
| **`app.py`** | **Backend Core Logic & API** | This is the heart of the application. It initializes the Flask server, loads the scoring rubric data, and implements the **`calculate_score(transcript)`** function. It exposes a primary API endpoint (e.g., `/api/score`) that receives the transcript and returns the detailed JSON score breakdown. |
| **`requirements.txt`** | **Dependencies** | Lists all necessary Python packages (e.g., `flask`, `numpy`, `pandas`, `scipy`, `scikit-learn`, `sentence-transformers`) required to run the scoring logic. |
| **`templates/index.html`** | **Frontend User Interface** | This file provides the clean, single-page web interface. It allows the user to paste the transcript, sends the text to the backend API via JavaScript, and then formats and displays the resulting overall score and per-criterion feedback. |

## 🛠️ Local Installation and Run Instructions

### Prerequisites

You must have **Python 3.9+** and `pip` installed.

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/chaithu10000/Nirmaan_AI_Scorer.git](https://github.com/chaithu10000/Nirmaan_AI_Scorer.git)
    cd Nirmaan_AI_Scorer
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Backend Server**
    Start the main application server:
    ```bash
    python app.py
    ```

4.  **Access the Application**
    Open your web browser and navigate to the application's URL. The application will be running locally at:

    ### **[http://127.0.0.1:5000/](http://127.0.0.1:5000/)**

---

### 📄 Detailed Documentation

For a full explanation of the deployment process, the specific NLP models used, and the derivation of the scoring formula, please refer to the dedicated document:

**[Deployment and Demo Explanation](Deployment_and_Demo_Explanation.md)**
