# Deployment and Demo Explanation

This document serves two primary purposes for the Nirmaan AI Intern Case Study submission:
1.  To detail the **step-by-step process** for running the application on a local server.
2.  To provide the **comprehensive technical explanation** of the scoring formula and logic implemented in `app.py`.

## 1. Local Server Deployment Steps

The application is built on a Python backend (Flask) and a light HTML/JavaScript frontend, making it easy to deploy locally.

### Environment and Structure

The core functionality relies on:
* **`app.py`:** Initializes the Flask server and hosts the scoring API endpoint.
* **`templates/index.html`:** The user interface for transcript input and score output display.
* **`requirements.txt`:** Lists Python libraries like `flask`, `pandas`, and `sentence-transformers`.
* **Rubric Data:** Assumes the rubric data (e.g., weights, keyword lists) has been loaded or hardcoded into `app.py` for calculation.

### Step-by-Step Execution

1.  **Preparation:** Ensure you have completed the prerequisite setup (Python installed, repository cloned, dependencies installed via `pip install -r requirements.txt`).
2.  **Start the Server:** Open your terminal in the project directory and execute the backend script:
    ```bash
    python app.py
    ```
    The console will confirm the server is running, typically at `http://127.0.0.1:5000/`.
3.  **Access the Interface:** Open the URL in your web browser.
4.  **Demonstration:** Paste a sample transcript (such as the self-introduction example from the case study) into the input box and click the **Analyze** button. The JavaScript in `index.html` sends the text to the `/api/score` endpoint in `app.py`, which returns the detailed JSON result.

## 2. Detailed Scoring Formula and Logic

The final score is a highly accurate, weighted assessment derived from a hybrid scoring model.

### A. The Overall Score Formula

The total score (normalized 0-100) is a weighted average of all calculated criterion scores. This is necessary because some criteria (like Content) hold significantly more weight than others (like Clarity).

$$\text{Overall Score} = \frac{\sum_{i=1}^{N} (\text{Normalized Score}_i \times \text{Criterion Weight}_i)}{\sum_{i=1}^{N} \text{Criterion Weight}_i} \times 100$$


[Image of a weighted average formula diagram]


* **$\text{Normalized Score}_i$:** The score for an individual criterion, scaled to a 0-1 range based on rubric tiers.
* **$\text{Criterion Weight}_i$:** The numerical weight assigned to that criterion from the rubric.

### B. Scoring Logic by Criterion

The core logic within `app.py` processes each criterion as follows:

| Criterion | Logic Type | Calculation and Role |
| :--- | :--- | :--- |
| **Content & Structure** | **Hybrid (Rule-Based & NLP)** | **Role:** The highest-weighted section. Evaluates completeness and quality of self-introduction elements. **Logic:** Combines **Keyword Presence** (Rule-Based count of Name, School, Age) with **Semantic Flow Score** (NLP-Based Cosine Similarity comparing the transcript's embedding to the "ideal flow" embedding). |
| **Language & Grammar** | **Rule-Based** | **Role:** Assess vocabulary richness and error rate. **Logic:** Calculated by: 1) **Type-Token Ratio (TTR)** for vocabulary diversity ($\text{TTR} = \frac{\text{Distinct Words}}{\text{Total Words}}$). 2) **Grammar Error Count** (simulated using regex for common errors or external NLP tools like LanguageTool). |
| **Speech Rate** | **Rule-Based** | **Role:** Ensure a conversational and understandable pace. **Logic:** Calculated as Words Per Minute (WPM) and mapped directly to the rubric's score bands. |
| **Clarity** | **Rule-Based** | **Role:** Evaluate fluency and professionalism. **Logic:** Calculated by the **Filler Word Rate** (count of words like "um," "uh," "like," relative to total word count). The score is inversely proportional to the rate. |

### C. Demonstration Script Summary

The screen recording demonstration will follow these steps:
1.  Confirm the server is running locally (terminal screen).
2.  Navigate to the local URL (`http://127.0.0.1:5000/`).
3.  Paste the sample transcript and click **Analyze**.
4.  Highlight and explain the **Overall Score** result based on the weighted formula.
5.  Scroll to the breakdown, focusing on the **Content & Structure** score and explaining that it passed both the Rule-Based (keyword) and NLP-Based (similarity) checks.
6.  Conclude by confirming the application successfully implemented the complex scoring logic as required by the case study.