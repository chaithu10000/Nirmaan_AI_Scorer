import json
from flask import Flask, request, jsonify, render_template
import numpy as np
import re
from collections import Counter
import language_tool_python

# --- NLP / SEMANTIC SCORING LIBRARIES ---
# IMPORTANT: Try to import SentenceTransformer and handle the ImportError if not installed.
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    SEMANTIC_SCORING_ENABLED = True
    print("Loading Sentence Transformer model: all-MiniLM-L6-v2...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Model loaded successfully.")
except ImportError:
    SEMANTIC_SCORING_ENABLED = False
    print("WARNING: Sentence-Transformers or Scikit-learn not found. Semantic scoring (Flow) will be disabled.")
except Exception as e:
    SEMANTIC_SCORING_ENABLED = False
    print(f"ERROR: Failed to load Sentence Transformer model. Semantic scoring disabled. Error: {e}")

# --- GRAMMAR CHECKING LIBRARY ---
try:
    # Initialize the LanguageTool. This can take a moment.
    grammar_tool = language_tool_python.LanguageTool('en-US')
    GRAMMAR_CHECK_ENABLED = True
    print("Language Tool for Grammar initialized.")
except Exception as e:
    GRAMMAR_CHECK_ENABLED = False
    print(f"WARNING: language-tool-python failed to initialize. Grammar scoring disabled. Error: {e}")


app = Flask(__name__, template_folder='templates')

# --- CONFIGURATION (RUBRIC & TARGETS) ---

# Target structure for semantic flow check
TARGET_FLOW_DESCRIPTION = (
    "A self-introduction must follow a logical order: Salutation/Greeting, "
    "stating Name, Age, and Mandatory Details (Class, School), "
    "followed by Optional Details (Family, Hobbies, Fun Fact/Unique Point), "
    "and concluding with a polite Closing/Thank You."
)

# List of keywords required for the Key Content Presence (30 points)
REQUIRED_KEYWORDS = [
    'name', 'age', 'class', 'school', 'family', 'hobbies', 'interests', 'goals', 'unique point', 'subject', 'cricket', 'kind hearted', 'soft spoken'
]

# List of common filler words for Clarity (10 points)
FILLER_WORDS = set([
    'um', 'uh', 'like', 'you know', 'so', 'actually', 'basically', 'right', 
    'i mean', 'well', 'kinda', 'sort of', 'okay', 'hmm', 'ah', 'and then', 
    'at the end of the day', 'literally'
])

# --- SCORING LOGIC ---

def calculate_key_content_presence(transcript):
    """
    Criterion: Key Content Presence (Weight: 30)
    Metric: Presence of REQUIRED_KEYWORDS.
    """
    transcript_lower = transcript.lower()
    found_count = 0
    total_required = len(REQUIRED_KEYWORDS)
    found_details = []

    for keyword in REQUIRED_KEYWORDS:
        if keyword in transcript_lower:
            found_count += 1
            found_details.append(keyword)

    # Score is proportional to the percentage of keywords found (normalized to 100)
    score_raw = (found_count / total_required) * 100
    
    # Apply floor and weight to 30 points
    weight = 30
    score = score_raw
    
    if score >= 90: # All major points covered
        feedback = f"Excellent content coverage! Found {found_count}/{total_required} key elements, including: {', '.join(found_details[:5])}..."
    elif score >= 60:
        feedback = f"Good coverage. Found {found_count}/{total_required} key elements. Consider adding more details about goals or a unique point."
    else:
        feedback = f"Low coverage. Found only {found_count}/{total_required} key elements. Ensure you mention name, class, school, family, and hobbies."
        
    return {
        "name": "Key Content Presence",
        "score": round(score),
        "weight": weight,
        "feedback": feedback,
        "raw_score_30": (score / 100) * weight # Weighted score out of 30
    }

def calculate_flow_and_organization(transcript):
    """
    Criterion: Flow (Weight: 5)
    Metric: Semantic similarity to the ideal structure/flow.
    """
    weight = 5
    if not SEMANTIC_SCORING_ENABLED:
        # Fallback if NLP model failed to load
        score = 50
        feedback = "Flow score is estimated (50/100) because the Sentence-Transformers library is not available for semantic analysis. Please install it for accurate results."
        return {
            "name": "Flow & Organization (Semantic)",
            "score": score,
            "weight": weight,
            "feedback": feedback,
            "raw_score_5": (score / 100) * weight
        }

    # 1. Embed the transcript and the target description
    embeddings = model.encode([transcript, TARGET_FLOW_DESCRIPTION])
    
    # 2. Calculate cosine similarity between the transcript and the target flow
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    
    # 3. Normalize similarity (0 to 1) to a score (0 to 100).
    score_raw = max(0, similarity * 125 - 50)
    score = min(100, score_raw)

    if score >= 80:
        feedback = f"Excellent structure. The introduction follows a logical, organized flow. Semantic similarity: {similarity:.2f}"
    elif score >= 50:
        feedback = f"Good structure. The major sections are present, but the order could be improved for a smoother presentation. Semantic similarity: {similarity:.2f}"
    else:
        feedback = f"The flow is confusing. Try to reorder sections to follow the standard self-introduction structure. Semantic similarity: {similarity:.2f}"
        
    return {
        "name": "Flow & Organization (Semantic)",
        "score": round(score),
        "weight": weight,
        "feedback": feedback,
        "raw_score_5": (score / 100) * weight
    }

def calculate_type_token_ratio(transcript):
    """
    Criterion: Vocabulary Richness (TTR) (Weight: 10)
    Metric: Type-Token Ratio (TTR = Distinct words / Total words)
    """
    weight = 10
    
    # Clean the transcript (remove punctuation)
    words = re.findall(r'\b\w+\b', transcript.lower())
    total_words = len(words)
    distinct_words = len(set(words))
    
    if total_words == 0:
        ttr = 0
    else:
        ttr = distinct_words / total_words

    # Score mapping based on the rubric (TTR range -> Score 0-100 equivalent)
    if ttr >= 0.9:
        score = 100  # Rubric: 0.9 - 1.0
        feedback = f"Excellent vocabulary (TTR: {ttr:.2f}). You used a high diversity of words."
    elif ttr >= 0.7:
        score = 80   # Rubric: 0.7 - 0.89
        feedback = f"Good vocabulary (TTR: {ttr:.2f}). The word choices are diverse."
    elif ttr >= 0.5:
        score = 60   # Rubric: 0.5 - 0.69
        feedback = f"Average vocabulary (TTR: {ttr:.2f}). Consider using more varied language."
    elif ttr >= 0.3:
        score = 40   # Rubric: 0.3 - 0.49
        feedback = f"Limited vocabulary (TTR: {ttr:.2f}). Too much repetition of common words."
    else:
        score = 20   # Rubric: 0 - 0.29
        feedback = f"Very limited vocabulary (TTR: {ttr:.2f}). Needs significant improvement in word choice variety."
        
    return {
        "name": "Vocabulary Richness (TTR)",
        "score": score,
        "weight": weight,
        "feedback": feedback,
        "raw_score_10": (score / 100) * weight
    }

def calculate_grammar_score(transcript, total_words):
    """
    Criterion: Language & Grammar (Weight: 10)
    Metric: Errors per 100 words using LanguageTool.
    """
    weight = 10
    
    if not GRAMMAR_CHECK_ENABLED:
        score = 50
        feedback = "Grammar score is estimated (50/100) because language-tool-python is not available. Please install and initialize it for accurate results."
        return {
            "name": "Language & Grammar (Error Count)",
            "score": score,
            "weight": weight,
            "feedback": feedback,
            "raw_score_10": (score / 100) * weight
        }

    if total_words < 5:
        return {
            "name": "Language & Grammar (Error Count)",
            "score": 100, # Max score for negligible sample
            "weight": weight,
            "feedback": "Transcript is too short to reliably assess grammar. Assuming perfect score.",
            "raw_score_10": weight
        }

    # Use the Language Tool to find matches (errors)
    matches = grammar_tool.check(transcript)
    error_count = len(matches)
    
    # Calculate errors per 100 words
    errors_per_100 = (error_count / total_words) * 100 if total_words > 0 else 0
    
    # Rubric: Score = 1 - min(errors_per_100 / 10, 1)
    rubric_score_0_1 = 1 - min(errors_per_100 / 10, 1)
    score = round(rubric_score_0_1 * 100)

    # Compile a list of specific error messages for feedback
    error_messages = [f"'{m.context}' -> Suggestion: {m.replacements}" for m in matches[:5]]
    
    if score >= 90:
        feedback = f"Excellent grammar (Errors/100 words: {errors_per_100:.2f}). Total errors found: {error_count}."
    elif score >= 50:
        feedback = f"Minor grammatical issues (Errors/100 words: {errors_per_100:.2f}). Total errors found: {error_count}. Review sentences like: {'; '.join(error_messages)}."
    else:
        feedback = f"Significant grammar issues (Errors/100 words: {errors_per_100:.2f}). Total errors found: {error_count}. Major errors include: {'; '.join(error_messages)}."
        
    return {
        "name": "Language & Grammar (Error Count)",
        "score": score,
        "weight": weight,
        "feedback": feedback,
        "raw_score_10": (score / 100) * weight
    }

def calculate_filler_word_rate(transcript, total_words):
    """
    Criterion: Clarity (Filler Word Rate) (Weight: 5) - Adjusted to 5 for total weight 60
    Metric: Filler Word Rate = (Number of filler words / Total words) * 100
    """
    weight = 5 # Using 5 to make total weight 60 (30+5+10+10+5)

    if total_words == 0:
        return {
            "name": "Clarity (Filler Word Rate)",
            "score": 100,
            "weight": weight,
            "feedback": "Transcript is empty. Assuming perfect clarity.",
            "raw_score_5": weight
        }
        
    transcript_words = re.findall(r'\b\w+\b', transcript.lower())
    
    filler_word_count = sum(1 for word in transcript_words if word in FILLER_WORDS)
    
    filler_rate = (filler_word_count / total_words) * 100

    # Rubric Mapping (Inverse relationship: lower rate = higher score)
    # Target: < 1% (score 100)
    if filler_rate <= 1.0:
        score = 100
        feedback = f"Excellent clarity ({filler_rate:.2f}% filler rate). No unnecessary filler words found."
    elif filler_rate <= 3.0:
        score = 80
        feedback = f"Good clarity ({filler_rate:.2f}% filler rate). Few minor fillers found ({filler_word_count} total). Try to eliminate these."
    elif filler_rate <= 5.0:
        score = 60
        feedback = f"Moderate clarity ({filler_rate:.2f}% filler rate). {filler_word_count} fillers found. Focus on speaking more directly."
    elif filler_rate <= 10.0:
        score = 40
        feedback = f"Low clarity ({filler_rate:.2f}% filler rate). {filler_word_count} fillers found. This significantly impacts perceived confidence."
    else:
        score = 20
        feedback = f"Very low clarity ({filler_rate:.2f}% filler rate). Excessive use of filler words ({filler_word_count} total). Needs immediate attention."

    return {
        "name": "Clarity (Filler Word Rate)",
        "score": score,
        "weight": weight,
        "feedback": feedback,
        "raw_score_5": (score / 100) * weight
    }

def calculate_final_score(transcript):
    """
    The main function to calculate all criterion scores and the overall weighted score.
    """
    # Pre-processing
    words = re.findall(r'\b\w+\b', transcript.lower())
    total_words = len(words)
    
    if total_words < 10:
        # Prevent scoring for transcripts that are too short to be meaningful
        raise ValueError("Transcript is too short. Please provide at least 10 words for a meaningful analysis.")
    
    # 1. Calculate Per-Criterion Scores (Normalized 0-100)
    content_score = calculate_key_content_presence(transcript) 
    flow_score = calculate_flow_and_organization(transcript)     
    ttr_score = calculate_type_token_ratio(transcript)           
    grammar_score = calculate_grammar_score(transcript, total_words) 
    clarity_score = calculate_filler_word_rate(transcript, total_words) 

    # 2. Compile All Criterion Results
    all_criteria = [
        content_score,
        flow_score,
        ttr_score,
        grammar_score,
        clarity_score
    ]

    # 3. Calculate Overall Weighted Score (Total Weight = 60)
    target_weights = {
        "Key Content Presence": 30,
        "Flow & Organization (Semantic)": 5,
        "Vocabulary Richness (TTR)": 10,
        "Language & Grammar (Error Count)": 10,
        "Clarity (Filler Word Rate)": 5 
    }
    total_raw_score_60 = 0
    total_possible_weight = 60
    
    for criterion in all_criteria:
        weight = target_weights.get(criterion["name"], 0)
        criterion["weight"] = weight # Update weight in criteria list for UI reporting
        total_raw_score_60 += (criterion["score"] / 100) * weight
    
    overall_score = round((total_raw_score_60 / total_possible_weight) * 100) if total_possible_weight > 0 else 0
    
    return {
        "overall_score": min(100, overall_score), # Cap score at 100
        "word_count": total_words,
        "per_criterion": all_criteria,
        "total_possible_weight": total_possible_weight
    }


# --- FLASK ROUTES ---

@app.route('/')
def index():
    """Renders the HTML front-end."""
    return render_template('index.html')

@app.route('/score', methods=['POST'])
def score():
    """API endpoint to receive transcript and return scores."""
    try:
        data = request.get_json()
        transcript = data.get('transcript', '')

        if not transcript:
            return jsonify({"error": "No transcript text provided."}), 400

        results = calculate_final_score(transcript)
        return jsonify(results)
    
    except ValueError as e:
        # Handle custom validation errors (e.g., transcript too short)
        return jsonify({"error": str(e)}), 400
    
    except Exception as e:
        # Log unexpected errors and return a 500
        app.logger.error(f"An unexpected error occurred during scoring: {e}")
        return jsonify({"error": f"Internal Server Error: {e}"}), 500

if __name__ == '__main__':
    # Using 0.0.0.0 allows it to be accessed via local IP in addition to localhost
    app.run(host='0.0.0.0', port=5000, debug=True)