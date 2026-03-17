
# importing important libraries
from flask import Flask, request, render_template, flash, session, jsonify
from markupsafe import Markup  # Correct import for Markup
# importing important libraries
from flask import Flask, request, render_template, flash, session, jsonify
from markupsafe import Markup
import html  # Required for html.escape()
# importing file in which our ml-algorithms are residing
from models import Model
import os
import re
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from flask import render_template_string

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

app = Flask(__name__)

# Initialize text preprocessing tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


# Suicidal ideation keywords and patterns
SUICIDAL_KEYWORDS = {
    'high': [
        'kill myself', 'end my life', 'commit suicide', 'take my own life',
        'want to die', 'better off dead', 'suicidal', 'ending it all',
        'no reason to live', 'worthless', 'hopeless', 'cant go on',
        'give up on life', 'ready to die', 'death is the only option',
        'i want to die', 'i will kill myself', 'plan to commit suicide',
        'going to end it', 'life is not worth living', 'i should die',
        'wish i was dead', 'want to end my life', 'thinking of suicide'
    ],
    'medium': [
        'hate myself', 'no hope', 'tired of living', 'pain inside',
        'empty inside', 'numb', 'cry for help', 'save me',
        'drowning', 'suffocating', 'alone in the world', 'burden to others',
        'no one cares', 'everyone hates me', 'i give up', 'can\'t do this anymore',
        'fed up with life', 'tired of everything', 'no way out'
    ],
    'low': [
        'sad', 'depressed', 'lonely', 'stressed', 'anxious',
        'overwhelmed', 'exhausted', 'helpless', 'broken',
        'lost', 'struggling', 'fighting depression', 'feeling down',
        'feeling low', 'mental health', 'not okay', 'struggling mentally'
    ]
}

# Suicide-related patterns (regex) - UPDATED VERSION
SUICIDE_PATTERNS = [
    # Original patterns
    r'\b(kill|end|take).{0,20}(life|myself|own)\b',
    r'\b(commit|contemplate|consider).{0,20}(suicide)\b',
    r'\b(want|going).{0,20}(die|end it)\b',
    r'\b(no).{0,20}(reason|point).{0,20}(live)\b',
    r'\b(better off).{0,20}(dead)\b',
    r'\b(how to).{0,20}(kill|suicide|die)\b',
    r'\b(suicide).{0,20}(methods|ways|plan|thoughts)\b',
    r'\b(last).{0,20}(time|day|night)\b.{0,20}(live)?',
    r'\b(can\'t|cannot).{0,20}(go on|continue)\b',
    r'\b(thinking about).{0,20}(dying|death|ending it)\b',
    
    # NEW PATTERNS - Critical for your type of message
    r'(world|everyone|people).{0,20}(better (off)?).{0,20}(without me|if i (was|were) (gone|dead|not here))',
    r'(better (off)?).{0,20}(without me|if i (was|were) (gone|dead))',
    r'(no one|nobody).{0,20}(would|will).{0,20}(care|miss|notice)',
    r'(feel|being|am).{0,20}(a|like a).{0,20}(burden|weight).{0,20}(on|to).{0,20}(others|everyone|family|anyone)',
    r'(sadness|pain|emptiness|hurt).{0,20}(keeps|continues|just keeps).{0,20}(growing|getting worse|increasing)',
    r'(fake|pretend|put on).{0,20}(smile|happy|happiness|emotions)',
    r'(inside|deep down).{0,20}(i).{0,20}(feel|am).{0,20}(empty|hollow|broken|dying)',
    r'(world|life).{0,20}(would be|is).{0,20}(better|easier).{0,20}(without me|if i (wasn\'t|weren\'t) here)',
    r'(everyone).{0,20}(would be).{0,20}(better off).{0,20}(without me)',
    r'(i\'?m?).{0,20}(just).{0,20}(a|such a).{0,20}(burden|problem|nuisance|failure)',
]

class SuicideDetectionModel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.model = None
        self.is_trained = False
        
    def preprocess_text(self, text):
        """Preprocess text for analysis"""
        try:
            # Convert to lowercase
            text = text.lower()
            
            # Remove special characters and digits
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            
            # Tokenize
            tokens = word_tokenize(text)
            
            # Remove stopwords and lemmatize
            tokens = [lemmatizer.lemmatize(token) for token in tokens 
                     if token not in stop_words and len(token) > 2]
            
            return ' '.join(tokens)
        except Exception as e:
            print(f"Error in preprocessing: {e}")
            return text
    
    def highlight_keywords(self, text):
        """Highlight keywords in text based on risk level"""
        try:
            text_lower = text.lower()
            highlighted_text = html.escape(text)  # Escape HTML characters first
            
            # Sort keywords by length (longest first) to avoid partial highlighting issues
            all_keywords = []
            
            # Add high-risk keywords with high priority
            for keyword in SUICIDAL_KEYWORDS['high']:
                all_keywords.append((keyword, 'high', 3))
            
            # Add medium-risk keywords
            for keyword in SUICIDAL_KEYWORDS['medium']:
                all_keywords.append((keyword, 'medium', 2))
            
            # Add low-risk keywords
            for keyword in SUICIDAL_KEYWORDS['low']:
                all_keywords.append((keyword, 'low', 1))
            
            # Sort by length (longest first) to prevent nested highlighting issues
            all_keywords.sort(key=lambda x: len(x[0]), reverse=True)
            
            # Create a copy for highlighting
            result_text = highlighted_text
            
            # Track highlighted positions to avoid overlapping highlights
            highlights = []
            
            # Find all keyword occurrences
            for keyword, risk_level, priority in all_keywords:
                # Find all occurrences of the keyword (case-insensitive)
                pattern = re.compile(re.escape(keyword), re.IGNORECASE)
                for match in pattern.finditer(text):
                    start, end = match.start(), match.end()
                    
                    # Check if this position is already highlighted with higher priority
                    overlap = False
                    for h_start, h_end, h_priority in highlights:
                        if (start < h_end and end > h_start):  # Overlap detected
                            if priority <= h_priority:  # Current priority is lower or equal
                                overlap = True
                                break
                            else:  # Current priority is higher, remove the lower priority highlight
                                highlights.remove((h_start, h_end, h_priority))
                    
                    if not overlap:
                        highlights.append((start, end, priority, risk_level, keyword))
            
            # Sort highlights by start position (reverse to avoid index shifting)
            highlights.sort(key=lambda x: x[0], reverse=True)
            
            # Apply highlights
            for start, end, priority, risk_level, keyword in highlights:
                original_substring = text[start:end]
                escaped_substring = html.escape(original_substring)
                
                # Define color based on risk level
                if risk_level == 'high':
                    color = '#ff4444'  # Bright red
                    bg_color = '#ffebee'  # Light red background
                    title = 'High Risk Keyword'
                elif risk_level == 'medium':
                    color = '#ff8800'  # Orange
                    bg_color = '#fff3e0'  # Light orange background
                    title = 'Medium Risk Keyword'
                else:
                    color = '#ffbb33'  # Yellow
                    bg_color = '#fff9e6'  # Light yellow background
                    title = 'Low Risk Keyword'
                
                # Create highlighted span
                highlighted = f'<span class="highlighted-keyword" style="background-color: {bg_color}; color: {color}; font-weight: bold; padding: 2px 4px; border-radius: 3px; border-left: 3px solid {color};" title="{title}: {keyword}">{escaped_substring}</span>'
                
                # Replace in result_text (need to account for HTML escaping)
                # Find the position in the escaped text
                escaped_before = html.escape(text[:start])
                escaped_after = html.escape(text[end:])
                escaped_keyword = html.escape(original_substring)
                
                # Reconstruct with highlighting
                result_text = escaped_before + highlighted + escaped_after
            
            # Add pattern matches highlighting
            for i, pattern in enumerate(SUICIDE_PATTERNS):
                try:
                    for match in re.finditer(pattern, text_lower, re.IGNORECASE):
                        matched_text = match.group()
                        if matched_text and len(matched_text) > 3:  # Avoid very short matches
                            # Check if this pattern match overlaps with existing highlights
                            start = text_lower.find(matched_text)
                            if start >= 0:
                                end = start + len(matched_text)
                                
                                # Only add if not already highlighted with higher priority
                                should_add = True
                                for h_start, h_end, h_priority, _, _ in highlights:
                                    if start < h_end and end > h_start and h_priority >= 2:
                                        should_add = False
                                        break
                                
                                if should_add:
                                    original_substring = text[start:end]
                                    escaped_substring = html.escape(original_substring)
                                    highlighted = f'<span class="pattern-match" style="background-color: #e1f5fe; color: #0277bd; font-weight: bold; padding: 2px 4px; border-radius: 3px; border-left: 3px solid #0277bd; text-decoration: underline wavy #0277bd;" title="Pattern Match: {pattern}">{escaped_substring}</span>'
                                    
                                    # Replace in result_text
                                    escaped_before = html.escape(text[:start])
                                    escaped_after = html.escape(text[end:])
                                    result_text = escaped_before + highlighted + escaped_after
                except:
                    continue
            
            return Markup(result_text)
            
        except Exception as e:
            print(f"Error in highlighting: {e}")
            return Markup(html.escape(text))
    
    def analyze_suicidal_content(self, text):
        """Analyze text for suicidal content using keyword matching and pattern recognition"""
        try:
            text_lower = text.lower()
            
            # Check for high-risk keywords
            high_risk_matches = []
            for keyword in SUICIDAL_KEYWORDS['high']:
                if keyword in text_lower:
                    high_risk_matches.append(keyword)
            
            # Check for medium-risk keywords
            medium_risk_matches = []
            for keyword in SUICIDAL_KEYWORDS['medium']:
                if keyword in text_lower:
                    medium_risk_matches.append(keyword)
            
            # Check for low-risk keywords
            low_risk_matches = []
            for keyword in SUICIDAL_KEYWORDS['low']:
                if keyword in text_lower:
                    low_risk_matches.append(keyword)
            
            # Enhanced pattern matching
            ENHANCED_PATTERNS = [
                # Direct suicidal statements
                r'\b(kill|end|take).{0,20}(life|myself|own)\b',
                r'\b(commit|contemplate|consider).{0,20}(suicide)\b',
                r'\b(want|going).{0,20}(die|end it)\b',
                r'\b(no).{0,20}(reason|point).{0,20}(live)\b',
                r'\b(better off).{0,20}(dead)\b',
                r'\b(how to).{0,20}(kill|suicide|die)\b',
                r'\b(suicide).{0,20}(methods|ways|plan|thoughts)\b',
                r'\b(last).{0,20}(time|day|night)\b.{0,20}(live)?',
                r'\b(can\'t|cannot).{0,20}(go on|continue)\b',
                r'\b(thinking about).{0,20}(dying|death|ending it)\b',
                
                # NEW PATTERNS for phrases like yours
                r'(world|everyone|people).{0,20}(better (off)?).{0,20}(without me|if i were gone|if i wasn\'t here)',
                r'(better (off)?).{0,20}(without me|if i (was|were) (gone|dead|not around))',
                r'(no one|nobody).{0,20}(would|will).{0,20}(care|miss|notice)',
                r'(feel|being).{0,20}(a|like a).{0,20}(burden|weight).{0,20}(on|to).{0,20}(others|everyone|family|people)',
                r'(sadness|pain|emptiness|hurt).{0,20}(keeps|just keeps|continues).{0,20}(growing|getting worse|increasing)',
                r'(fake|pretend|put on).{0,20}(smile|happy|happiness)',
                r'(inside|deep down).{0,20}(i).{0,20}(feel|am).{0,20}(empty|hollow|broken|dying)',
                r'(world|life).{0,20}(would be|is).{0,20}(better|easier).{0,20}(without me|if i (wasn\'t|weren\'t) here)',
                r'(i\'?m?).{0,20}(just).{0,20}(a|such a).{0,20}(burden|problem|nuisance)',
                r'(everyone).{0,20}(would be).{0,20}(better off).{0,20}(without me)',
            ]
            
            # Check enhanced patterns
            pattern_matches = []
            for pattern in ENHANCED_PATTERNS:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    pattern_matches.append(pattern)
            
            # Check for emotional distress indicators
            distress_indicators = [
                'empty', 'hollow', 'numb', 'broken', 'dying inside',
                'sadness', 'pain', 'hurt', 'suffering', 'anguish',
                'alone', 'lonely', 'isolated', 'abandoned',
                'worthless', 'useless', 'hopeless', 'helpless',
                'burden', 'problem', 'nuisance', 'trouble',
                'fake smile', 'pretend', 'hiding', 'mask'
            ]
            
            distress_matches = []
            for indicator in distress_indicators:
                if indicator in text_lower:
                    distress_matches.append(indicator)
            
            # Check for specific concerning combinations
            concerning_combinations = 0
            if 'better' in text_lower and 'without me' in text_lower:
                concerning_combinations += 2
            if 'burden' in text_lower and any(word in text_lower for word in ['everyone', 'others', 'family']):
                concerning_combinations += 1
            if ('empty' in text_lower or 'hollow' in text_lower) and ('inside' in text_lower or 'feel' in text_lower):
                concerning_combinations += 1
            if ('smile' in text_lower and ('fake' in text_lower or 'pretend' in text_lower)):
                concerning_combinations += 1
            
            # Calculate risk score with improved weighting
            risk_score = (
                len(high_risk_matches) * 3 +
                len(medium_risk_matches) * 2 +
                len(low_risk_matches) * 1 +
                len(pattern_matches) * 3 +  # Increased weight for patterns
                len(distress_matches) * 1 +
                concerning_combinations * 2
            )
            
            # Check if the text contains serious concerning phrases (like your message)
            serious_concern = False
            if ('world' in text_lower and 'better' in text_lower and 'without me' in text_lower):
                serious_concern = True
            if ('burden' in text_lower and 'everyone' in text_lower):
                serious_concern = True
            if (('empty' in text_lower or 'hollow' in text_lower) and 
                ('inside' in text_lower or 'feel' in text_lower) and
                ('sadness' in text_lower or 'pain' in text_lower)):
                serious_concern = True

        
            # Determine risk level with improved logic
            if risk_score >= 8 or len(high_risk_matches) > 0 or serious_concern:
                risk_level = "HIGH RISK"
                confidence = min(risk_score / 15, 1.0) * 100
                needs_immediate_attention = True
                recommendations = [
                    "IMMEDIATE ACTION REQUIRED - Contact emergency services (112)",
                    "Call National Suicide Prevention Lifeline: 1-800-273-8255",
                    "Text HOME to 741741 for Crisis Text Line",
                    "Do not leave the person alone if possible",
                    "Remove any means of self-harm if safe to do so",
                    "You are NOT a burden - these feelings are treatable"
                ]
            elif risk_score >= 4 or len(medium_risk_matches) >= 2 or len(pattern_matches) >= 2:
                risk_level = "MODERATE RISK"
                confidence = (risk_score / 15) * 100
                needs_immediate_attention = False
                recommendations = [
                    "Please reach out to a mental health professional today",
                    "Contact a crisis helpline for support: 1-800-273-8255",
                    "You deserve support - you are not alone",
                    "These feelings are valid and help is available",
                    "Consider telling someone you trust how you're feeling"
                ]
            elif risk_score >= 1:
                risk_level = "LOW RISK"
                confidence = (risk_score / 15) * 100
                needs_immediate_attention = False
                recommendations = [
                    "Monitor for any worsening of symptoms",
                    "Maintain connection with support systems",
                    "Practice stress management techniques",
                    "Consider talking to a counselor if concerns persist",
                    "Reach out to mental health resources if needed"
                ]
            else:
                risk_level = "NO IMMEDIATE RISK"
                confidence = 0
                needs_immediate_attention = False
                recommendations = [
                    "No immediate suicidal indicators detected",
                    "Continue monitoring mental health",
                    "Maintain healthy coping strategies",
                    "Reach out if feelings change"
                ]
            
            # Prepare warning signs with more specific descriptions
            warning_signs = []
            if high_risk_matches or serious_concern:
                warning_signs.append("Statements suggesting life would be better without them")
            if medium_risk_matches:
                warning_signs.append("Expressions of hopelessness, worthlessness, or feeling like a burden")
            if low_risk_matches:
                warning_signs.append("Mentions of depression, sadness, or emotional distress")
            if pattern_matches:
                warning_signs.append("Language patterns consistent with suicidal thinking")
            if distress_matches:
                warning_signs.append(f"Emotional distress indicators: {', '.join(distress_matches[:3])}")
            
            # Add specific note for the "world better without me" pattern
            special_note = None
            if 'world' in text_lower and 'better' in text_lower and 'without me' in text_lower:
                special_note = "⚠️ CRITICAL: Statement about 'world being better without me' is a serious indicator of suicidal thoughts"
            
            # Generate highlighted text
            highlighted_text = self.highlight_keywords(text)
            
            return {
                'risk_level': risk_level,
                'confidence': round(confidence, 2),
                'risk_score': risk_score,
                'high_risk_keywords': high_risk_matches[:5],
                'medium_risk_keywords': medium_risk_matches[:5],
                'low_risk_keywords': low_risk_matches[:5],
                'patterns_detected': len(pattern_matches),
                'distress_indicators': distress_matches[:5],
                'warning_signs': warning_signs,
                'special_note': special_note,
                'needs_immediate_attention': needs_immediate_attention,
                'recommendations': recommendations,
                'highlighted_text': highlighted_text,
                'processed_text': self.preprocess_text(text)[:200] + '...' if len(text) > 200 else self.preprocess_text(text)
            }
            
        except Exception as e:
            print(f"Error in analysis: {e}")
            return {
                'risk_level': "ANALYSIS ERROR",
                'confidence': 0,
                'risk_score': 0,
                'error': str(e),
                'needs_immediate_attention': False,
                'recommendations': ["Unable to analyze text. Please try again or contact support."],
                'highlighted_text': Markup(html.escape(text))
            }

# Initialize suicide detection model
suicide_detector = SuicideDetectionModel()

@app.route('/')
def root():
    return render_template('index.html')

@app.route('/predict', methods=["POST"])
def predict():
    try:
        # getting values from the form
        q1 = int(request.form['a1'])
        q2 = int(request.form['a2'])
        q3 = int(request.form['a3'])
        q4 = int(request.form['a4'])
        q5 = int(request.form['a5'])
        q6 = int(request.form['a6'])
        q7 = int(request.form['a7'])
        q8 = int(request.form['a8'])
        q9 = int(request.form['a9'])
        q10 = int(request.form['a10'])

        values = [q1, q2, q3, q4, q5, q6, q7, q8, q9, q10]
        model = Model()
        classifier = model.svm_classifier()
        prediction = classifier.predict([values])
        
        total_score = sum(values)
        
        if prediction[0] == 0:
            result = 'Test result : No Depression'
            return render_template("result1.html", result=result, score=[q1, q2, q3, q4, q5, q6, q7, q8, q9, q10], total_score=total_score)
        elif prediction[0] == 1:
            result = 'Test result : Mild Depression'
            return render_template("result2.html", result=result, score=[q1, q2, q3, q4, q5, q6, q7, q8, q9, q10], total_score=total_score)
        elif prediction[0] == 2:
            result = 'Test result : Moderate Depression'
            return render_template("result2.html", result=result, score=[q1, q2, q3, q4, q5, q6, q7, q8, q9, q10], total_score=total_score)
        elif prediction[0] == 3:
            result = 'Test result : Moderately severe Depression'
            return render_template("result2.html", result=result, score=[q1, q2, q3, q4, q5, q6, q7, q8, q9, q10], total_score=total_score)
        elif prediction[0] == 4:
            result = 'Test result : Severe Depression'
            return render_template("result2.html", result=result, score=[q1, q2, q3, q4, q5, q6, q7, q8, q9, q10], total_score=total_score)
        else:
            result = 'Test result : Unable to determine'
            return render_template("result2.html", result=result, score=[q1, q2, q3, q4, q5, q6, q7, q8, q9, q10], total_score=total_score)
            
    except Exception as e:
        flash(f'Error in prediction: {str(e)}')
        return render_template('index.html')

@app.route('/detect_suicidal', methods=["POST"])
def detect_suicidal():
    """Endpoint to detect suicidal ideation from social media text"""
    try:
        # Get social media text from form
        social_media_text = request.form['social_text']
        
        # Check if text is empty
        if not social_media_text or len(social_media_text.strip()) == 0:
            flash('Please enter some text to analyze.')
            return render_template('suicidal_detection.html')
        
        # Analyze the text
        analysis_result = suicide_detector.analyze_suicidal_content(social_media_text)
        
        # Prepare resources for the user
        resources = {
            'immediate_help': [
                {'name': 'National Suicide Prevention Lifeline', 'number': '1-800-273-8255', 'description': '24/7, free and confidential support'},
                {'name': 'Crisis Text Line', 'number': 'Text HOME to 741741', 'description': '24/7 support via text message'},
                {'name': 'Emergency Services', 'number': '112', 'description': 'For immediate emergencies'}
            ],
            'online_resources': [
                {'name': 'Suicide Prevention Resource Center', 'url': 'https://www.sprc.org', 'description': 'Resources for prevention'},
                {'name': 'American Foundation for Suicide Prevention', 'url': 'https://afsp.org', 'description': 'Research and education'},
                {'name': 'National Institute of Mental Health', 'url': 'https://www.nimh.nih.gov', 'description': 'Mental health information'}
            ],
            'helplines': [
                {'name': 'SAMHSA National Helpline', 'number': '1-800-662-4357', 'description': 'Treatment referral and information'},
                {'name': 'Veterans Crisis Line', 'number': '1-800-273-8255 press 1', 'description': 'Support for veterans'},
                {'name': 'The Trevor Project', 'number': '1-866-488-7386', 'description': 'LGBTQ youth support'}
            ]
        }
        
        # Render appropriate template based on risk level
        if analysis_result['needs_immediate_attention']:
            return render_template("suicidal_high_risk.html", 
                                 result=analysis_result, 
                                 resources=resources,
                                 original_text=social_media_text)
        elif analysis_result['risk_level'] == "MODERATE RISK":
            return render_template("suicidal_moderate_risk.html", 
                                 result=analysis_result, 
                                 resources=resources,
                                 original_text=social_media_text)
        else:
            return render_template("suicidal_low_risk.html", 
                                 result=analysis_result, 
                                 resources=resources,
                                 original_text=social_media_text)
    
    except Exception as e:
        flash(f'Error analyzing text: {str(e)}')
        return render_template('suicidal_detection.html')

@app.route('/suicidal_detection')
def suicidal_detection_page():
    """Render the suicidal ideation detection page"""
    return render_template('suicidal_detection.html')

@app.route('/aboutus')
def about_us():
    return render_template('aboutus.html')

@app.route('/contactus')
def Contact_us():
    return render_template('contact2.html')

# API endpoint for AJAX requests
@app.route('/api/detect_suicidal', methods=['POST'])
def api_detect_suicidal():
    """API endpoint for suicidal ideation detection"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        analysis_result = suicide_detector.analyze_suicidal_content(text)
        
        # Convert highlighted_text to string for JSON
        if 'highlighted_text' in analysis_result:
            analysis_result['highlighted_text'] = str(analysis_result['highlighted_text'])
        
        return jsonify(analysis_result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Add crisis resources page
@app.route('/crisis-resources')
def crisis_resources():
    """Display crisis resources and helplines"""
    resources = {
        'immediate_help': [
            {'name': 'National Suicide Prevention Lifeline', 'number': '1-800-273-8255', 'description': '24/7, free and confidential support for people in distress'},
            {'name': 'Crisis Text Line', 'number': 'Text HOME to 741741', 'description': '24/7 support via text message from anywhere in the US'},
            {'name': 'Emergency Services', 'number': '112', 'description': 'For immediate emergencies, call 112'}
        ],
        'helplines': [
            {'name': 'SAMHSA National Helpline', 'number': '1-800-662-4357', 'description': 'Treatment referral and information service for individuals facing mental health or substance use disorders'},
            {'name': 'Veterans Crisis Line', 'number': '1-800-273-8255 press 1', 'description': 'Confidential support for veterans and their families'},
            {'name': 'The Trevor Project', 'number': '1-866-488-7386', 'description': 'Crisis intervention and suicide prevention services for LGBTQ youth'},
            {'name': 'Trans Lifeline', 'number': '1-877-565-8860', 'description': 'Peer support for trans and questioning individuals'},
            {'name': 'National Child Abuse Hotline', 'number': '1-800-422-4453', 'description': '24/7 support for children and families in crisis'}
        ],
        'online_resources': [
            {'name': 'Suicide Prevention Resource Center', 'url': 'https://www.sprc.org', 'description': 'Resources for suicide prevention, best practices, and training'},
            {'name': 'American Foundation for Suicide Prevention', 'url': 'https://afsp.org', 'description': 'Research, education, and advocacy for suicide prevention'},
            {'name': 'National Institute of Mental Health', 'url': 'https://www.nimh.nih.gov', 'description': 'Mental health information and research'},
            {'name': 'Mental Health America', 'url': 'https://www.mhanational.org', 'description': 'Mental health screening tools and resources'},
            {'name': 'NAMI', 'url': 'https://www.nami.org', 'description': 'National Alliance on Mental Illness support and education'}
        ]
    }
    return render_template('crisis_resources.html', resources=resources)

app.secret_key = os.urandom(12)

if __name__ == '__main__':
    app.run(port=5987, host='0.0.0.0', debug=True)