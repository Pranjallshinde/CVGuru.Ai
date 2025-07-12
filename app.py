from flask import Flask, render_template, request, jsonify, session
import os
import time
from functools import wraps
import hashlib

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'railway-secret-key-123')

# Railway port configuration
PORT = int(os.environ.get('PORT', 8080))

# Server-side rate limiting
RATE_LIMIT_STORAGE = {}
RATE_LIMIT_WINDOW = 300  # 5 minutes
RATE_LIMIT_MAX_REQUESTS = 3  # Max 3 requests per 5 minutes per IP

def rate_limit_decorator(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        client_ip = request.environ.get('HTTP_X_FORWARDED_FOR', request.environ.get('REMOTE_ADDR', 'unknown'))
        current_time = time.time()
        
        # Clean old entries
        RATE_LIMIT_STORAGE[client_ip] = [
            timestamp for timestamp in RATE_LIMIT_STORAGE.get(client_ip, [])
            if current_time - timestamp < RATE_LIMIT_WINDOW
        ]
        
        # Check rate limit
        if len(RATE_LIMIT_STORAGE.get(client_ip, [])) >= RATE_LIMIT_MAX_REQUESTS:
            return render_template('predict.html', 
                error="Rate limit exceeded. Please wait 5 minutes before making another request.")
        
        # Record this request
        if client_ip not in RATE_LIMIT_STORAGE:
            RATE_LIMIT_STORAGE[client_ip] = []
        RATE_LIMIT_STORAGE[client_ip].append(current_time)
        
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
def index():
    return render_template('front.html')

@app.route('/test')
def test():
    return f"Flask is working on Railway! Port: {PORT} 🚀"

@app.route('/api/health')
def health():
    return jsonify({
        'status': 'healthy', 
        'platform': 'railway',
        'port': PORT,
        'env_vars': {
            'PORT': os.environ.get('PORT', 'Not set'),
            'GEMINI_API_KEY': 'Set' if os.environ.get('GEMINI_API_KEY') else 'Not set'
        }
    })

@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/generate_questions', methods=['POST'])
@rate_limit_decorator
def generate_questions():
    try:
        import google.generativeai as genai
        import pdfplumber
        
        api_key = os.environ.get('GEMINI_API_KEY')
        if not api_key:
            return render_template('predict.html', error="API key not configured")
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        if 'pdf_file' not in request.files:
            return render_template('predict.html', error="No file uploaded")
        
        file = request.files['pdf_file']
        job_title = request.form.get('job_title', '')
        
        if not file.filename or not job_title:
            return render_template('predict.html', error="Please select file and job title")
        
        # Extract text from PDF with better error handling
        text_content = ""
        try:
            with pdfplumber.open(file) as pdf:
                # Extract text from first 2 pages only to reduce token usage
                for page in pdf.pages[:2]:
                    page_text = page.extract_text()
                    if page_text:
                        text_content += page_text + "\n"
        except Exception as pdf_error:
            return render_template('predict.html', error=f"Error reading PDF file: {str(pdf_error)}")
        
        if not text_content.strip():
            return render_template('predict.html', error="Could not extract text from PDF")
        
        # Limit text content to reduce API usage
        text_content = text_content[:2500]  # Reduced from 4000
        
        # SINGLE API CALL - Combined validation and question generation
        combined_prompt = f"""
        Analyze the following resume content and perform two tasks:

        1. First, determine if this is a valid resume/CV
        2. If valid, generate exactly 10 relevant interview questions for {job_title}

        Resume Content: {text_content}

        RESPONSE FORMAT:
        VALIDATION: [VALID_RESUME or NOT_RESUME with brief explanation]
        
        QUESTIONS:
        1. [Question 1]
        2. [Question 2]
        3. [Question 3]
        4. [Question 4]
        5. [Question 5]
        6. [Question 6]
        7. [Question 7]
        8. [Question 8]
        9. [Question 9]
        10. [Question 10]

        Requirements for questions:
        - Specific to the candidate's experience in the resume
        - Mix of technical and behavioral questions for {job_title}
        - Challenging but fair
        - Based on actual projects/technologies mentioned
        """
        
        # Single API call instead of two
        response = model.generate_content(combined_prompt)
        response_text = response.text
        
        # Parse validation result
        if "NOT_RESUME" in response_text:
            return render_template('predict.html',
                error="The uploaded document doesn't appear to be a resume. Please upload a valid resume.")
        
        # Extract questions
        questions = []
        lines = response_text.split('\n')
        in_questions_section = False
        
        for line in lines:
            line = line.strip()
            if line.startswith('QUESTIONS:'):
                in_questions_section = True
                continue
            
            if in_questions_section and line:
                # Extract question text after number
                if line and (line[0].isdigit() or line.startswith('Q')):
                    if '.' in line:
                        question = line.split('.', 1)[-1].strip()
                        if len(question) > 15:
                            questions.append(question)
        
        # Ensure we have exactly 10 questions
        questions = questions[:10]
        
        if len(questions) < 5:
            return render_template('predict.html', 
                error="Could not generate sufficient questions. Please try with a different resume.")
        
        # Store in session with reduced context
        session['questions'] = questions
        session['job_title'] = job_title
        session['resume_text'] = text_content[:1500]  # Reduced storage
        
        return render_template('questions_result.html',
                              questions=questions,
                              job_title=job_title)
        
    except Exception as e:
        error_message = str(e)
        if "quota" in error_message.lower():
            error_message = "API quota exceeded. Please try again in a few hours."
        elif "api" in error_message.lower():
            error_message = "API service temporarily unavailable. Please try again later."
        
        return render_template('predict.html', error=f"Error: {error_message}")

@app.route('/generate_answers', methods=['POST'])
@rate_limit_decorator
def generate_answers():
    try:
        import google.generativeai as genai
        
        questions = session.get('questions', [])
        job_title = session.get('job_title', '')
        resume_text = session.get('resume_text', '')
        
        if not questions:
            return jsonify({'error': 'No questions found. Please generate questions first.'})
        
        api_key = os.environ.get('GEMINI_API_KEY')
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        # Optimized prompt for better token efficiency
        prompt = f"""
        Generate concise STAR method answers for these interview questions:
        
        Job: {job_title}
        Resume Context: {resume_text[:1000]}  # Further reduced
        
        Questions: {chr(10).join([f"{i+1}. {q}" for i, q in enumerate(questions[:5])])}  # Limit to 5 questions
        
        Format each answer as:
        ANSWER_X:
        *Situation:* [1-2 sentences]
        *Task:* [1 sentence]
        *Action:* [1-2 sentences]
        *Result:* [1 sentence]
        
        Keep answers concise and professional.
        """
        
        response = model.generate_content(prompt)
        
        # Parse answers (simplified)
        import re
        answers = {}
        matches = re.findall(r'ANSWER_(\d+):\s*(.*?)(?=ANSWER_\d+:|$)', response.text, re.DOTALL)
        
        for match in matches:
            answer_num = int(match[0])
            answer_text = match[1].strip()
            
            if len(answer_text) > 20:
                # Format for HTML display
                formatted_answer = answer_text.replace('*Situation:*', '<strong>Situation:</strong>')
                formatted_answer = formatted_answer.replace('*Task:*', '<strong>Task:</strong>')
                formatted_answer = formatted_answer.replace('*Action:*', '<strong>Action:</strong>')
                formatted_answer = formatted_answer.replace('*Result:*', '<strong>Result:</strong>')
                formatted_answer = formatted_answer.replace('\n', '<br>')
                answers[answer_num] = formatted_answer
        
        return jsonify({
            'success': True,
            'structured_answers': answers,
            'total_questions': len(questions),
            'method_used': 'STAR Method'
        })
        
    except Exception as e:
        error_message = str(e)
        if "quota" in error_message.lower():
            error_message = "API quota exceeded. Please try again in a few hours."
        
        return jsonify({'error': f'Answer generation failed: {error_message}'})

@app.route('/how_to_use')
def how_to_use():
    return render_template('how_to_use.html')

@app.errorhandler(404)
def not_found_error(error):
    return render_template('predict.html', error="Page not found"), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('predict.html', error="Internal server error occurred"), 500

# File size limit
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB limit

@app.errorhandler(413)
def too_large(e):
    return render_template('predict.html', error="File too large. Please upload a file smaller than 5MB."), 413

if __name__ == '__main__':
    print(f"Starting Flask app on port {PORT}")
    print("Optimizations enabled:")
    print("- Server-side rate limiting")
    print("- Single API call per request")
    print("- Reduced token usage")
    print("- Enhanced error handling")
    
    app.run(
        debug=False,
        host='0.0.0.0',
        port=PORT
    )
