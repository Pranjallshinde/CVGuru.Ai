from flask import Flask, render_template, request, jsonify, session
import os
import time
from functools import wraps
import hashlib
import google.generativeai as genai
import pdfplumber
import google.api_core.exceptions # Import specific exceptions

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'railway-secret-key-123')

# Railway port configuration
PORT = int(os.environ.get('PORT', 8080))

# Server-side rate limiting
RATE_LIMIT_STORAGE = {}
RATE_LIMIT_WINDOW = 600  # 10 minutes (increased from 5 minutes)
RATE_LIMIT_MAX_REQUESTS = 10  # Max 10 requests per 10 minutes per IP (increased from 3)

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
            # Calculate time until next request is allowed
            oldest_request_time = RATE_LIMIT_STORAGE[client_ip][0]
            time_to_wait = int(RATE_LIMIT_WINDOW - (current_time - oldest_request_time))
            return render_template('predict.html', 
                                 error=f"Too many requests. Please wait {time_to_wait} seconds before making another request.")
                
        # Record this request
        if client_ip not in RATE_LIMIT_STORAGE:
            RATE_LIMIT_STORAGE[client_ip] = []
        RATE_LIMIT_STORAGE[client_ip].append(current_time)
                
        return f(*args, **kwargs)
    return decorated_function

# NEW: Retry decorator for Gemini API calls
def retry_gemini_api(max_retries=3, initial_delay=1):
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            retries = 0
            delay = initial_delay
            while retries < max_retries:
                try:
                    return f(*args, **kwargs)
                except (google.api_core.exceptions.ResourceExhausted, # Quota exceeded
                        google.api_core.exceptions.InternalServerError) as e: # Transient server error
                    retries += 1
                    print(f"Gemini API call failed (retry {retries}/{max_retries}): {e}")
                    if retries < max_retries:
                        time.sleep(delay)
                        delay *= 2 # Exponential backoff
                    else:
                        raise # Re-raise if max retries reached
                except Exception as e:
                    # For other unexpected errors, re-raise immediately
                    raise
        return wrapper
    return decorator

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
        api_key = os.environ.get('GEMINI_API_KEY')
        if not api_key:
            return render_template('predict.html', error="API key not configured. Please set GEMINI_API_KEY environment variable.")
                
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")
                
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
            return render_template('predict.html', error="Could not extract text from PDF. Please ensure it's a readable PDF.")
                
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
                
        # Apply retry logic to the API call
        @retry_gemini_api()
        def generate_content_with_retry(model, prompt):
            return model.generate_content(prompt)

        response = generate_content_with_retry(model, combined_prompt)
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
                error="Could not generate sufficient questions. Please try with a different resume or job title.")
                
        # Store in session with reduced context
        session['questions'] = questions
        session['job_title'] = job_title
        session['resume_text'] = text_content[:1500]  # Reduced storage
                
        return render_template('questions_result.html',
                              questions=questions,
                              job_title=job_title)
            
    except google.api_core.exceptions.ResourceExhausted as e:
        return render_template('predict.html', error="API quota exceeded. Please check your Google Cloud Console for usage limits or try again later.")
    except google.api_core.exceptions.GoogleAPIError as e:
        return render_template('predict.html', error=f"Google API error: {str(e)}. Please try again.")
    except Exception as e:
        return render_template('predict.html', error=f"An unexpected error occurred: {str(e)}. Please try again.")

@app.route('/generate_answers', methods=['POST'])
@rate_limit_decorator
def generate_answers():
    try:
        questions = session.get('questions', [])
        job_title = session.get('job_title', '')
        resume_text = session.get('resume_text', '')
                
        if not questions:
            return jsonify({'error': 'No questions found. Please generate questions first.'})
                
        api_key = os.environ.get('GEMINI_API_KEY')
        if not api_key:
            return jsonify({'error': 'API key not configured on server.'})

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.5-flash")
                
        # Optimized prompt for better token efficiency
        prompt = f"""
        Generate concise STAR method answers for these interview questions:
                
        Job: {job_title}
        Resume Context: {resume_text[:1000]}  # Further reduced
                
        Questions: {chr(10).join([f"{i+1}. {q}" for i, q in enumerate(questions[:10])])}  # Limit to 5 questions
                
        Format each answer as:
        ANSWER_X:
        *Situation:* [1-2 sentences]
        *Task:* [1 sentence]
        *Action:* [1-2 sentences]
        *Result:* [1 sentence]
                
        Keep answers concise and professional.
        """
                
        # Apply retry logic to the API call
        @retry_gemini_api()
        def generate_content_with_retry(model, prompt):
            return model.generate_content(prompt)

        response = generate_content_with_retry(model, prompt)
                
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
            
    except google.api_core.exceptions.ResourceExhausted as e:
        return jsonify({'error': "API quota exceeded for answer generation. Please check your Google Cloud Console for usage limits or try again later."})
    except google.api_core.exceptions.GoogleAPIError as e:
        return jsonify({'error': f"Google API error during answer generation: {str(e)}. Please try again."})
    except Exception as e:
        return jsonify({'error': f'An unexpected error occurred during answer generation: {str(e)}'})

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
    print("- Server-side rate limiting (10 requests per 10 minutes)")
    print("- Server-side retry logic for Gemini API calls")
    print("- Single API call per question generation request")
    print("- Reduced token usage")
    print("- Enhanced error handling")
        
    app.run(
        debug=False,
        host='0.0.0.0',
        port=PORT
    )
