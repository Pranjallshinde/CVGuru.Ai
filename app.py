from flask import Flask, render_template, request, jsonify, session
import os
import time
from functools import wraps
import hashlib
import google.generativeai as genai
import pdfplumber
import google.api_core.exceptions # Import specific exceptions

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'render-secret-key-123')

# Railway port configuration
PORT = int(os.environ.get('PORT', 10000))

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
        'platform': 'render',
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
        - Easy to Medium level but fair
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
        
        # Process each question individually to guarantee all answers
        answers = {}
        
        for i, question in enumerate(questions[:10], 1):
            try:
                # Individual prompt for each question
                prompt = f"""
Generate a STAR method answer for this interview question:

Job Title: {job_title}
Resume Context: {resume_text[:500]}

Question {i}: {question}

Provide your answer in this EXACT format:
SITUATION: [Brief description of the situation - 1-2 sentences]
TASK: [What needed to be accomplished - 1 sentence]  
ACTION: [Specific actions you took - 1-2 sentences]
RESULT: [The outcome achieved - 1 sentence]

Make it professional and relevant to the job title. Do not include any other text or formatting.
"""
                
                @retry_gemini_api()
                def generate_single_answer(model, prompt):
                    return model.generate_content(prompt)
                
                response = generate_single_answer(model, prompt)
                
                # Parse the individual response
                answer_text = response.text.strip()
                formatted_answer = parse_single_answer(answer_text)
                
                answers[i] = formatted_answer
                
                # Small delay to avoid rate limiting
                import time
                time.sleep(0.3)
                
            except Exception as e:
                print(f"Error generating answer for question {i}: {str(e)}")
                # Create fallback answer to maintain order
                answers[i] = create_fallback_answer(question, job_title, i)
        
        # Final check - ensure we have exactly 10 answers in order
        final_answers = {}
        for i in range(1, 11):
            if i in answers:
                final_answers[i] = answers[i]
            else:
                # Create missing answer
                question_text = questions[i-1] if i <= len(questions) else "General interview question"
                final_answers[i] = create_fallback_answer(question_text, job_title, i)
        
        return jsonify({
            'success': True,
            'structured_answers': final_answers,
            'total_questions': 10,
            'method_used': 'STAR Method'
        })
        
    except google.api_core.exceptions.ResourceExhausted as e:
        return jsonify({'error': "API quota exceeded for answer generation. Please check your Google Cloud Console for usage limits or try again later."})
    except google.api_core.exceptions.GoogleAPIError as e:
        return jsonify({'error': f"Google API error during answer generation: {str(e)}. Please try again."})
    except Exception as e:
        return jsonify({'error': f'An unexpected error occurred during answer generation: {str(e)}'})

def parse_single_answer(answer_text):
    """Parse a single answer response"""
    import re
    
    # Initialize components
    situation = ""
    task = ""
    action = ""
    result = ""
    
    # Split by lines and process
    lines = answer_text.split('\n')
    current_section = None
    current_content = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check for section headers
        if re.match(r'^SITUATION:', line, re.IGNORECASE):
            if current_section and current_content:
                save_section_content(current_section, current_content, locals())
            current_section = 'situation'
            current_content = [line.split(':', 1)[1].strip() if ':' in line else '']
            
        elif re.match(r'^TASK:', line, re.IGNORECASE):
            if current_section and current_content:
                save_section_content(current_section, current_content, locals())
            current_section = 'task'
            current_content = [line.split(':', 1)[1].strip() if ':' in line else '']
            
        elif re.match(r'^ACTION:', line, re.IGNORECASE):
            if current_section and current_content:
                save_section_content(current_section, current_content, locals())
            current_section = 'action'
            current_content = [line.split(':', 1)[1].strip() if ':' in line else '']
            
        elif re.match(r'^RESULT:', line, re.IGNORECASE):
            if current_section and current_content:
                save_section_content(current_section, current_content, locals())
            current_section = 'result'
            current_content = [line.split(':', 1)[1].strip() if ':' in line else '']
            
        else:
            # Continue current section
            if current_section:
                current_content.append(line)
    
    # Save the last section
    if current_section and current_content:
        save_section_content(current_section, current_content, locals())
    
    # If parsing failed, try simple extraction
    if not situation and not task and not action and not result:
        situation, task, action, result = extract_star_simple(answer_text)
    
    # Format for HTML display
    formatted_answer = f"""
<strong>Situation:</strong> {situation or 'Relevant professional situation from my experience.'}<br>
<strong>Task:</strong> {task or 'Clear objective that needed to be accomplished.'}<br>
<strong>Action:</strong> {action or 'Systematic approach and specific steps taken.'}<br>
<strong>Result:</strong> {result or 'Successful outcome with measurable impact.'}
"""
    
    return formatted_answer.strip()

def save_section_content(section, content_list, local_vars):
    """Helper function to save section content"""
    content = ' '.join(content_list).strip()
    if content:
        local_vars[section] = content

def extract_star_simple(text):
    """Simple extraction as fallback"""
    import re
    
    situation = ""
    task = ""
    action = ""
    result = ""
    
    # Try to find each component with regex
    sit_match = re.search(r'situation[:\-\s]+(.*?)(?=task|action|result|$)', text, re.IGNORECASE | re.DOTALL)
    if sit_match:
        situation = sit_match.group(1).strip()
    
    task_match = re.search(r'task[:\-\s]+(.*?)(?=action|result|situation|$)', text, re.IGNORECASE | re.DOTALL)
    if task_match:
        task = task_match.group(1).strip()
    
    action_match = re.search(r'action[:\-\s]+(.*?)(?=result|situation|task|$)', text, re.IGNORECASE | re.DOTALL)
    if action_match:
        action = action_match.group(1).strip()
    
    result_match = re.search(r'result[:\-\s]+(.*?)(?=situation|task|action|$)', text, re.IGNORECASE | re.DOTALL)
    if result_match:
        result = result_match.group(1).strip()
    
    return situation, task, action, result

def create_fallback_answer(question, job_title, question_num):
    """Create a structured fallback answer"""
    
    # Create contextual fallback based on question content
    question_lower = question.lower()
    
    if 'challenge' in question_lower or 'difficult' in question_lower:
        situation = f"In my role as a {job_title}, I encountered a challenging situation that required immediate attention."
        task = "I needed to resolve the issue while maintaining quality standards and meeting deadlines."
        action = "I analyzed the problem systematically, consulted with relevant stakeholders, and implemented a step-by-step solution."
        result = "The challenge was successfully resolved, and I gained valuable experience for handling similar situations."
        
    elif 'team' in question_lower or 'collaborate' in question_lower:
        situation = f"While working as a {job_title}, I was part of a diverse team working on an important project."
        task = "I needed to ensure effective collaboration and contribute to achieving our team objectives."
        action = "I actively participated in team discussions, shared my expertise, and supported colleagues when needed."
        result = "Our team successfully completed the project on time and received positive feedback from stakeholders."
        
    elif 'learn' in question_lower or 'new' in question_lower:
        situation = f"In my position as a {job_title}, I encountered a situation requiring skills I hadn't used before."
        task = "I needed to quickly acquire new knowledge and apply it effectively to meet project requirements."
        action = "I dedicated time to research, sought guidance from experts, and practiced the new skills systematically."
        result = "I successfully mastered the new skills and applied them to deliver quality results."
        
    else:
        # Generic fallback
        situation = f"During my experience as a {job_title}, I faced a situation that required professional judgment and action."
        task = "I needed to address the situation effectively while maintaining high standards."
        action = "I approached the challenge methodically, gathered necessary information, and implemented appropriate solutions."
        result = "The situation was resolved successfully, contributing to positive outcomes for the organization."
    
    return f"""
<strong>Situation:</strong> {situation}<br>
<strong>Task:</strong> {task}<br>
<strong>Action:</strong> {action}<br>
<strong>Result:</strong> {result}
"""
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

if _name_ == '_main_':
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
        
   



