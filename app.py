from flask import Flask, request, render_template, jsonify, session
import os
import time
from collections import defaultdict

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-here')

# Import only essential libraries
try:
    import google.generativeai as genai
    import pdfplumber
except ImportError as e:
    print(f"Import error: {e}")

# Configure Gemini API
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', 'AIzaSyBthyBU74hKTO_Ux8pUOY8oq3O4fUesRXI')
genai.configure(api_key=GEMINI_API_KEY)

# Model configuration
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)

# Simple rate limiting
request_times = defaultdict(list)

def check_rate_limit(user_ip, max_requests=5, time_window=3600):
    """Simple rate limiting"""
    current_time = time.time()
    user_requests = request_times[user_ip]
    
    # Clean old requests
    user_requests[:] = [req_time for req_time in user_requests 
                       if current_time - req_time < time_window]
    
    if len(user_requests) >= max_requests:
        return False
    
    user_requests.append(current_time)
    return True

def safe_gemini_send(chat_session, query, max_retries=2):
    """Safe function to send requests to Gemini"""
    for attempt in range(max_retries):
        try:
            response = chat_session.send_message(query)
            return response
        except Exception as e:
            print(f"API Error (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(30)
                continue
            return None
    return None

@app.route('/')
def index():
    return render_template('front.html')

@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/interview_prep')
def interview_prep():
    return render_template('predict.html')

@app.route('/generate_questions', methods=['POST'])
def generate_questions():
    user_ip = request.remote_addr
    if not check_rate_limit(user_ip):
        return render_template('predict.html', 
                             error="Too many requests. Please wait 1 hour before trying again.")
    
    if 'pdf_file' not in request.files:
        return render_template('predict.html', error="No file uploaded.")
    
    file = request.files['pdf_file']
    job_title = request.form.get('job_title', '')
    
    if file.filename == '':
        return render_template('predict.html', error="No file selected.")
    
    if not job_title.strip():
        return render_template('predict.html', error="Please select a job title.")
    
    # Extract text from PDF
    text_content = ""
    try:
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_content += page_text + "\n"
    except Exception as e:
        return render_template('predict.html', error=f"Error reading PDF: {str(e)}")
    
    if not text_content.strip():
        return render_template('predict.html', error="Could not extract text from PDF.")
    
    # Limit text length
    if len(text_content) > 8000:
        text_content = text_content[:8000] + "..."
    
    # Create prompt
    prompt = f"""
    Analyze this resume and generate exactly 15 relevant interview questions for a {job_title} position.
    Format each question with a number (1., 2., etc.) on separate lines.
    Focus on the candidate's experience and skills mentioned in the resume.
    
    Resume content:
    {text_content}
    """
    
    # Generate questions
    try:
        chat_session = model.start_chat(history=[])
        response = safe_gemini_send(chat_session, prompt)
        
        if response is None:
            return render_template('predict.html', 
                                 error="API error. Please try again later.")
        
        # Process questions
        questions = []
        for line in response.text.split('\n'):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('Q')):
                # Clean up the question
                question = line.split('.', 1)[-1].strip()
                if len(question) > 10:
                    questions.append(question)
        
        if len(questions) < 5:
            return render_template('predict.html', 
                                 error="Could not generate enough questions. Please try with a different resume.")
        
        # Store in session
        session['questions'] = questions[:15]  # Limit to 15
        session['resume_text'] = text_content
        session['job_title'] = job_title
        
        return render_template('questions_result.html', 
                             questions=questions[:15], 
                             job_title=job_title)
        
    except Exception as e:
        return render_template('predict.html', 
                             error=f"Error generating questions: {str(e)}")

@app.route('/generate_answers', methods=['POST'])
def generate_answers():
    user_ip = request.remote_addr
    if not check_rate_limit(user_ip):
        return jsonify({'error': 'Too many requests. Please wait before trying again.'})
    
    questions = session.get('questions', [])
    resume_text = session.get('resume_text', '')
    job_title = session.get('job_title', '')
    
    if not questions:
        return jsonify({'error': 'No questions found. Please generate questions first.'})
    
    # Create answers prompt
    prompt = f"""
    Create sample answers for these interview questions based on the resume content.
    Use the STAR method where appropriate. Keep answers concise (2-3 sentences each).
    
    Job Role: {job_title}
    Resume: {resume_text[:3000]}
    
    Questions:
    {chr(10).join([f"{i+1}. {q}" for i, q in enumerate(questions)])}
    
    Format your response as:
    ANSWER_1: [answer for question 1]
    ANSWER_2: [answer for question 2]
    And so on...
    """
    
    try:
        chat_session = model.start_chat(history=[])
        response = safe_gemini_send(chat_session, prompt)
        
        if response is None:
            return jsonify({'error': 'API error. Please try again later.'})
        
        # Parse answers
        import re
        answer_matches = re.findall(r'ANSWER_(\d+):\s*(.*?)(?=ANSWER_\d+:|$)', 
                                  response.text, re.DOTALL)
        
        structured_answers = {}
        for match in answer_matches:
            answer_num = int(match[0])
            answer_content = match[1].strip()
            structured_answers[answer_num] = answer_content
        
        return jsonify({
            'success': True,
            'structured_answers': structured_answers,
            'total_questions': len(questions)
        })
        
    except Exception as e:
        return jsonify({'error': f'Error generating answers: {str(e)}'})

@app.route('/how_to_use')
def how_to_use():
    return render_template('how_to_use.html')

# Health check for Vercel
@app.route('/api/health')
def health():
    return jsonify({'status': 'healthy'})

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return render_template('predict.html', error="Page not found"), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('predict.html', error="Internal server error"), 500

# For Vercel
if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
