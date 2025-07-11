from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
import os
import pdfplumber
import google.generativeai as genai
from datetime import datetime, timedelta
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'railway-secret-key-123')

# Railway port configuration
PORT = int(os.environ.get('PORT', 8080))

# Configure Gemini AI
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-pro')
else:
    logger.warning("GEMINI_API_KEY not found in environment variables")
    model = None

# Rate limiting storage (in production, use Redis or database)
rate_limit_storage = {}

def check_rate_limit(user_ip, endpoint_type="upload", max_requests=5, window_minutes=5):
    """Simple rate limiting implementation"""
    current_time = datetime.now()
    key = f"{user_ip}_{endpoint_type}"
    
    if key not in rate_limit_storage:
        rate_limit_storage[key] = []
    
    # Clean old requests
    rate_limit_storage[key] = [
        req_time for req_time in rate_limit_storage[key]
        if current_time - req_time < timedelta(minutes=window_minutes)
    ]
    
    # Check if limit exceeded
    if len(rate_limit_storage[key]) >= max_requests:
        return False
    
    # Add current request
    rate_limit_storage[key].append(current_time)
    return True

def safe_gemini_send(chat_session, query, max_retries=3):
    """Safely send request to Gemini with retry logic"""
    if not model:
        logger.error("Gemini model not configured")
        return None
    
    for attempt in range(max_retries):
        try:
            response = chat_session.send_message(query)
            return response
        except Exception as e:
            logger.error(f"Gemini API error (attempt {attempt + 1}): {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                return None
    return None

@app.route('/')
def index():
    """Home page"""
    return render_template('predict.html')

@app.route('/login')
def login():
    """Login page"""
    try:
        return render_template('auth/login.html')
    except:
        # Fallback if auth template doesn't exist
        return render_template('predict.html')

@app.route('/predict')
def index1():
    """Predict page"""
    return render_template('predict.html')

@app.route('/test_generate', methods=['POST'])
def test_generate():
    """Generate interview questions from uploaded resume"""
    try:
        # Apply rate limit only for actual processing
        user_ip = request.remote_addr
        if not check_rate_limit(user_ip, endpoint_type="upload"):
            return render_template('predict.html',
                                  error="Too many uploads. Please wait 5 minutes before trying again.")

        if 'pdf_file' not in request.files:
            return render_template('predict.html', error="No file uploaded.")

        file = request.files['pdf_file']
        job_title = request.form.get('job_title', '')

        if file.filename == '':
            return render_template('predict.html', error="No file selected.")

        if not job_title.strip():
            return render_template('predict.html', error="Please select a job title.")

        # Extract text from the PDF file
        text_content = ""
        if file and file.filename.lower().endswith('.pdf'):
            try:
                with pdfplumber.open(file) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text_content += page_text + "\n"
            except Exception as e:
                logger.error(f"PDF processing error: {str(e)}")
                return render_template('predict.html', error=f"Error reading PDF: {str(e)}")
        else:
            return render_template('predict.html', error="Please upload a valid PDF file.")

        if not text_content.strip():
            return render_template('predict.html', error="Could not extract text from PDF.")

        # Limit text length to avoid token limits
        if len(text_content) > 10000:
            text_content = text_content[:10000] + "..."

        # Check if Gemini is available
        if not model:
            # Fallback to mock questions if Gemini is not available
            mock_questions = generate_mock_questions(job_title)
            session['questions'] = mock_questions
            session['resume_text'] = text_content
            session['job_title'] = job_title
            return render_template('questions_result.html',
                                  questions=mock_questions,
                                  job_title=job_title)

        # Enhanced prompt for better questions
        basequery = (
            "Below is text extracted from a professional resume. If this appears to be a valid resume, "
            f"generate exactly 15 relevant interview questions for the role of '{job_title}'. "
            "Format each question on a new line with a number (1., 2., etc.). "
            "Focus on the candidate's experience, skills, and projects mentioned in the resume. "
            "If this doesn't appear to be a resume, respond with 'This is not a resume.'\n\n"
        )
        query = basequery + text_content

        # Send to Gemini with error handling
        chat_session = model.start_chat(history=[])
        response = safe_gemini_send(chat_session, query)

        if response is None:
            # Fallback to mock questions
            mock_questions = generate_mock_questions(job_title)
            session['questions'] = mock_questions
            session['resume_text'] = text_content
            session['job_title'] = job_title
            return render_template('questions_result.html',
                                  questions=mock_questions,
                                  job_title=job_title,
                                  note="Using fallback questions due to API limitations.")

        # Check if it's a valid resume
        if response.text and response.text.strip().lower().startswith("this is not a resume"):
            return render_template('predict.html',
                                  error="The uploaded file doesn't look like a resume. Please upload a proper resume.")

        # Process questions
        questions = response.text.split("\n") if response.text else []
        questions = [q.strip() for q in questions if q.strip() and len(q.strip()) > 10]

        if not questions:
            # Fallback to mock questions
            questions = generate_mock_questions(job_title)

        # Store data in session for answer generation
        session['questions'] = questions
        session['resume_text'] = text_content
        session['job_title'] = job_title

        # Return questions with option to generate answers
        return render_template('questions_result.html',
                              questions=questions,
                              job_title=job_title)

    except Exception as e:
        logger.error(f"Unexpected error in test_generate: {str(e)}")
        return render_template('predict.html', error="An unexpected error occurred. Please try again.")

def generate_mock_questions(job_title):
    """Generate mock questions as fallback"""
    questions_db = {
        "Data Scientist": [
            "1. Tell me about your experience with machine learning algorithms.",
            "2. How do you handle missing data in your datasets?",
            "3. Describe a challenging data science project you've worked on.",
            "4. What's your approach to feature selection and engineering?",
            "5. How do you validate your machine learning models?",
            "6. Explain the difference between supervised and unsupervised learning.",
            "7. How do you communicate complex data insights to non-technical stakeholders?",
            "8. What tools and programming languages do you prefer for data analysis?",
            "9. Describe your experience with data visualization.",
            "10. How do you ensure data quality and integrity?",
            "11. What's your approach to handling large datasets?",
            "12. Tell me about a time when your analysis led to a business decision.",
            "13. How do you stay updated with the latest trends in data science?",
            "14. Describe your experience with cloud platforms for data science.",
            "15. What's your process for exploratory data analysis?"
        ],
        "Software Engineer": [
            "1. Describe your software development process.",
            "2. How do you approach debugging complex issues?",
            "3. Tell me about a challenging technical problem you solved.",
            "4. What's your experience with version control systems?",
            "5. How do you ensure code quality and maintainability?",
            "6. Describe your experience with different programming languages.",
            "7. How do you handle technical debt in your projects?",
            "8. What's your approach to testing and quality assurance?",
            "9. Tell me about a time you had to learn a new technology quickly.",
            "10. How do you collaborate with other developers on a team?",
            "11. Describe your experience with agile development methodologies.",
            "12. What's your approach to performance optimization?",
            "13. How do you handle conflicting requirements from stakeholders?",
            "14. Tell me about your experience with database design.",
            "15. What's your process for code reviews?"
        ]
    }
    
    # Get questions for the specific job title, or use generic ones
    return questions_db.get(job_title, [
        "1. Tell me about yourself and your background.",
        "2. Why are you interested in this role?",
        "3. What are your greatest strengths?",
        "4. Describe a challenging project you've worked on.",
        "5. How do you handle working under pressure?",
        "6. What motivates you in your work?",
        "7. Where do you see yourself in 5 years?",
        "8. How do you stay updated with industry trends?",
        "9. Describe a time you had to work in a team.",
        "10. What's your approach to problem-solving?",
        "11. How do you handle feedback and criticism?",
        "12. Tell me about a mistake you made and how you handled it.",
        "13. What questions do you have for us?",
        "14. Why should we hire you?",
        "15. What are your salary expectations?"
    ])

@app.route('/generate_answers', methods=['POST'])
def generate_answers():
    """Generate sample answers for the interview questions"""
    try:
        # Check rate limit for API calls
        user_ip = request.remote_addr
        if not check_rate_limit(user_ip, endpoint_type="api"):
            return jsonify({'error': 'Too many API requests. Please wait 5 minutes before trying again.'})

        # Get data from session
        questions = session.get('questions', [])
        resume_text = session.get('resume_text', '')
        job_title = session.get('job_title', '')

        if not questions or not resume_text:
            return jsonify({'error': 'Session expired. Please generate questions again.'})

        # Check if Gemini is available
        if not model:
            # Generate mock answers
            mock_answers = generate_mock_answers(questions, job_title)
            return jsonify({
                'success': True,
                'structured_answers': mock_answers,
                'total_questions': len(questions)
            })

        # Create prompt for generating answers
        answers_prompt = f"""
        Based on the following resume and job role, provide sample answers for these interview questions.
        Make the answers personal and specific to the candidate's experience mentioned in the resume.
        Use the STAR method where appropriate. Keep each answer concise (2-3 sentences).

        Job Role: {job_title}
        Resume Content: {resume_text[:5000]}

        Questions and required format:
        {chr(10).join([f"{i+1}. {q}" for i, q in enumerate(questions)])}

        IMPORTANT: Provide answers in this exact format:
        ANSWER_1: [Your answer for question 1]
        ANSWER_2: [Your answer for question 2]
        ANSWER_3: [Your answer for question 3]
        ... and so on for all questions.

        Make sure each answer relates to the candidate's actual experience from the resume.
        """

        # Generate answers
        chat_session = model.start_chat(history=[])
        response = safe_gemini_send(chat_session, answers_prompt)

        if response is None:
            # Fallback to mock answers
            mock_answers = generate_mock_answers(questions, job_title)
            return jsonify({
                'success': True,
                'structured_answers': mock_answers,
                'total_questions': len(questions),
                'note': 'Using fallback answers due to API limitations.'
            })

        # Process answers
        answer_text = response.text if response.text else "No answers generated."

        # Parse answers by ANSWER_X format
        import re
        answer_matches = re.findall(r'ANSWER_(\d+):\s*(.*?)(?=ANSWER_\d+:|$)', answer_text, re.DOTALL)

        # Create structured answers
        structured_answers = {}
        for match in answer_matches:
            answer_num = int(match[0])
            answer_content = match[1].strip()
            structured_answers[answer_num] = answer_content

        # If no structured answers found, generate mock ones
        if not structured_answers:
            structured_answers = generate_mock_answers(questions, job_title)

        return jsonify({
            'success': True,
            'structured_answers': structured_answers,
            'total_questions': len(questions)
        })

    except Exception as e:
        logger.error(f"Error in generate_answers: {str(e)}")
        return jsonify({'error': 'An error occurred while generating answers. Please try again.'})

def generate_mock_answers(questions, job_title):
    """Generate mock answers as fallback"""
    mock_answers = {}
    for i, question in enumerate(questions, 1):
        mock_answers[i] = f"Based on my experience in {job_title.lower()}, I would approach this by first analyzing the situation, then taking specific actions based on best practices, and measuring the results to ensure success. This aligns with my professional background and demonstrates my problem-solving approach."
    return mock_answers

@app.route('/how_to_use')
def how_to_use():
    """How to use page"""
    return render_template('how_to_use.html')

@app.route('/interview_prep')
def interview_prep():
    """Interview prep page"""
    try:
        return render_template('interview_prep.html')
    except:
        # Fallback if template doesn't exist
        return render_template('predict.html')

@app.errorhandler(404)
def not_found_error(error):
    return render_template('predict.html', error="Page not found"), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {str(error)}")
    return render_template('predict.html', error="Internal server error occurred"), 500

if __name__ == '__main__':
    print(f"🚀 Starting CVGuru Interview Prep on port {PORT}")
    app.run(
        debug=False,
        host='0.0.0.0',
        port=PORT
    )
