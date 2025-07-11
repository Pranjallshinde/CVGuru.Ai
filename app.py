from flask import Flask, render_template, request, jsonify, session
import os

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'railway-secret-key-123')

# Railway port configuration
PORT = int(os.environ.get('PORT', 8080))

@app.route('/')
def index():
    # No rate limiting for home page
    return render_template('predict.html')

@app.route('/login')
def login():
    return render_template('auth/login.html')  # if login.html is in templates/auth/


@app.route('/predict')
def index1():
    # No rate limiting for form page
    return render_template('predict.html')

@app.route('/test_generate', methods=['POST'])
def test_generate():
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
        return render_template('predict.html', error="Please enter a job title.")
    
    # Extract text from the PDF file
    text_content = ""
    if file and file.filename.endswith('.pdf'):
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
    
    # Limit text length to avoid token limits
    if len(text_content) > 10000:
        text_content = text_content[:10000] + "..."
    
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
        return render_template('predict.html', 
                             error="API quota exceeded. Please try again later.")
    
    # Check if it's a valid resume
    if response.text.strip().lower().startswith("this is not a resume"):
        return render_template('predict.html', 
                             cresults=["The uploaded file doesn't look like a resume. Please upload a proper resume."])
    
    # Process questions
    questions = response.text.split("\n") if response.text else []
    questions = [q.strip() for q in questions if q.strip() and len(q.strip()) > 10]
    
    # Store data in session for answer generation
    session['questions'] = questions
    session['resume_text'] = text_content
    session['job_title'] = job_title
    
    # Return questions with option to generate answers
    return render_template('questions_result.html', 
                         questions=questions, 
                         job_title=job_title)

@app.route('/generate_answers', methods=['POST'])
def generate_answers():
    """Generate sample answers for the interview questions"""
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
    
    # Create prompt for generating answers - FIXED FORMAT
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
        return jsonify({'error': 'API quota exceeded. Please try again later.'})
    
    # Process answers - FIXED PARSING
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
    
    return jsonify({
        'success': True,
        'structured_answers': structured_answers,
        'total_questions': len(questions)
    })

@app.route('/how_to_use')
def how_to_use():
    # No rate limiting for info page
    return render_template('how_to_use.html')

@app.route('/interview_prep')
def interview_prep():
    return render_template('interview_prep.html')


if __name__ == "__main__":
    print("🚀 Starting CVGuru Interview Prep with Answer Suggestions")
    app.run(debug=True)
    
# if __name__ == '__main__':
#     print(f"Starting Flask app on port {PORT}")
#     app.run(
#         debug=False,
#         host='0.0.0.0',
#         port=PORT
    )
