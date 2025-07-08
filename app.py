from flask import Flask, render_template, request, jsonify, session
import os

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'railway-secret-key-123')

# Railway uses PORT environment variable
PORT = int(os.environ.get('PORT', 5000))

@app.route('/')
def index():
    return render_template('front.html')

@app.route('/test')
def test():
    return "Flask is working on Railway! 🚀"

@app.route('/api/health')
def health():
    return jsonify({
        'status': 'healthy', 
        'platform': 'railway',
        'port': PORT
    })

@app.route('/predict')
def predict():
    return render_template('predict.html')

# Only import heavy libraries when needed
@app.route('/generate_questions', methods=['POST'])
def generate_questions():
    try:
        # Import only when needed to reduce cold start
        import google.generativeai as genai
        import pdfplumber
        
        # Configure API
        api_key = os.environ.get('GEMINI_API_KEY')
        if not api_key:
            return render_template('predict.html', error="API key not configured. Please contact admin.")
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        # Check file upload
        if 'pdf_file' not in request.files:
            return render_template('predict.html', error="No file uploaded")
        
        file = request.files['pdf_file']
        job_title = request.form.get('job_title', '')
        
        if not file.filename or not job_title:
            return render_template('predict.html', error="Please select file and job title")
        
        # Extract text
        text_content = ""
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages[:3]:  # Limit to first 3 pages
                page_text = page.extract_text()
                if page_text:
                    text_content += page_text + "\n"
        
        if not text_content.strip():
            return render_template('predict.html', error="Could not extract text from PDF")
        
        # Limit text length for Railway
        text_content = text_content[:4000]
        
        # Generate questions
        prompt = f"""Generate exactly 10 interview questions for {job_title} position based on this resume:
        
        {text_content}
        
        Format each question as:
        1. Question here
        2. Question here
        etc."""
        
        response = model.generate_content(prompt)
        
        # Parse questions
        questions = []
        for line in response.text.split('\n'):
            line = line.strip()
            if line and line[0].isdigit():
                question = line.split('.', 1)[-1].strip()
                if len(question) > 10:
                    questions.append(question)
        
        # Store in session
        session['questions'] = questions[:10]
        session['job_title'] = job_title
        session['resume_text'] = text_content[:2000]  # Store limited text
        
        return render_template('questions_result.html', 
                             questions=questions[:10], 
                             job_title=job_title)
        
    except Exception as e:
        return render_template('predict.html', error=f"Error: {str(e)}")

@app.route('/generate_answers', methods=['POST'])
def generate_answers():
    try:
        import google.generativeai as genai
        
        questions = session.get('questions', [])
        job_title = session.get('job_title', '')
        resume_text = session.get('resume_text', '')
        
        if not questions:
            return jsonify({'error': 'No questions found. Please generate questions first.'})
        
        api_key = os.environ.get('GEMINI_API_KEY')
        if not api_key:
            return jsonify({'error': 'API key not configured'})
            
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        # Generate answers
        prompt = f"""Create brief sample answers for these {job_title} interview questions based on the resume:

        Resume: {resume_text}
        
        Questions:
        {chr(10).join([f"{i+1}. {q}" for i, q in enumerate(questions)])}
        
        Format each answer as:
        ANSWER_1: [brief answer using STAR method if applicable]
        ANSWER_2: [brief answer using STAR method if applicable]
        etc."""
        
        response = model.generate_content(prompt)
        
        # Parse answers
        import re
        answers = {}
        matches = re.findall(r'ANSWER_(\d+):\s*(.*?)(?=ANSWER_\d+:|$)', response.text, re.DOTALL)
        
        for match in matches:
            answers[int(match[0])] = match[1].strip()
        
        return jsonify({
            'success': True, 
            'structured_answers': answers,
            'total_questions': len(questions)
        })
        
    except Exception as e:
        return jsonify({'error': f'Error generating answers: {str(e)}'})

@app.route('/how_to_use')
def how_to_use():
    return render_template('how_to_use.html')

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return render_template('predict.html', error="Page not found"), 404

@app.errorhandler(500)
def internal_error(error):
    return f"Internal server error: {str(error)}", 500

# Railway-specific configuration
if __name__ == '__main__':
    # Railway automatically sets PORT
    app.run(
        debug=False,  # Set to False for production
        host='0.0.0.0',  # Railway needs this
        port=PORT
    )
