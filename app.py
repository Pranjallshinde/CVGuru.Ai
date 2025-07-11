from flask import Flask, render_template, request, jsonify, session
import os

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'railway-secret-key-123')

# Railway port configuration
PORT = int(os.environ.get('PORT', 8080))

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
        
        text_content = ""
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages[:2]:
                page_text = page.extract_text()
                if page_text:
                    text_content += page_text + "\n"
        
        if not text_content.strip():
            return render_template('predict.html', error="Could not extract text from PDF")
        
        text_content = text_content[:3000]
        
        prompt = f"""Generate 10 interview questions for {job_title}:
        
        Resume: {text_content}
        
        Format: 1. Question"""
        
        response = model.generate_content(prompt)
        
        questions = []
        for line in response.text.split('\n'):
            line = line.strip()
            if line and line[0].isdigit():
                question = line.split('.', 1)[-1].strip()
                if len(question) > 10:
                    questions.append(question)
        
        session['questions'] = questions[:10]
        session['job_title'] = job_title
        session['resume_text'] = text_content[:1500]
        
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
            return jsonify({'error': 'No questions found'})
        
        api_key = os.environ.get('GEMINI_API_KEY')
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        prompt = f"""Brief answers for {job_title} questions:
        
        Resume: {resume_text}
        Questions: {chr(10).join([f"{i+1}. {q}" for i, q in enumerate(questions)])}
        
        Format: ANSWER_1: [answer]"""
        
        response = model.generate_content(prompt)
        
        import re
        answers = {}
        matches = re.findall(r'ANSWER_(\d+):\s*(.*?)(?=ANSWER_\d+:|$)', response.text, re.DOTALL)
        
        for match in matches:
            answers[int(match[0])] = match[1].strip()
        
        return jsonify({'success': True, 'structured_answers': answers})
        
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/how_to_use')
def how_to_use():
    return render_template('how_to_use.html')

if __name__ == '__main__':
    print(f"Starting Flask app on port {PORT}")
    app.run(
        debug=False,
        host='0.0.0.0',
        port=PORT
    )
