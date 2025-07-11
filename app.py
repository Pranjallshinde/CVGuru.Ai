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
        
        # CHANGE 1: Extract text from PDF with better error handling
        text_content = ""
        try:
            with pdfplumber.open(file) as pdf:
                # Extract text from first 3 pages to get comprehensive content
                for page in pdf.pages[:3]:  # Increased from 2 to 3 pages
                    page_text = page.extract_text()
                    if page_text:
                        text_content += page_text + "\n"
        except Exception as pdf_error:
            return render_template('predict.html', error=f"Error reading PDF file: {str(pdf_error)}")
        
        if not text_content.strip():
            return render_template('predict.html', error="Could not extract text from PDF")
        
        # CHANGE 2: Resume validation - Check if uploaded document is actually a resume
        resume_validation_prompt = f"""
        Analyze the following text and determine if this is a professional resume/CV.
        
        Text to analyze: {text_content[:2000]}
        
        A resume typically contains:
        - Personal information (name, contact details)
        - Work experience or employment history
        - Education details
        - Skills section
        - Professional summary or objective
        
        Respond with only "VALID_RESUME" if this appears to be a resume, or "NOT_RESUME" if it's not a resume.
        If it's not a resume, briefly explain what type of document it appears to be.
        """
        
        # Validate if the document is a resume
        validation_response = model.generate_content(resume_validation_prompt)
        validation_result = validation_response.text.strip()
        
        # CHANGE 3: Check validation result and reject non-resume documents
        if not validation_result.startswith("VALID_RESUME"):
            return render_template('predict.html', 
                                 error=f"The uploaded document doesn't appear to be a resume. {validation_result.replace('NOT_RESUME', '').strip()}")
        
        # Limit text content for processing
        text_content = text_content[:4000]  # Increased limit for better context
        
        # CHANGE 4: Enhanced prompt for better question generation
        prompt = f"""
        Based on the following resume content, generate exactly 10 relevant and specific interview questions for the position of {job_title}.
        
        Resume Content: {text_content}
        
        Requirements:
        1. Questions should be specific to the candidate's experience mentioned in the resume
        2. Include both technical and behavioral questions appropriate for {job_title}
        3. Questions should help assess the candidate's fit for the role
        4. Format each question as: "1. Question text"
        5. Make questions challenging but fair
        6. Include questions about specific projects, technologies, or experiences mentioned in the resume
        
        Generate exactly 10 questions in numbered format.
        """
        
        response = model.generate_content(prompt)
        
        # CHANGE 5: Improved question parsing with better validation
        questions = []
        for line in response.text.split('\n'):
            line = line.strip()
            # Check if line starts with a number followed by a dot or parenthesis
            if line and (line[0].isdigit() or line.startswith('Q')):
                # Extract question text after the number/marker
                if '.' in line:
                    question = line.split('.', 1)[-1].strip()
                elif ')' in line:
                    question = line.split(')', 1)[-1].strip()
                else:
                    question = line
                
                # Only add substantial questions (more than 15 characters)
                if len(question) > 15:
                    questions.append(question)
        
        # Ensure we have exactly 10 questions
        questions = questions[:10]
        
        # CHANGE 6: Store additional context in session for better answer generation
        session['questions'] = questions
        session['job_title'] = job_title
        session['resume_text'] = text_content[:2000]  # Store more context
        session['full_resume'] = text_content  # Store full resume for comprehensive answers
        
        return render_template('questions_result.html',
                              questions=questions,
                              job_title=job_title)
        
    except Exception as e:
        # CHANGE 7: Better error handling with more specific error messages
        error_message = str(e)
        if "quota" in error_message.lower():
            error_message = "API quota exceeded. Please try again later."
        elif "api" in error_message.lower():
            error_message = "API service temporarily unavailable. Please try again."
        
        return render_template('predict.html', error=f"Error: {error_message}")

@app.route('/generate_answers', methods=['POST'])
def generate_answers():
    try:
        import google.generativeai as genai
        
        # Retrieve session data
        questions = session.get('questions', [])
        job_title = session.get('job_title', '')
        resume_text = session.get('resume_text', '')
        full_resume = session.get('full_resume', resume_text)  # CHANGE 8: Use full resume context
        
        if not questions:
            return jsonify({'error': 'No questions found. Please generate questions first.'})
        
        api_key = os.environ.get('GEMINI_API_KEY')
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        
     # Enhanced prompt for STAR method answers with strict formatting
        prompt = f"""
        Generate professional sample answers for the following interview questions using the STAR method (Situation, Task, Action, Result).
        
        Job Position: {job_title}
        Candidate's Resume: {full_resume}
        
        CRITICAL FORMATTING REQUIREMENTS:
        1. Each answer MUST follow this exact format:
        *Situation:* [Describe the context and background]
        *Task:* [Describe what needed to be accomplished]
        *Action:* [Explain the specific actions taken]
        *Result:* [Share the outcomes and what was learned]
        
        2. Each component (Situation, Task, Action, Result) MUST be on a separate line
        3. Each component MUST start with the bold label exactly as shown: *Situation:, **Task:, **Action:, **Result:*
        4. Do NOT combine components in one paragraph
        5. Keep each component concise (1-2 sentences)
        
        Content Requirements:
        - Base answers on the actual experience and skills mentioned in the resume
        - Make answers specific and relevant to the {job_title} role
        - Use first-person perspective ("I did...", "I achieved...")
        - Include quantifiable results where possible
        - Show problem-solving skills and professional growth
        
        Questions to answer:
        {chr(10).join([f"{i+1}. {q}" for i, q in enumerate(questions)])}
        
        Format your response exactly as:
        ANSWER_1:
        *Situation:* [situation description]
        *Task:* [task description]
        *Action:* [action description]
        *Result:* [result description]
        
        ANSWER_2:
        *Situation:* [situation description]
        *Task:* [task description]
        *Action:* [action description]
        *Result:* [result description]
        
        Continue this pattern for all questions. Remember: Each STAR component MUST be on its own line with bold labels.
        """
        
        response = model.generate_content(prompt)
        
        # CHANGE 10: Improved answer parsing with better error handling
        import re
        answers = {}
        
        # Parse answers using regex
        matches = re.findall(r'ANSWER_(\d+):\s*(.*?)(?=ANSWER_\d+:|$)', response.text, re.DOTALL)
        
        for match in matches:
            answer_num = int(match[0])
            answer_text = match[1].strip()
            
          # Validate and format answers while preserving STAR structure
            if len(answer_text) > 20:  # Ensure substantial answers
                # Clean up excessive whitespace but preserve line breaks for STAR components
                lines = answer_text.split('\n')
                cleaned_lines = []
                
                for line in lines:
                    line = line.strip()
                    if line:  # Only keep non-empty lines
                        # Convert markdown bold to HTML bold for display
                        line = line.replace('*Situation:*', '<strong>Situation:</strong>')
                        line = line.replace('*Task:*', '<strong>Task:</strong>')
                        line = line.replace('*Action:*', '<strong>Action:</strong>')
                        line = line.replace('*Result:*', '<strong>Result:</strong>')
                        cleaned_lines.append(line)
                
                # Join with HTML line breaks for proper display
                formatted_answer = '<br>'.join(cleaned_lines)
                answers[answer_num] = formatted_answer
        
        # CHANGE 12: Generate fallback answers if parsing fails
        if len(answers) < len(questions):
            for i in range(1, len(questions) + 1):
                if i not in answers:
                    # Create a basic STAR method template answer
                    fallback_answer = f"In my role as mentioned in my resume, I encountered a situation where I needed to demonstrate skills relevant to {job_title}. My task was to deliver results that align with the job requirements. I took specific actions based on my experience and training. As a result, I successfully achieved the objectives and gained valuable experience that makes me suitable for this position."
                    answers[i] = fallback_answer
        
        return jsonify({
            'success': True, 
            'structured_answers': answers,
            'total_questions': len(questions),
            'method_used': 'STAR Method'  # CHANGE 13: Indicate method used
        })
        
    except Exception as e:
        # CHANGE 14: Enhanced error handling for answer generation
        error_message = str(e)
        if "quota" in error_message.lower():
            error_message = "API quota exceeded for answer generation. Please try again later."
        elif "timeout" in error_message.lower():
            error_message = "Request timeout. Please try again."
        
        return jsonify({'error': f'Answer generation failed: {error_message}'})

@app.route('/how_to_use')
def how_to_use():
    return render_template('how_to_use.html')

# CHANGE 15: Add error handlers for better user experience
@app.errorhandler(404)
def not_found_error(error):
    return render_template('predict.html', error="Page not found"), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('predict.html', error="Internal server error occurred"), 500

# CHANGE 16: Add file size limit to prevent large file uploads
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

@app.errorhandler(413)
def too_large(e):
    return render_template('predict.html', error="File too large. Please upload a file smaller than 16MB."), 413

if __name__ == '__main__':
    print(f"Starting Flask app on port {PORT}")
    # CHANGE 17: Add startup message with feature information
    print("Features enabled:")
    print("- Resume validation")
    print("- STAR method answers")
    print("- Enhanced error handling")
    
    app.run(
        debug=False,
        host='0.0.0.0',
        port=PORT
    )
