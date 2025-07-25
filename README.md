��#   C V M a p . A i 
 # CVGuru - AI Interview Preparation

AI-powered interview preparation tool that generates personalized questions and answers based on your resume.

## Features

- **Resume Analysis**: Upload PDF resume for AI analysis
- **Personalized Questions**: Get 15 tailored interview questions
- **Sample Answers**: AI-generated answers using STAR method
- **Modern UI**: Clean, responsive design
- **Rate Limiting**: Built-in API protection

## Quick Start

1. **Clone the repository**
2. **Install dependencies**: `pip install -r requirements.txt`
3. **Set environment variables**:
   - `GEMINI_API_KEY`: Your Google Gemini API key
   - `SECRET_KEY`: Flask secret key
4. **Run locally**: `python app.py`
5. **Deploy to Vercel**: Push to GitHub and connect to Vercel

## Environment Variables

Create a `.env` file or set in Vercel:

\`\`\`
GEMINI_API_KEY=your_gemini_api_key_here
SECRET_KEY=your_secret_key_here
\`\`\`

## Deployment

This app is optimized for Vercel deployment with minimal dependencies to stay under the 250MB limit.

## Usage

1. Upload your PDF resume
2. Select target job role
3. Generate interview questions
4. Get AI-powered sample answers

## Tech Stack

- **Backend**: Flask (Python)
- **AI**: Google Gemini API
- **PDF Processing**: pdfplumber
- **Frontend**: Bootstrap 5, HTML/CSS/JS
- **Deployment**: Vercel

## License

MIT License

 
