# app.py

from flask import Flask, render_template, request, jsonify, session
from flask_session import Session
import os
import logging
from openai import OpenAI
import assemblyai as aai
import anthropic
import uuid
import PyPDF2
import io

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

# Load environment variables
ASSEMBLYAI_API_KEY = os.getenv('ASSEMBLYAI')
OPENAI_API_KEY = os.getenv('OPENAI')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC')

# Initialize clients
aai.settings.api_key = ASSEMBLYAI_API_KEY
openai_client = OpenAI(api_key=OPENAI_API_KEY)
anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# Ensure static directory exists
os.makedirs('static', exist_ok=True)

@app.route('/')
def index():
    logger.info("Rendering index page")
    session['conversation'] = []
    return render_template('index.html')

@app.route('/process_audio', methods=['POST'])
def process_audio():
    logger.info("Received audio processing request")
    audio_file = request.files['audio']
    role = request.form['role']
    interview_type = request.form['interview_type']
    resume_file = request.files['resume']

    logger.info(f"Processing audio for role: {role}, interview type: {interview_type}")

    # Process the resume
    resume_text = process_resume(resume_file)
    logger.info("Resume processed successfully")

    # 1. Speech to Text using AssemblyAI
    transcript = transcribe_audio(audio_file)
    logger.info(f"Transcription result: {transcript}")

    # 2. Generate response using Claude API
    claude_response = generate_claude_response(transcript, role, interview_type, resume_text)
    logger.info(f"Claude response: {claude_response}")

    # 3. Text to Speech using OpenAI (only for the latest response)
    audio_response = text_to_speech(claude_response)
    logger.info(f"Generated audio response: {audio_response}")

    # 4. Update conversation history
    session['conversation'].append({"role": "user", "content": transcript})
    session['conversation'].append({"role": "assistant", "content": claude_response})

    return jsonify({
        'transcript': transcript,
        'response': claude_response,
        'audio_url': audio_response,
        'conversation': session['conversation']
    })

def process_resume(resume_file):
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(resume_file.read()))
        resume_text = ""
        for page in pdf_reader.pages:
            resume_text += page.extract_text()
        return resume_text
    except Exception as e:
        logger.error(f"Error processing resume: {str(e)}")
        return ""

def transcribe_audio(audio_file):
    logger.info("Starting audio transcription with AssemblyAI")
    transcriber = aai.Transcriber()
    config = aai.TranscriptionConfig(
        punctuate=True,
        format_text=True
    )
    transcript = transcriber.transcribe(audio_file, config=config)
    logger.info("Audio transcription completed")
    return transcript.text

def generate_claude_response(transcript, role, interview_type, resume_text):
    logger.info("Generating Claude response")
    try:
        conversation = session.get('conversation', [])
        system_prompt = f"""You are an interviewer conducting a {interview_type} interview for a {role} position. 
        The candidate has provided the following resume:

        {resume_text}

        Based on this resume and the candidate's previous answers, respond to their latest answer and ask a relevant follow-up question. You don't have to read out the whole resume in any questions, just ask about one thing at a time. Keep your questions short and real interview like, in a real interview, questions are short and too the point. The interviewer is also to the point.  
        Keep the responses concise and professional, focusing on the candidate's experience and skills as they relate to the {role} position."""

        message = anthropic_client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1000,
            system=system_prompt,
            messages=conversation + [{"role": "user", "content": transcript}]
        )
        logger.info("Claude response generated successfully")
        return message.content[0].text
    except Exception as e:
        logger.error(f"Error generating Claude response: {str(e)}")
        return "I apologize, but I encountered an error while processing your response. Could you please try again?"

def text_to_speech(text):
    logger.info("Converting text to speech with OpenAI")
    try:
        response = openai_client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=text
        )
        
        # Generate a unique filename for each audio response
        unique_filename = f"response_{uuid.uuid4()}.mp3"
        output_file = os.path.join("static", unique_filename)
        
        with open(output_file, 'wb') as f:
            for chunk in response.iter_bytes():
                f.write(chunk)
        logger.info(f"Audio file saved: {output_file}")
        return f"/static/{unique_filename}"
    except Exception as e:
        logger.error(f"Error in text-to-speech conversion: {str(e)}")
        return None

if __name__ == '__main__':
    logger.info("Starting Flask application")
    app.run(debug=True)
