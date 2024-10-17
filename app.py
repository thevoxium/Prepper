# app.py

from flask import Flask, render_template, request, jsonify, session
from flask_session import Session
import os
import logging
from openai import OpenAI
import assemblyai as aai
import anthropic
import uuid

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

    logger.info(f"Processing audio for role: {role}, interview type: {interview_type}")

    # 1. Speech to Text using AssemblyAI
    transcript = transcribe_audio(audio_file)
    logger.info(f"Transcription result: {transcript}")

    # 2. Generate response using Claude API
    claude_response = generate_claude_response(transcript, role, interview_type)
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

def generate_claude_response(transcript, role, interview_type):
    logger.info("Generating Claude response")
    try:
        conversation = session.get('conversation', [])
        message = anthropic_client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1000,
            system=f"You are an interviewer conducting a {interview_type} interview for a {role} position. Respond to the candidate's answer and ask a follow-up question. Keep the responses to the point, dont overspeak, keep it professional.",
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
