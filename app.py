# app.py

from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_session import Session
import os
import logging
from openai import OpenAI
import assemblyai as aai
import uuid
import PyPDF2
import io
from groq import Groq
from authlib.integrations.flask_client import OAuth
from authlib.integrations.base_client.errors import OAuthError
# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

# Load environment variables
ASSEMBLYAI_API_KEY = os.getenv('ASSEMBLYAI')
OPENAI_API_KEY = os.getenv('OPENAI')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
GOOGLE_CLIENT_ID = os.getenv('GOOGLE_CLIENT_ID')
GOOGLE_CLIENT_SECRET = os.getenv('GOOGLE_CLIENT_SECRET')
print(GOOGLE_CLIENT_SECRET)
print(GOOGLE_CLIENT_ID)
# Print out environment variables for debugging
logger.debug(f"GOOGLE_CLIENT_ID: {GOOGLE_CLIENT_ID}")
logger.debug(f"GOOGLE_CLIENT_SECRET: {'*' * len(GOOGLE_CLIENT_SECRET) if GOOGLE_CLIENT_SECRET else 'Not set'}")

# Initialize clients
aai.settings.api_key = ASSEMBLYAI_API_KEY
openai_client = OpenAI(api_key=OPENAI_API_KEY)
groq_client = Groq(api_key = GROQ_API_KEY)

# Ensure static directory exists
os.makedirs('static', exist_ok=True)

# OAuth Setup
oauth = OAuth(app)


google = oauth.register(
    name='google',
    client_id=GOOGLE_CLIENT_ID,
    client_secret=GOOGLE_CLIENT_SECRET,
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={
        'scope': 'openid email profile',
        'prompt': 'select_account'
    }
)

google = oauth.register(
    name='google',
    client_id=GOOGLE_CLIENT_ID,
    client_secret=GOOGLE_CLIENT_SECRET,
    access_token_url='https://accounts.google.com/o/oauth2/token',
    access_token_params=None,
    authorize_url='https://accounts.google.com/o/oauth2/auth',
    authorize_params=None,
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    api_base_url='https://www.googleapis.com/oauth2/v1/',
    client_kwargs={'scope': 'email profile'},
)

@app.route('/')
def landing():
    logger.info("Rendering landing page")
    return render_template('landing.html')

@app.route('/login')
def login():
    redirect_uri = url_for('authorized', _external=True)
    logger.debug(f"Redirect URI: {redirect_uri}")
    return google.authorize_redirect(redirect_uri)

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('landing'))

@app.route('/login/callback')
def authorized():
    try:
        logger.debug("Entering authorized callback")
        token = google.authorize_access_token()
        logger.debug(f"Received token: {token}")
        resp = google.get('https://www.googleapis.com/oauth2/v3/userinfo')
        logger.debug(f"User info response: {resp.json()}")
        user_info = resp.json()
        session['user'] = user_info
        logger.info(f"User {user_info.get('email')} successfully authenticated")
        return redirect(url_for('index'))
    except OAuthError as e:
        logger.error(f"OAuth Error: {str(e)}")
        return f"An OAuth error occurred: {str(e)}", 400
    except Exception as e:
        logger.error(f"Unexpected error during authorization: {str(e)}")
        return f"An unexpected error occurred: {str(e)}", 500

@app.route('/app')
def index():
    if 'user' not in session:
        logger.info("User not in session, redirecting to login")
        return redirect(url_for('login'))
    logger.info('Rendering app page')
    session['conversation'] = []
    return render_template('index.html')

@app.route('/process_audio', methods=['POST'])
def process_audio():
    logger.info("Received audio processing request")
    audio_file = request.files['audio']
    role = request.form['role']
    interview_type = request.form['interview_type']
    resume_file = request.files['resume']
    difficulty_input = request.form['difficulty']
    difficulty = ""

    if difficulty_input == "hard":
        difficulty = "Keep the interview very difficult, ask very hard questions (whether techincal or HR like) from the user. Remember to ask hard quality questions."
    else:
        difficulty = "User has asked for easy interview, but ask quality questions."

    logger.info(f"Processing audio for role: {role}, interview type: {interview_type}")

    # Process the resume
    resume_text = process_resume(resume_file)
    logger.info("Resume processed successfully")

    # 1. Speech to Text using AssemblyAI
    transcript = transcribe_audio(audio_file)
    logger.info(f"Transcription result: {transcript}")

    # 2. Generate response using groq API
    groq_response = generate_groq_response(transcript, role, interview_type, resume_text, difficulty)
    logger.info(f"groq response: {groq_response}")

    # 3. Text to Speech using OpenAI (only for the latest response)
    audio_response = text_to_speech(groq_response)
    logger.info(f"Generated audio response: {audio_response}")

    # 4. Update conversation history
    session['conversation'].append({"role": "user", "content": transcript})
    session['conversation'].append({"role": "assistant", "content": groq_response})

    return jsonify({
        'transcript': transcript,
        'response': groq_response,
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

def generate_groq_response(transcript, role, interview_type, resume_text, difficulty):
    logger.info("Generating groq response")
    try:
        conversation = session.get('conversation', [])
        system_prompt = f"""You are an experienced technical interviewer conducting a {interview_type} interview for a {role} position. Your goal is to rigorously assess the candidate's technical skills and problem-solving abilities through direct, focused questioning and dynamic follow-ups.
Difficult set for the interview: {difficulty}
Resume: {resume_text}

Guidelines:
1. Ask pointed technical questions based on the required skills for the {role} position.
2. Keep initial questions concise and technically specific.
3. When the candidate provides an answer, use it as a springboard for follow-up technical questions, even if the topics aren't explicitly mentioned in their resume.
4. Dive deeper into areas where the candidate shows knowledge, exploring the breadth and depth of their understanding.
5. If a candidate cannot answer a question or struggles significantly, acknowledge their attempt and move on to a different question.
6. Adapt your questions based on the candidate's responses, adjusting difficulty and exploring tangential but relevant technical areas.
7. Maintain a balance between scripted questions and spontaneous, knowledge-probing follow-ups.

Question Types:
- Coding problems (e.g., algorithm implementation, data structure usage)
- System design questions
- Language-specific questions (based on the required skills for the role)
- Debugging scenarios
- Performance optimization problems
- Technical best practices and patterns
- Conceptual questions on related technologies or methodologies

Example Flow:
1. Initial question: "Can you explain the architecture of a project you've worked on that used machine learning?"
2. Candidate mentions using CNNs for image classification.
3. Follow-up: "Interesting. Have you worked with any other deep learning architectures? What about transformers?"
4. Based on their response, you might ask about specific aspects of transformers, their applications, or comparisons to CNNs.

Response Format:
1. Briefly acknowledge the candidate's previous answer.
2. If the answer opens up new areas of discussion, ask a follow-up question to explore deeper.
3. If the candidate couldn't answer, move on to a new topic without dwelling on it.
4. For solved problems, ask about optimizations, alternative approaches, or related concepts.

Continuously evaluate the candidate's:
- Technical knowledge depth and breadth
- Problem-solving approach
- Code quality and efficiency (for coding questions)
- Communication of technical concepts
- Ability to handle pressure and unknown topics
- Capability to make connections between different technical concepts

Remember to keep the interview challenging but not overwhelming. Use the candidate's responses to guide the direction of the interview, exploring various technical areas relevant to the role. This approach allows for a comprehensive assessment of both the candidate's listed skills and their overall technical acumen."""       

        chat_completion = groq_client.chat.completions.create(
        messages=[
            {
                "role":"system",
                "content": system_prompt,
            },
            {
                "role": "user", "content": transcript,
            }
        ],
        model="llama-3.2-90b-vision-preview",
        )
        print(chat_completion.choices[0].message.content)
        logger.info("groq response generated successfully")
        return chat_completion.choices[0].message.content
    except Exception as e:
        logger.error(f"Error generating groq response: {str(e)}")
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
