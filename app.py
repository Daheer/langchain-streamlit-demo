import pyngrok as ngrok
import speech_recognition as sr
import streamlit as st
from audiorecorder import audiorecorder
import speech_recognition 
from pydub import AudioSegment
import whisper
import pickle
import os 
from typing import Tuple
from torch.cuda import is_available as gpu_ready
from langchain.agents import load_tools, initialize_agent
import langchain
from langchain.llms import OpenAI
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain import OpenAI, LLMChain
from langchain.utilities import SerpAPIWrapper
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor

with open('openai_api_key.txt') as open_ai:
  openai_api_key = open_ai.read()
with open('serpapi_api_key.txt') as serp_api:
  serpapi_api_key = serp_api.read()

#openai_api_key = ''
#serp_api_key = ''

os.environ['OPENAI_API_KEY'] = openai_api_key
os.environ['SERPAPI_API_KEY'] = serp_api_key

search = SerpAPIWrapper()
tools = [
    Tool(
        name = "Search",
        func=search.run,
        description="useful for when you need to answer questions about current events"
    )
]
prefix = """Have a conversation with a human, answering the following questions as best you can. You have access to the following tools:"""
suffix = """Begin!"

{chat_history}
Question: {input}
{agent_scratchpad}"""

prompt = ZeroShotAgent.create_prompt(
    tools, 
    prefix=prefix, 
    suffix=suffix, 
    input_variables=["input", "chat_history", "agent_scratchpad"]
)
memory = ConversationBufferMemory(memory_key="chat_history")
llm_chain = LLMChain(llm=OpenAI(temperature=0), prompt=prompt)
agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
agent_chain = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True, memory=memory)

if not os.path.exists('agent_chain.bin'):
  with open('agent_chain.bin', 'wb') as f:
      pickle.dump(agent_chain, f)

### BACKEND / NLP ### 

def transcribe_whisper(audio_path: str) -> Tuple[str, bool]:

  '''
  Get the transcription of speech contained in an audio file.
  Uses OpenAI's whisper model.


  Parameters
  ----------
  audio_path: python string containing the filepath
              of the audio file to be transcribed,
              audio file can be in .wav or .mp3 formats.

  Returns
  -------
  Tuple (transcription, success_state) containing transcribed
  text and True if English speech was detected in the audio, 
  otherwise returns 'No speech detected' and False

  '''

  # Load base whisper model from OpenAI
  model = whisper.load_model("base")

  # Move model to GPU as it doesn't work on CPU
  model = model.to('cuda')

  # Preprocess audio file to be 30 seconds long
  audio = whisper.load_audio(audio_path)
  audio = whisper.pad_or_trim(audio)

  # Get mel spectrogram of the audio
  mel = whisper.log_mel_spectrogram(audio).to(model.device)
  options = whisper.DecodingOptions()

  # Get audio transcription
  result = whisper.decode(model, mel, options)

  # Detect language spoken in the audio
  _, probs = model.detect_language(mel)

  # Only return transcription if detected language is English 
  if max(probs, key=probs.get) == 'en':
    return result.text, True
  
  else: return 'No speech detected', False

def transcribe_google(audio_path: str) -> Tuple[str, bool]:

  '''
  Get the transcription of speech contained in an audio file.
  Uses Google voice recognition API.

  Parameters
  ----------
  audio_path: python string containing the filepath
              of the audio file to be transcribed,
              audio file can only be in .wav format.

  Returns
  -------
  Tuple (transcription, success_state) containing transcribed
  text and True if speech was detected in the audio, 
  otherwise returns 'No speech detected' and False

  '''

  # Get recognizer from SpeechRecognition library
  r = sr.Recognizer()

  # Preprocess audio
  audio_file = sr.AudioFile(audio_path)
  with audio_file as source:
    audio = r.record(source)
  
  # Get audio transcription using Google API
  result = r.recognize_google(audio, show_all = True)
  
  # Only return transcription if speech is detected
  if isinstance(result, dict):
    return result['alternative'][0]['transcript'], True
  else:
    return 'No speech detected', False
    
def respond(prompt: str) -> Tuple[str, bool]:

  '''
  Returns response from a Langchain language model using
  speech transcribed from audio recording as prompt.
  Uses OpenAI's GPT-3 on Langchain.

  Parameters
  ----------
  prompt: transcribed speech from audio recording

  Returns
  -------
  Tuple (llm_response, success_state) containing response
  from language model and True if there's a valid prompt.
  Otherwise asks user to try recording a prompt again and False.

  '''

  # Only respond if prompt is valid
  if prompt == 'No speech detected':
    return 'Try again?', False
  else:
    with open('agent_chain.bin', 'rb') as f:
      agent_chain = pickle.load(f)
    response = agent_chain.run(prompt)
    with open('agent_chain.bin', 'wb') as f:
      pickle.dump(agent_chain, f)
    return response, True

### FRONTEND / STREAMLIT ###

# Project title
title = 'Speak to your personal AI assistant backed up by OpenAI GPT-3'
st.markdown(f"<h1 style='text-align: center; color: grey;'>{title}</h1>", unsafe_allow_html=True)

# Define layout
col_1, col_2, col_3 = st.columns(3)

# Column 1: Recording Section
with col_1:
  st.markdown(f"<h3 style='text-align: center; color: grey;'>Record your prompt</h3>", unsafe_allow_html=True)
  
  # Streamlit audio recorder, from: https://github.com/theevann/streamlit-audiorecorder 
  with st.columns(3)[1]:
    audio = audiorecorder("🎙️", "...")

if len(audio) > 0:

  with col_1:
    st.audio(audio.tobytes())

  # Save audio in .mp3 format
  wav_file = open("audio_mp3.mp3", "wb")
  wav_file.write(audio.tobytes())
                                          # Both formats are kept because Google API doesn't work
                                          #  with .mp3 files
  # Convert .mp3 audio to .wav
  sound = AudioSegment.from_file('audio_mp3.mp3') 
  sound.export("audio_wav.wav", format="wav")

  # Column 2: Transcription Section
  with col_2:
    st.markdown("<h3 style='text-align: center; color: grey;'>Transcription</h3>", unsafe_allow_html=True)

    with st.spinner("Transcribing..."):

      # Use Whisper model if GPU is available, otherwise use Google model
      res_trans = transcribe_whisper('audio_wav.wav') if gpu_ready() else transcribe_google('audio_wav.wav')
    
    st.write(res_trans[0])
    if res_trans[1]:
      st.success('Transcription Successful', icon = '✅')  
    else: st.error('Transcription Failed', icon = '❌')
  
  with col_3:
    st.markdown("<h3 style='text-align: center; color: grey;'>Response</h3>", unsafe_allow_html=True)
    
    #with st.spinner("Thinking..."):
    res_think = respond(res_trans[0])
    
    st.write(res_think[0])
    if res_think[1]: 
      st.success('Response Success', icon = '✅')  
    else: st.error('Response Failed', icon = '❌')  