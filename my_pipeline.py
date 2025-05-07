# 📦 DEPENDENCIES & IMPORTS

# Standard Library
import os
import subprocess
from typing import List

# Third-Party Libraries
import yt_dlp
import whisper
import chromadb
from gtts import gTTS
import tempfile


# LangChain Components
from langchain.text_splitter import RecursiveCharacterTextSplitter
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langsmith import traceable, Client

###################################################################################################
# Import seprate files 
#from agent_runner import run_agent_pipeline
from openai import OpenAI
from dotenv import load_dotenv

##################################################################################################
# Initialize LangSmith client
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGSMITH_API_KEY"] = "LANGSMITH_API_KEY"

langsmith_client = Client()

##################################################################################################
# Set your OpenAI API key
load_dotenv()  # Loads from .env file

# Initialize ONCE (reuse this client everywhere)
client = OpenAI(api_key="YOUR_API_KEY")

#################################################################################################

ENABLE_AUDIO = True  # Set to False to run in quiet mode

################################################################################################

@traceable(name="Download YouTube Audio")
def download_youtube_audio(url, output_path='./'):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': f'{output_path}/%(title)s.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '8', # change to 8 IF NEEDED 
        }],
        'quiet': False,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info).rsplit('.', 1)[0] + '.mp3'
            print(f"✅ Audio saved as: {filename}")
            return filename
    except Exception as e:
        print(f"❌ Error: {e}")
        return None
##############################################################################################

##############################################################################################

@traceable(name="Transcribe Audio")
def transcribe_audio_with_whisper(audio_file):
    try:
        print("🗣️ Loading Whisper tiny model...")
        model = whisper.load_model("tiny") # CHNAGE TO FAST WHISPER IF NEEDED 
        
        print(f"🎧 Transcribing: {audio_file}")
        result = model.transcribe(audio_file)
        
        transcript = result['text']
        print("✅ Transcription completed.")
        return transcript
    except Exception as e:
        print(f"❌ Error during transcription: {e}")
        return None
#############################################################################################

#############################################################################################

@traceable(name="Split Text into Chunks")
def split_text_into_chunks(text, chunk_size=600, overlap=200):
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end]
        chunks.append(chunk)
        
        if end == text_length:
            break  # Stop if we've reached the end
        
        start += chunk_size - overlap  # Move forward with overlap
    
    print(f"✅ Split into {len(chunks)} chunks.")
    return chunks
###########################################################################################

###########################################################################################

@traceable(name="Embed and Store Chunks")
def embed_and_store_chunks(chunks, collection_name='youtube_transcripts'):
    try:
        # Load embedding model (you can switch to another model if you want)
        print("🧠 Loading embedding model...")
        embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Create Chroma DB client
        client = chromadb.PersistentClient(path="./chroma_db")
         # ✅ Clear old collection to avoid mixing
        client.delete_collection(name=collection_name)
        # Create or get collection
        collection = client.get_or_create_collection(name=collection_name)
        
        # Prepare and add data
        for i, chunk in enumerate(chunks):
            doc_id = f"chunk_{i}"
            embedding = embedder.encode(chunk).tolist()
            
            collection.add(
                ids=[doc_id],
                documents=[chunk],
                embeddings=[embedding]
            )
            print(f"✅ Saved {doc_id}")
        
        print(f"🎉 All {len(chunks)} chunks saved to Chroma DB collection '{collection_name}'.")
        return collection
    
    except Exception as e:
        print(f"❌ Error during embedding or storage: {e}")
        return None
    
######################################################################################################


######################################################################################################

def adapt_for_audience(answer: str, audience: str) -> str:
    if audience == "child":
        instruction = "Simplify this answer for a child aged 8-12. Use simple words, short sentences, and fun emojis."
    elif audience == "student":
        instruction = "Reformat this answer for a high school or university student. Use clear, structured academic language and bullet points if helpful."
    else:
        return answer

    client2 = OpenAI(api_key='YOUR_API_KEY')
    response = client2.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You rewrite answers to match different audiences."},
            {"role": "user", "content": f"{instruction}\n\n{answer}"}
        ]
    )
    return response.choices[0].message.content
#####################################################################################

def run_chatbot():
    client3 = OpenAI(api_key='YOUR_API_KEY')
    # Load embedding model
    embedder = SentenceTransformer('all-MiniLM-L6-v2')

    # Connect to ChromaDB
    client_chroma = chromadb.PersistentClient(path="./chroma_db")
    collection = client_chroma.get_collection(name='youtube_transcripts')

    print("🤖 Chatbot ready! Type 'exit' to stop.")
    print("Choose your guide: 🌟 AstroBuddy (for kids) | 🧠 SpaceMentor (for students)")

    # This lets the user pick the audience at launch.
    audience = input("Who is your guide? Type 'child' or 'student': ").strip().lower()

    while True:
        user_question = input("You: ")
        if user_question.lower() == 'exit':
            print("👋 Goodbye!")
            break

        # Embed user question
        question_embedding = embedder.encode(user_question).tolist()

        # Query top 3 chunks try (3,5,10) Raise the number of results if needed. 
        results = collection.query(
            query_embeddings=[question_embedding],
            n_results=3
        )

        retrieved_docs = results['documents'][0]
        

        if not retrieved_docs:
            print("Bot: I don't know.")
            continue

        # Build context
        context = "\n".join(retrieved_docs)

        # Build prompt for GPT-3.5
        system_prompt = (
            "You are a friendly and educational assistant for children and students. "
            "You answer space-related questions strictly usingw the provided context. "
            "Make your answers simple, engaging, and fun, using kid-friendly language, short sentences, and emojis when appropriate. "
            "If the answer is not in the context, reply: 'Hmm, this is not related to the content!'.")

        user_prompt = f"Context:\n{context}\n\nQuestion: {user_question}"

        # Ask GPT-3.5
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )

        answer = response.choices[0].message.content
        answer = adapt_for_audience(answer, audience)
        print(f"Bot: {answer}")

###############################################################################################################
 # Generate speech with gTTS

    if ENABLE_AUDIO:    
        try:
                tts = gTTS(answer)
                tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
                tts.save(tmp_file.name)

                # Play the audio (use appropriate command for your OS)
                if os.name == 'nt':  # Windows
                    os.system(f'start {tmp_file.name}')
                elif os.name == 'posix':  # macOS or Linux
                    os.system(f'afplay {tmp_file.name}' if sys.platform == 'darwin' else f'mpg123 {tmp_file.name}')
                else:
                    print("Audio playback not supported on this OS.")
        except Exception as audio_error:
                print(f"⚠️ Error playing audio: {audio_error}")

###############################################################################################################

def main():
    print("🚀 Starting full pipeline...")
    
    # Ask for YouTube video URL
    video_url = input("🎥 Please paste the YouTube video URL: ").strip()
    
    # Step 1: Download audio
    audio_file = download_youtube_audio(video_url)
    print(f"✅ Audio downloaded: {audio_file}")
    
    # Step 2: Transcribe audio
    transcript = transcribe_audio_with_whisper(audio_file)
    print(f"✅ Transcription done. Transcript length: {len(transcript)} characters")
    
    # Step 3: Split into chunks
    chunks = split_text_into_chunks(transcript)
    print(f"✅ Split into {len(chunks)} chunks.")
    
    # Step 4: Embed and store in ChromaDB
    embed_and_store_chunks(chunks)
    print("🎯 All steps completed successfully!")

##########################################################################


if __name__ == "__main__":
    print("🚀 Starting agent-integrated pipeline...")
    video_url = input("🎥 Please paste the YouTube video URL: ").strip()
    run_chatbot()




