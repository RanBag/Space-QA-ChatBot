# 📦 DEPENDENCIES & IMPORTS

# Standard Library
import streamlit as st
import tempfile
import chromadb
from chromadb.utils import embedding_functions
from gtts import gTTS
import yt_dlp
from openai import OpenAI
import openai


# LangChain Components
from langsmith import Client
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.stdout  import StdOutCallbackHandler
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.agents import initialize_agent, Tool
from langchain.chains import RetrievalQA

# 2. Environment setup
import os
from dotenv import load_dotenv
load_dotenv()

# Initialize Streamlit chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Initialize LangSmith client
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"

# Set OpenAI API key
openai_client = OpenAI(api_key='YOUR_API_KEY')
langsmith_client = Client()

# Import pipeline functions
from my_pipeline import (
    download_youtube_audio,
    transcribe_audio_with_whisper,
    split_text_into_chunks,
    embed_and_store_chunks
)   


# Initialize LangSmith callback
callback_manager = CallbackManager([StdOutCallbackHandler()])

# Initialize Chroma client and embedder
client_chroma = chromadb.PersistentClient(path="./chroma_db")
embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2",
    device="cpu"
)


# ⚠ Delete and recreate collection to ensure clean embedder match
if 'collection_initialized' not in st.session_state:
    try:
        client_chroma.delete_collection(name='youtube_transcripts')
    except Exception:
        pass  # Ignore if it doesn't exist
    collection = client_chroma.get_or_create_collection(
        name='youtube_transcripts',
        embedding_function=embedder
    )
    st.session_state.collection_initialized = True
else:
    collection = client_chroma.get_or_create_collection(
        name='youtube_transcripts'
    )

# Initialize Chroma DB
embedding_function = OpenAIEmbeddings()
db = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding_function
)



# Set up RetrievalQA chainx``
llm = ChatOpenAI(
    model="gpt-3.5-turbo-0125",  
    temperature=0,
    streaming=True,  # Recommended for LangSmith tracing
    verbose=True,  # Helps with debugging
    callbacks=[StdOutCallbackHandler()]  
)


qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=db.as_retriever(),
    chain_type="stuff"                 
)


# Streamlit app UI
st.title("👨‍🚀 Space QA ChatBot — Your Universe of Knowledge!🎥")

# Audience selection
audience = st.radio(
    "Choose your guide:",
    ["🚀AstroBuddy (for kids)", "🧠 SpaceMentor (for students)"],
    index=0
)
audience_role = "child" if "AstroBuddy" in audience else "student"

# Helper: Get YouTube metadata (title + description)
def get_youtube_metadata(url):
    ydl_opts = {
        'quiet': True,
        'skip_download': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        title = info.get('title', '')
        description = info.get('description', '')
        return title, description

# space-related keywords and filter function
SPACE_KEYWORDS = ['space', 'astronomy', 'nasa', 'types of planets', 'moon', 'mars', 'galaxy', 'satellite', 'universe', 'astrophysics'"Space science", "Astronomy", "Rocket propulsion", "Aerospace engineering",
    "Satellites", "Space exploration", "Orbital mechanics", "Rocket engine design", "venus", "stars",
    "Space missions (NASA, SpaceX, ESA, ISRO)", "Solar system and planets", "Cosmology",
    "Human spaceflight", "Mars exploration", "Moon missions", "Planetary defense",
    "Space robotics", "Zero-gravity", "Black holes", "Exoplanets", "Astrobiology", 
    "Quantum physics applied to space", "Hypersonic flight", "Space telescopes", 
    "Cryogenics in space", "ISS", "Reusable rockets", "GNC", "EDL systems"]
                    

# Check if text is space-related
def is_space_related(text):
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in SPACE_KEYWORDS)

# Step 1: Process YouTube video
if 'pipeline_done' not in st.session_state:
    st.session_state.pipeline_done = False

video_url = st.text_input("Please Paste YouTube Video URL:")

if st.button("Process Video"):
    with st.spinner("Checking video content..."):
        try:
            title, description = get_youtube_metadata(video_url)
            full_text = title + " " + description

            if not is_space_related(full_text):
                st.warning("🚫 This video does not appear to be related to space topics. Please provide a space-related video.")
            else:
                with st.spinner("Downloading and processing..."):
                    audio_file = download_youtube_audio(video_url)
                    transcript = transcribe_audio_with_whisper(audio_file)
                    chunks = split_text_into_chunks(transcript)
                    embed_and_store_chunks(chunks)
                    st.session_state.pipeline_done = True
                    st.success("✅ Video processed! You may now ask your question.")
        except Exception as e:
            st.error(f"❌ Error during checking or processing: {e}")

# Step 2: Chat interface (enabled only after video is processed)
if st.session_state.pipeline_done:
    st.header("💬 Ask about the Video")

    user_question = st.text_input("Your question:")

    if st.button("Get Answer"):
        if user_question.strip() == "":
            st.warning("Please enter a question.")
        else:
            with st.spinner("🤔 Thinking..."):
                try:
                    # Run QA chain
                    response = qa_chain.run(user_question)

                    # Query ChromaDB for context
                    results = collection.query(
                        query_texts=[user_question],
                        n_results=3
                    )
                    retrieved_docs = results['documents'][0]

                    if not retrieved_docs:
                        st.info("🤖 Hmm, this question is not related to the video content! Please ask something about the video topic.")
                    else:
                        context = "\n".join(retrieved_docs)
                        st.session_state.context = context

                        # Select system prompt based on audience
                        if audience_role == "child":
                            system_prompt = (
                                "You are a fun and educational assistant for children (ages 8–12). "
                                "Answer only if the answer is in the context provided. "
                                "If not, say: 'Hmm, this question is not related to the video content! Let's stick to the topic! 🌟'. "
                                "Use playful language and emojis. "
                                "Answer in simple, playful language with short sentences."
                            )
                        else:
                            system_prompt = (
                                "You are an academic assistant for students. "
                                "Answer strictly based on the provided context. "
                                "If the answer is not in the context, say: 'I don't know; please ask a relevant question.'. "
                                "Provide clear, informative, and detailed answers using academic language and bullet points if helpful."
                            )

                        user_prompt = f"Context:\n{context}\n\nQuestion: {user_question}"

                        response = openai_client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt}
                            ]
                        )

                        answer = response.choices[0].message.content.strip()

                         # ── Append to chat history ─────────────────────────
                        st.session_state.chat_history.append(("You", user_question))
                        st.session_state.chat_history.append(("Bot", answer))

                        #  Show answer as balloon for kids, normal box for students
                        if audience_role == "child":
                            balloon_html = f"""
                            <div style="
                                background-color: #ffb6c1;
                                color: black;
                                padding: 15px;
                                border-radius: 30px;
                                box-shadow: 2px 2px 10px rgba(0,0,0,0.2);
                                max-width: 500px;
                                margin: 10px auto;
                                font-size: 18px;
                                position: relative;
                            ">
                                🎈 {answer}
                                <div style="
                                    content: '';
                                    position: absolute;
                                    bottom: -20px;
                                    left: 50px;
                                    width: 0;
                                    height: 0;
                                    border: 10px solid transparent;
                                    border-top-color: #ffb6c1;
                                "></div>
                            </div>
                            """
                            st.markdown(balloon_html, unsafe_allow_html=True)
                        else:
                            st.success(f"🤖 {answer}")

                        # Audio playback
                        tts = gTTS(answer)
                        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
                        tts.save(tmp_file.name)
                        audio_bytes = open(tmp_file.name, 'rb').read()
                        st.audio(audio_bytes, format='audio/mp3')

                except Exception as e:
                    st.error(f"❌ Error during answering: {e}")

    # Fun Fact + DALL·E image buttons for kids || Insight button for students
    if 'context' in st.session_state and st.session_state.context:
        if audience_role == "child":
            if st.button("🎈 Give me a Fun Fact!"):
                with st.spinner("✨ Searching for fun facts..."):
                    factprompt = f"Give me one fun, surprising, and playful fact for kids (ages 8-12) based on this context:\n{st.session_state.context}"
                    response = openai_client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You provide short, playful fun facts for kids with emojis."},
                            {"role": "user", "content": factprompt}
                        ]
                    )
                    fact = response.choices[0].message.content.strip()
                    st.info(f"🚀 Fun Fact: {fact}")

            if st.button("🖼️ Generate an Image About This Video!"):
                with st.spinner("🎨 Creating a space image from the video..."):
                    try:
                        image_prompt = (
                            f"Create a fun, colorful cartoon illustration for kids aged 8-12, "
                            f"inspired by the YouTube video titled '{st.session_state.video_title}', "
                            f"which is about: {st.session_state.video_description}. "
                            f"Include rockets, astronauts, planets, or space elements as relevant."
                        )

                        dalle_response = openai_client.images.generate(
                            model="dall-e-3",
                            prompt=image_prompt,
                            size="512x512",
                            n=1
                        )
                        image_url = dalle_response.data[0].url
                        st.image(image_url, caption="🚀 Here’s a space image inspired by your video!", use_column_width=True)
                    except Exception as e:
                        st.error(f"❌ Error generating image: {e}")
        else:
            if st.button("💡 Give me an Insight!"):
                with st.spinner("🔍 Generating insights..."):
                    insight_prompt = f"Provide one deep insight or key takeaway from this context suitable for students:\n{st.session_state.context}"
                    response = openai_client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You provide deeper insights and takeaways for students."},
                            {"role": "user", "content": insight_prompt}
                        ]
                    )
                    insight = response.choices[0].message.content.strip()
                    st.success(f"🧠 Insight: {insight}")

##################################################################################################################################

                # for the layout of the streamlit 

###################################################################################################################################

st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://lightwallpapers.com/downloads/pixel-space-background/24.b1bdc1ae539dcbd1a7c33cef3e5f2d9a.gif");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    
    .css-1cpxqw2 {
        background-color: rgba(0, 0, 80, 0.8) !important;
        color: white;
        border: none;
        border-radius: 10px;
        padding: 15px;
        transition: all 0.3s ease-in-out;
        text-align: left; /* Left-align buttons */
    }
    
    .css-1cpxqw2:hover {
        background-color: rgba(50, 50, 255, 0.8) !important;
        transform: scale(1.05);
        color: white;
    }
    
    /* Custom buttons with glow effects */
    .stButton > button {
        background-color: #1e90ff;
        color: white;
        border-radius: 10px;
        transition: all 0.3s ease-in-out;
        text-align: left; /* Left-align buttons */
    }
    
    .stButton > button:hover {
        background-color: #4682b4;
        box-shadow: 0 0 20px #1e90ff;
        transform: scale(1.05);
    }

    /* Input fields for typing */
    input {
        background-color: rgba(0, 0, 80, 0.8);
        color: white;
        border-radius: 10px;
        border: none;
        padding: 10px;
    }

    /* Smooth fade-in effect for elements */
    .stApp {
        animation: fadeIn 1s ease-in-out;
    }
    
    @keyframes fadeIn {
        from {
            opacity: 0;
        }
        to {
            opacity: 1;
        }
    }
</style>
""", unsafe_allow_html=True)


# ── Chat History Sidebar ────────────────────────────────────────────────
with st.sidebar:
    st.header("💬 Conversation")
    for role, msg in st.session_state.chat_history:
        if role == "You":
            st.markdown(f"**You:** {msg}")
        else:
            st.markdown(
                f"<span style='color:#0f62fe'><strong>Bot:</strong> {msg}</span>",
                unsafe_allow_html=True
            )

             
   


