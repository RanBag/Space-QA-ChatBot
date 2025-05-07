# ✅ Step 1: Agent-integrated pipeline with fallback

from langchain.agents import initialize_agent, Tool
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory

from my_pipeline import (
    download_youtube_audio,
    transcribe_audio_with_whisper,
    split_text_into_chunks,
    embed_and_store_chunks
)

# Small memory: keep last 5 interactions
memory = ConversationBufferWindowMemory(k=5)

# Simple local cache (recent Q&A)
recent_cache = {}

# Step 2: Setup tools + agent
def youtube_tool(url: str) -> str:
    return download_youtube_audio(url)

def whisper_tool(audio_path: str) -> str:
    return transcribe_audio_with_whisper(audio_path)

def embed_tool(transcript: str) -> str:
    chunks = split_text_into_chunks(transcript)
    embed_and_store_chunks(chunks)
    return "Embedding complete"

# ✅ LLM setup (NO LangChain API keys, just LangSmith if needed)
llm = ChatOpenAI(model="gpt-3.5-turbo")

tools = [
    Tool(name="YouTubeDownloader", func=youtube_tool, description="Download YouTube audio"),
    Tool(name="WhisperTranscriber", func=whisper_tool, description="Transcribe audio to text"),
    Tool(name="Embedder", func=embed_tool, description="Embed transcript into ChromaDB")
]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="zero-shot-react-description",
    memory=memory,  # ✅ add memory here
    verbose=True
)

def run_agent_pipeline(video_url):
    try:
        print(f"▶ Running agent on: {video_url}")

        # Check if we already processed this URL
        if video_url in recent_cache:
            print("✅ Retrieved from local cache!")
            print(recent_cache[video_url])
        else:
            #  Uses invoke() → future-proof, no deprecated run()
            result = agent.invoke(f"Download, transcribe, and embed video from {video_url}")
            recent_cache[video_url] = result  # Store in cache
            print("✅ Agent pipeline completed successfully!")
            print(result)

    except Exception as e:
        print(f"⚠ Agent pipeline failed: {e}")
        print("👉 Falling back to manual pipeline...")
        try:
            audio_file = download_youtube_audio(video_url)
            transcript = transcribe_audio_with_whisper(audio_file)
            chunks = split_text_into_chunks(transcript)
            embed_and_store_chunks(chunks)
            print("✅ Manual pipeline completed successfully!")
        except Exception as fallback_error:
            print(f"❌ Manual fallback failed: {fallback_error}")

##############################################################################################
if __name__ == "__main__":
    video_url = input("🎥 Please paste the YouTube video URL: ").strip()
    run_agent_pipeline(video_url)
