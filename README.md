# Space-QA-ChatBot
Space QA Bot is a multimodal educational chatbot that allows children and students to ask questions about space content from YouTube videos. It uses audio and text modalities to deliver answers in a way that’s both engaging and educational. Built with LangChain, Whisper, and GPT, this bot enhances accessibility, learning, and curiosity

# Space QA Bot

**Multimodal educational chatbot for space content**

---

## 🚀 Project Overview

**Space QA Bot** is a multimodal educational chatbot that allows children and students to ask questions about space content extracted from YouTube videos. Leveraging both audio and text inputs, the bot delivers answers in an engaging, age-appropriate manner to spark curiosity and deepen understanding of the universe.

Key technologies:

* **LangChain** for orchestration of LLM workflows
* **Whisper** for accurate audio transcription
* **OpenAI’s GPT** models for natural language understanding and response generation
* **Streamlit** for an interactive web interface

---

## 🌟 Features

* **Multimodal input**: Ask questions via microphone or text box
* **Dynamic audience adaptation**:

  * **Child**: Simplified explanations with friendly tone and emojis
  * **Student**: Detailed academic-style responses with bullet points
* **YouTube video ingestion**: Fetch, transcribe, and chunk video content automatically
* **Contextual retrieval**: Embedding-based search to find relevant video segments
* **Real-time Q\&A**: Instant answer generation with sources cited
* **Accessibility**: Audio playback of responses and adjustable text size

---

## 🔧 Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/your-org/space-qa-bot.git
   cd space-qa-bot
   ```

2. **Create and activate a virtual environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   Create a `.env` file in the project root:

   ```dotenv
   OPENAI_API_KEY=your_openai_api_key
   YOUTUBE_API_KEY=your_youtube_api_key
   ```

---

## ⚙️ Pipeline & Architecture

1. **YouTube Ingestion**:

   * Provide a YouTube URL or search term
   * Download video metadata and audio stream

2. **Audio Transcription**:

   * Whisper transcribes audio into text
   * Chunk transcripts for embedding

3. **Embedding & Indexing**:

   * Generate vector embeddings for each transcript chunk
   * Store in a vector store (e.g., ChromaDB)

4. **Query Processing**:

   * User submits a question via text or microphone
   * Transcribe spoken question (if audio)
   * Embed question and perform similarity search against transcript chunks

5. **Response Generation**:

   * Retrieve top-k relevant chunks
   * LangChain constructs prompt with audience context
   * GPT generates the answer, citing video timestamps

6. **Presentation Layer**:

   * Streamlit UI adapts response style based on audience
   * Option to play audio response using TTS

---

## 🏁 Usage

Run the app locally:

```bash
streamlit run app.py
```

1. Open [http://localhost:8501](http://localhost:8501) in your browser
2. Choose the audience type (Child or Student)
3. Enter a YouTube URL or search term
4. Ask your question via text or click the mic icon to speak
5. Receive an interactive answer with text and optional audio playback

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m "Add YourFeature"`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Open a Pull Request

Please ensure code quality and update tests where applicable.


---

## 📬 Contact

For questions and support, open an issue or reach out to the maintainer:

Enjoy exploring the universe with Space QA Bot! 🌌




https://drive.google.com/file/d/1tGogEpq0S2sO-86lbF67F6_ate3xW0OB/view?usp=sharing
