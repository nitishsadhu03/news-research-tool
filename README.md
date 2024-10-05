# News Research Tool 📈

This project utilizes LangChain and the Gemini API to create a tool for researching news articles. Users can input URLs of news articles, and the tool will process and extract information to answer user queries based on the content of the articles.

## Overview

Users can input up to three URLs of news articles. The tool processes these articles, generates embeddings using the Gemini API, and stores them in a FAISS index. Users can then input questions, and the tool will retrieve and provide answers based on the content of the articles.

## Features

- **URL Input**: Users can input up to three URLs of news articles.
- **Data Processing**: The tool processes and splits the text from the articles.
- **Embeddings Generation**: Generates embeddings using the Gemini API and stores them in a FAISS index.
- **Question Answering**: Users can ask questions, and the tool retrieves answers based on the processed articles.

## Setup Instructions

Follow the steps below to set up and run the News Research Tool project.

### Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/nitishsadhu03/learning-generative-ai-with-projects.git
    cd news-research-tool
    ```

2. **Set up the virtual environment:**

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Configure environment variables:**

    Create a `.env` file in the `news-research-tool` directory and add your Google API key and LangChain API key:

    ```env
    GOOGLE_API_KEY=your_google_api_key_here
    LANGCHAIN_API_KEY=your_langchain_api_key_here
    ```

### Running the Application

1. **Start the Streamlit app:**

    ```bash
    streamlit run main.py
    ```

2. **Open your browser** and navigate to `http://localhost:8501` to use the application.

## Project Structure

- `main.py`: The main script for running the application.
- `requirements.txt`: Lists the dependencies required to run the project.
- `.env`: Environment variables file (not included in the repository).
- `venv/`: Virtual environment directory (not included in the repository).
