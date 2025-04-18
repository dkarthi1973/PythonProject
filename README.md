# Advanced Document Analyzer

A Streamlit web application for extracting text from documents, performing analysis, generating AI-powered summaries, and answering questions.

## Features

-   **Multi-format Support:**  Supports PDF, DOCX, and TXT file uploads.
-   **Text Extraction and Splitting:** Extracts text from uploaded files and splits it into manageable paragraphs or chunks.
-   **Semantic Search:** Implements TF-IDF vectorization and cosine similarity to enable searching within the document.
-   **AI Summarization:** Generates summaries using the Ollama API.
-   **Question Answering:** Answers user queries based on document content using the Ollama API.
-   **Text-to-Speech:** Converts text to speech using gTTS.
-   **Document History:** Keeps track of recently uploaded documents.
-   **Document Statistics:** Calculates and displays various document statistics.
-   **Document Comparison:** Compares two documents for similarity.
-   **Export Features:** Allows exporting activity logs as CSV and analysis results as ZIP archives.
-   **Customizable Settings:** Includes configurable options for summary length, search result highlighting, dark mode, and more.
-   **Robust Logging:** Logs user actions and any errors encountered.

## Installation

1.  **Clone the repository:**

    ```
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **Create a virtual environment (recommended):**

    ```
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate  # On Windows
    ```

3.  **Install the required packages:**

    ```
    pip install -r requirements.txt
    ```

4.  **Set up Ollama:**
    *   Ensure that you have [Ollama](https://ollama.com/) installed and running.
    *   Verify that the Ollama API is accessible at `http://localhost:11434`.

## Usage

1.  **Run the Streamlit app:**

    ```
    streamlit run docsummarizer.py
    ```

2.  **Upload Documents:**
    *   Use the file uploader to upload your PDF, DOCX, or TXT files.

3.  **Explore Features:**
    *   View extracted text.
    *   Search within the document.
    *   Generate summaries and ask questions using the AI model.
    *   Listen to the text using the text-to-speech functionality.
    *   Compare documents if desired.
    *   View and export activity logs.

## Configuration

Adjust the following settings in the sidebar:

-   **Default Model:** Select the default AI model to use.
-   **Summary Length:** Choose between short, medium, and long summaries.
-   **Search Results Count:** Set the number of search results to display.
-   **Highlight Enabled:** Toggle search term highlighting.
-   **Save Logs:** Enable or disable saving logs to a file.
-   **Dark Mode:** Switch between light and dark themes.

## Logging

*   The application logs various activities and errors.  Logs are stored in `document_analyzer.log`.
*   You can disable logging via the sidebar settings.
*   Export activity logs to CSV format for analysis.

## Dependencies

*   streamlit
*   pandas
*   PyPDF2
*   python-docx
*   sklearn
*   gTTS
*   requests
*   plotly
*   numpy

## Troubleshooting

*   **Ollama API Connection Issues:** Ensure Ollama is running and accessible at `http://localhost:11434`.
*   **File Extraction Errors:**  Check the file format and ensure it's a valid PDF, DOCX, or TXT file.

## License

This project is licensed under the [MIT License](LICENSE).
