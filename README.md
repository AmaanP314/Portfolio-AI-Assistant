# Portfolio AI Assistant

Developed a Retrieval-Augmented Generation (RAG) based chatbot. Its purpose is to answer questions about my professional background, projects, skills, and learning journey. The system uses a Pinecone vector database for efficient data retrieval and includes custom solutions to improve search and conversation flow.

## Table of Contents

* [Features](#features)

* [Data Collection](#data-collection)

* [Data Preprocessing](#data-preprocessing)

* [Challenges and Solutions](#challenges-and-solutions)

  * [Chunking Strategy for RAG Systems](#chunking-strategy-for-rag-systems)

  * [Search Mechanism Enhancements](#search-mechanism-enhancements)

  * [Conversational Memory Management](#conversational-memory-management)

* [Technological Stack](#technological-stack)

* [Setup and Deployment](#setup-and-deployment)

## Features

* **Comprehensive Knowledge Base:** The chatbot provides detailed information on all projects (including GitHub documentation, personal insights, and relevant social links), skill sets, certifications, educational background, and current professional activities.

* **Context-Aware Responses:** A custom RAG pipeline generates accurate and relevant answers based on retrieved documents.

* **Enhanced Search Reliability:** A custom hybrid search retriever, which includes a fallback system, ensures reliable document retrieval even when sparse encoding is not fully effective.

* **Seamless Conversational Flow:** A unique approach to conversational memory is integrated, which transforms follow-up questions into clear, self-contained queries to maintain consistent understanding.

## Data Collection

The chatbot's knowledge base is built from carefully collected and structured data:

* **Project Data:** Information for each project is gathered from GitHub documentation. This is enhanced with personal thoughts on its development, the reasons behind it, skills gained, and relevant social links.

* **Professional Background:** This section includes a full overview of my skills, certifications, education, and current studies or work.

* **Learning Journey:** A detailed record of my learning path, from when it started to my current progress, including all projects and skills learned along the way.

### Data Formatting: Markdown

All collected data is formatted in Markdown for several key reasons:

* **Structured Organization:** Markdown allows content to be clearly organized using headers (like `#` and `##`), making it easy to structure information into main topics and sub-topics.

* **Readability and Maintenance:** Markdown's simple syntax makes the content easy to read, maintain, and update.

* **Compatibility:** Many existing documentation formats, such as GitHub documentation, are already in Markdown, which simplifies the process of bringing in data.

**Example File Format (`Curl Counter`):**

```markdown
---
title: Curl Counter
id: curl-cnt
type: project
duration: Jan 2025 - Feb 2025
---

# Curl Counter

## Overview:

This project is a web application designed to track and count bicep curls using computer vision. The app utilizes MediaPipe for real-time pose estimation and OpenCV to process video frames from the user's camera. The core functionality of this app is to count repetitions of bicep curls, providing feedback based on the user's arm movements.

## Features:

-   **Real-time Bicep Curl Counting**: Automatically counts the number of curls based on arm movements.
-   **Dual Arm Support**: Tracks both left and right arms simultaneously.
-   **Stage Feedback**: Detects and displays the current state of the curl, whether the arm is "Relaxed" or "Flexed."
-   **MediaPipe Pose Estimation**: Leverages Google's MediaPipe library for real-time, accurate pose detection.
-   **User-Friendly**: Simple and intuitive UI for starting the camera and viewing the curl count.

## Usage:

-   This tool is ideal for tracking exercise form and counting reps for workouts.
-   Can be used by fitness enthusiasts, trainers, or developers interested in understanding computer vision applications.
-   Demonstrates real-world use cases of OpenCV and MediaPipe for pose detection.
````

## Data Preprocessing

Before adding data to the Pinecone vector database, several preprocessing steps are applied to prepare the content for embedding and retrieval. The main goal is to convert the structured Markdown into plain text, as embedding models generally work better with plain text.

Here is the code pipeline used for preprocessing:

```python
# Function to load markdown files with frontmatter
def load_markdown(file_path):
    post = frontmatter.load(file_path)
    meta = post.metadata
    body = post.content
    return meta, body

# Find markdown files in the current directory
directory_path = os.getcwd()
markdown_files = [f for f in os.listdir(directory_path) if f.endswith('.md')]

chunks = []
ids = []
texts = []
metadatas = []

# Define headers for splitting markdown documents
headers_to_split_on = [("#", "Header 1"), ("##", "Header 2")]
header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

# Variable to store the previous chunk's last sentence for overlap
previous_chunk_text = ""
for file_path in markdown_files:
    meta, body = load_markdown(file_path)
    # Split the markdown body into chunks based on headers
    chunk = header_splitter.split_text(body)
    for i, c in enumerate(chunk):
        chunk_id = f"{meta.get('id')}-{i+1}"

        # Add headers into content before embedding to preserve context
        header_context = ""
        if 'Header 1' in c.metadata:
            header_context += f"{c.metadata['Header 1']}\n"
        if 'Header 2' in c.metadata:
            header_context += f"{c.metadata['Header 2']}\n"

        full_text = header_context + "\n" + c.page_content
        if previous_chunk_text:
            overlap_content = previous_chunk_text 
            full_text = overlap_content + "\n" + full_text
        # Store the current chunk text and metadata for the next chunk
        previous_chunk_text = c.page_content.split("\n")[-1]

        chunks.append(c)
        texts.append(full_text)
        metadatas.append({**meta, **c.metadata})
        ids.append(chunk_id)

# Function to convert markdown string to plain text
def markdown_to_text(markdown_str: str) -> str:
    # 1) Convert Markdown to HTML
    html = md.markdown(markdown_str)
    # 2) Remove HTML tags to get plain text
    return BeautifulSoup(html, "html.parser").get_text(separator=" ")

# Convert all chunk contents to plain text for optimal embedding
plain_texts = [markdown_to_text(text) + ' \n\n' for text in texts]
```

## Problems Faced and Solutions

During the development of this RAG-based chatbot, several challenges were encountered related to data processing, search effectiveness, and maintaining conversation flow. Each problem was addressed with a specific solution to improve the chatbot's performance and reliability.

### Chunking Strategy for RAG Systems

In Retrieval-Augmented Generation (RAG) systems, **chunking** is the process of dividing raw documents or text into smaller parts (chunks). These chunks are then embedded and stored in a vector database for similarity search. The way chunking is done significantly impacts the **quality of retrieval, how well context is kept**, and the **accuracy of answers** from the system.

This section explains:

  * The **common chunking method** used in RAG pipelines.
  * My **custom chunking method** that uses semantic structure and contextual overlap.
  * A **direct comparison** of both methods.
  * The **limitations of common chunking** and how my method addresses them.

-----

#### 📦 Common Chunking Method

##### 🔹 Key Characteristics:

  * **Fixed-size chunks** (efor example, 500 or 1000 tokens).
  * **Standard overlap** between chunks (e.g., 200 tokens).
  * **No understanding of content structure** (paragraphs, headings, sections, etc.).
  * Content is split **only based on token length**, without considering its meaning.

##### ⚠️ Weaknesses:

| Problem | Description |
| :------------------------ | :----------------------------------------------------------------------------------------------------------------- |
| ❌ Semantic Breaks | Chunks often split mid-thought or mid-section, losing meaning. |
| ❌ Disconnected Context | Related paragraphs or sections may be isolated, reducing coherence. |
| ❌ Mechanical Overlap | Overlap includes token sequences, not contextual bridge content. |
| ❌ Low Retrieval Precision | Retrieved chunks might be too general or lack enough context to answer the query effectively. |
| ❌ No Structure Awareness | Fails to distinguish between sections like "Overview," "Features," etc., leading to flat, unstructured chunks. |

-----

#### 🧠 Semantic Chunking with Contextual Overlap

I developed a custom chunking method specifically for my personal documentation (projects, learning journey, skills, etc.). In this method, I **manually structure the content** using a hierarchical Markdown format, and then apply smart chunking based on meaningful sections and continuous context.

##### 🔹 Key Characteristics:

  * 📘 **Markdown-Based Structure**: Content uses headers (`#`, `##`) to show the document's hierarchy (e.g., Project → Overview → Features).
  * 🧩 **Header-Aware Chunking**: Chunks are split at the `Header 2` level and include their parent `Header 1` to keep context.
  * 🔄 **Contextual Overlap**: Each chunk includes the last sentence or key content from the previous chunk to maintain continuity between sections.
  * 🧼 **Markdown to Plain Text Conversion**: Before embedding, Markdown formatting is removed for cleaner and more effective embeddings.
  * 🧾 **Rich Metadata**: Each chunk carries structured metadata (e.g., `title`, `type`, `duration`, `skills used`) to support smarter retrieval and filtering.
  * ✅ **Controlled Chunk Size**: I ensure that each section remains under a practical token limit (e.g., ~1000 tokens), which avoids long or overflowing chunks, reducing the need for mid-paragraph splitting.

-----

#### 🔍 Side-by-Side Comparison

| Feature | Common Chunking | My Semantic + Overlap Chunking |
| :------------------------------ | :-------------------------------- | :------------------------------------- |
| **Chunk Boundaries** | Fixed token count | Semantic (section-aware |
| **Overlap Logic** | Fixed token-based | Context-aware (last sentence or topic) |
| **Context Preservation** | ❌ Often broken | ✅ Strong due to overlap and structure |
| **Metadata Support** | ⚠️ Usually minimal | ✅ Rich (id, title, type, skills, etc.) |
| **Content Structure Awareness** | ❌ None | ✅ Hierarchical (project → section) |
| **Search Relevance** | ⚠️ Moderate | ✅ High |
| **Redundancy / Noise** | High risk with fixed overlap | Minimal, focused overlap |
| **Flexibility & Scalability** | Requires re-chunking | Modular and scalable |
| **Retrieval Relevance** | Inconsistent | Reliable and coherent |

-----

#### ✅ Why this Approach Is Better

##### 🔧 Solves Key Issues in Common Chunking:

| Issue in Common Chunking | How My Approach Fixes It |
| :-------------------------------- | :--------------------------------- |
| **Context gets split** | Chunks align with logical sections |
| **No awareness of document flow** | Headers are kept in chunks |
| **Fixed overlap** | Smart overlap using the last sentence |
| **Poor metadata support** | Metadata included with each chunk |
| **Flat structure** | Shows the true hierarchy of content |

##### 📈 Benefits in Practice:

  * Better search results for **project-specific questions** (e.g., “What technologies did you use in the curl counter project?”).
  * More natural and complete answers because of **coherent chunks**.
  * Improved performance for **chat-based interactions**, especially when context needs to be carried across turns.

-----

### Search Mechanism Enhancements

**Problem:** The `PineconeHybridSearchRetriever` sometimes failed or returned no relevant documents if a query's exact words did not match any chunk during the sparse encoding phase. This led to unreliable search results, particularly for queries that were conceptually related but lacked direct keyword matches.

**Solution:** A `CustomHybridSearchRetriever` was implemented, which extends the standard `PineconeHybridSearchRetriever`. This custom class includes a fallback system: if the initial hybrid search (which combines dense and sparse vectors) encounters an error related to sparse encoding (for example, "Sparse vector must contain at least one value"), it automatically switches to performing a dense-only search. This design ensures that even if a query does not have strong keyword matches, the system can still retrieve relevant documents based on semantic similarity.

**Code Snippet (from `chatbot.py`):**

```python
class CustomHybridSearchRetriever(PineconeHybridSearchRetriever):
    def _get_relevant_documents(
        self, query: str, *, run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        """Get documents relevant to the query using hybrid search with fallback to dense-only."""
        try:
            # Attempt hybrid search initially
            return super()._get_relevant_documents(query, run_manager=run_manager)
        except Exception as e:
            # If sparse encoding fails, switch to dense-only search
            if "Sparse vector must contain at least one value" in str(e):
                print("Falling back to dense-only search for query:", query)
                # Generate dense embeddings
                embedding = self.embeddings.embed_query(query)
                # Perform search using only dense vectors
                results = self.index.query(
                    vector=embedding,
                    top_k=self.top_k,
                    include_metadata=True,
                    namespace=self.namespace,
                )
                # Convert Pinecone results into LangChain Document objects
                return self._process_pinecone_results(results)
            else:
                # Re-raise the exception if it is a different error
                raise e
    
    def _process_pinecone_results(self, results):
        """Process Pinecone results into Document objects."""
        docs = []
        for result in results.matches:
            metadata = result.metadata or {}
            # Create Document with page content and metadata
            doc = Document(
                page_content=metadata.pop("text", ""),
                metadata=metadata,
            )
            docs.append(doc)
        return docs
    
retriever = CustomHybridSearchRetriever( 
        embeddings=embeddings, 
        sparse_encoder=bm25_encoder, 
        index=index,
        top_k=3
    )
```

### Conversational Memory Management

**Problem:** In a RAG system, follow-up questions often lack clear keywords from previous parts of the conversation. This can make it hard for the retrieval system to find relevant documents. For example, if a user asks "Tell me about the Tweets Scraper project," and then asks "Can you give me the link for this project?", the second question might not retrieve the project's link because "Tweets Scraper" is not mentioned. Standard conversational memory (just passing the chat history) does not always solve this retrieval problem effectively for RAG.

**Solution:** To handle this, a method was implemented that converts follow-up questions into standalone queries using a Large Language Model (LLM). This LLM, which understands the full conversation context, rephrases the user's question to be self-contained and suitable for retrieval.

**Example Flow:**

1.  **User Question 1:** "Tell me about the Tweets Scraper project."
2.  **Chatbot Response:** The chatbot finds and presents information about the Tweets Scraper project.
3.  **User Question 2 (Follow-up):** "Okay, now give me the link for this project."
4.  **LLM's Role:** The LLM receives "Okay, now give me the link for this project" along with the previous conversation. It then creates a new, independent query like "link for tweets scraper project."
5.  **Retrieval:** This new, clear query ("link for tweets scraper project") is then used for the search, ensuring the correct project link is found.


## Technological Stack

The chatbot is built using the following key technologies and libraries:

  * **Backend Framework:** FastAPI

  * **Vector Database:** Pinecone

  * **Large Language Model (LLM) Integration:** LangChain, Google Generative AI Embeddings, ChatGoogleGenerativeAI

  * **Text Processing:** `pinecone-text`, `langchain-text_splitters`, `markdown`, `BeautifulSoup`, `frontmatter`

  * **Environment Management:** `python-dotenv`

  * **Server:** `uvicorn`

## Setup and Deployment

To set up and run this project locally, follow these steps:

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/your-username/portfolio-ai-assistant.git
    cd portfolio-ai-assistant
    ```

2.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

    (Note: The `requirements.txt` file is based on the provided context.)

3.  **Configure Environment Variables:**
    Create a `.env` file in the root directory and add your API keys and model names:

    ```
    GOOGLE_API_KEY="your_google_api_key"
    PINECONE_API_KEY="your_pinecone_api_key"
    EMBEDDING_MODEL="models/embedding-001" # or your chosen embedding model
    LLM_MODEL="gemini-2.0-flash-lite" # or your chosen LLM model
    RETRIEVER_PROMPT="YOUR RETRIEVER PROMPT"
    SYSTEM_PROMPT="YOUR SYSTEM PROMPT"
    ```

4.  **Run the FastAPI Application:**

    ```bash
    uvicorn app:app --reload
    ```

    This command will start the API server, usually accessible at `http://127.0.0.1:8000`.

## Usage

The chatbot provides a `/chat` endpoint for interaction.

**Endpoint:** `POST /chat`

**Request Body:**

```json
{
    "message": "Your query message",
    "session_id": "A unique session identifier"
}
```

**Example:**

```python
import requests

url = "[http://127.0.0.1:8000/chat](http://127.0.0.1:8000/chat)"
payload = {
    "message": "Tell me about your YouTube Video Recommender project.",
    "session_id": "user_session_123"
}

response = requests.post(url, json=payload)
print(response.json())
```
