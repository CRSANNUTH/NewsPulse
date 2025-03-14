
# NewsPulse : News Research Tool 

NewsPulse is a user-friendly news research tool designed for effortless information retrieval. Users can input article URLs and ask questions to receive relevant insights from the stock market and financial domain.

Features
Load News Article URLs or Upload Text Files:

Users can easily input multiple URLs or upload a text file containing URLs directly into the application. This allows for seamless fetching of article content from various online sources.
Content Processing with LangChain's Unstructured URL Loader:

The app uses LangChain’s Unstructured URL Loader to efficiently extract and process the content of the provided news articles. This tool ensures that the raw content is cleaned and structured for better analysis and processing.
Embedding Vector Construction Using Gemini API:

The app employs the Gemini API to generate embeddings from the processed content. These embeddings are vector representations of the articles that capture the semantic meaning of the text, making it easier to compare and retrieve relevant information.
Fast and Efficient Similarity Search with FAISS:

To enable quick and efficient retrieval of relevant information, FAISS (Facebook AI Similarity Search) is used to index the embeddings. FAISS helps find the most relevant content based on similarity searches, ensuring that users receive the most pertinent responses to their queries.
Query Answering and Insights Retrieval via Gemini API:

After indexing the articles, users can interact with the application by asking questions related to the news articles. The application sends these queries to the Gemini API, which generates responses based on the content of the indexed articles.
Source URL Tracking:

For each answer generated by the Gemini API, the system also provides the source URLs from which the information was derived. This ensures transparency and allows users to trace back the data to its original source.
Pre-processing and Embedding Storage:

The embeddings are stored and indexed using FAISS to significantly speed up the retrieval process. The FAISS index is stored as a pickle file locally, allowing for persistent storage and re-use of the index across multiple sessions.
Efficient User Interaction:

The user interface is built with Streamlit, making it interactive and easy to use. Users simply input URLs, click on a button to process them, and then start querying the system to get answers based on the processed news articles.
Customizable API Key Support:

The application uses an .env file for storing sensitive information like the Gemini API key. This makes it secure and customizable so that users can input their own API keys to interact with the Gemini service.
Real-Time Updates:

Once the URLs are processed, the FAISS index is updated with the latest embeddings, enabling real-time querying of new content. Users can continue to process new URLs and ask questions about them without restarting the application.

Here are the technologies used in your NewsPulse application:

Streamlit,
Gemini API,
LangChain,
FAISS,
Python,
Pickle,
Dotenv,
GitHub,


