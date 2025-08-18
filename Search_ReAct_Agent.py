import json
import spacy
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langchain.agents import initialize_agent, AgentType
from Search_Tools import wikipedia_search, wikimedia_commons_search, google_dataset_search, github_search, neo4j_search
from Neo4j_setup import insert_structured_data_to_neo4j, driver
from langchain.memory import ConversationBufferWindowMemory
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer

load_dotenv()  # Load environment variables from .env file
# Initialize LLM (set your OPENAI_API_KEY environment variable)

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, max_tokens=1500)
# Load embedding model once at startup (MiniLM is fast + accurate)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
kw_model = KeyBERT(model=embedding_model)

def get_search_agent():
    # Initialize your LLM (OpenAI GPT-4 here)
    memory = ConversationBufferWindowMemory(
        k=5,                  # keep only last 5 interactions            
        memory_key="chat_history",
        return_messages=True  # return messages instead of concatenated string
    )
    agent = initialize_agent(
        tools=[
            wikipedia_search,
            wikimedia_commons_search,
            google_dataset_search,
            github_search,
        ],
        llm=llm,
        agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,  # <-- NOT ZERO_SHOT_REACT_DESCRIPTION!
        handle_parsing_errors=True,
        verbose=True,
        memory=memory,
        max_iterations=3,  # Limit iterations to avoid infinite loops
    )

    return agent

def chunk_text(text, max_len=1500):
    """Split text into chunks of max_len characters."""
    return [text[i:i+max_len] for i in range(0, len(text), max_len)]

def summarize_chunks(chunks, llm):
    summaries = []
    for chunk in chunks:
        system_msg = SystemMessage(content="Summarize this text.")
        human_msg = HumanMessage(content=chunk)
        response = llm.predict_messages([system_msg, human_msg])
        summaries.append(response.content.strip())
    combined_summary = " ".join(summaries)
    return combined_summary

def parse_output_with_llm(text):
    chunks = chunk_text(text, max_len=1500)
    summarized_text = summarize_chunks(chunks, llm)

    system_msg = SystemMessage(
        content=(
            "You are a parser that extracts structured information from text about "
            "concepts, GitHub repos, datasets, Wikipedia summaries, and Neo4j search results. "
            "Only respond with valid JSON containing keys: "
            "`concept` (name, description), "
            "`repositories` (list of {name, url, description}), "
            "`datasets` (list of {name, url, description}), "
            "`wikipedia` (title, summary, url)."
        )
    )
    human_msg = HumanMessage(content=f"Extract structured data from this text:\n'''{summarized_text}'''")

    response_text = llm.predict_messages([system_msg, human_msg])
    try:
        parsed = json.loads(response_text.content)
        return parsed
    except json.JSONDecodeError as e:
        print("Failed to parse JSON from LLM response:", e)
        print("LLM response was:", response_text)
        return None
    
# def extract_search_phrase_from_query(user_query: str) -> str:

#     system_msg = SystemMessage(
#         content=(f"You are an AI assistant that extracts the core subject or concept from a user's query. "
#                 f"Your job is to return only the key topic the query is about, such as a technical term, library name, or concept."
#                 f"Return a **single phrase** in its **singular canonical form** only."
#         )
#     )
#     human_msg = HumanMessage(
#         content=f"Extract the main search topic from this user query:\n'''{user_query}'''"
#     )


#     response = llm.predict_messages([system_msg, human_msg])
#     # The response is the output text containing the search phrase
#     search_phrase = response.content.strip().strip('"\'')  # remove quotes if present
#     return search_phrase/


def extract_keywords_keybert(query: str, top_n: int = 5, ngram_range=(1, 3)):
    """
    Extracts top keywords/keyphrases from a user query using KeyBERT.

    Args:
        query (str): The user query text.
        top_n (int): Number of keywords/phrases to return (default 5).
        ngram_range (tuple): Size of keyphrases, e.g. (1,3) for unigrams, bigrams, trigrams.

    Returns:
        List[str]: Ranked list of keywords/phrases.
    """
    keywords = kw_model.extract_keywords(
        query,
        keyphrase_ngram_range=ngram_range,  # allow unigrams, bigrams, trigrams
        stop_words="english",
        top_n=top_n,
    )
    # Extract only the keyword text (drop scores)
    return [kw for kw, score in keywords]

    