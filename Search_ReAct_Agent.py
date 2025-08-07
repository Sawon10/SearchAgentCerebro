import os
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langchain.agents import initialize_agent, AgentType
from Search_Tools import wikipedia_search, wikimedia_commons_search, google_dataset_search, github_search, neo4j_search
from Neo4j_setup import insert_structured_data_to_neo4j, driver

load_dotenv()  # Load environment variables from .env file
# Initialize LLM (set your OPENAI_API_KEY environment variable)

llm = ChatOpenAI(model_name="gpt-4", temperature=0, max_tokens=1500)

def get_search_agent():
    # Initialize your LLM (OpenAI GPT-4 here)
    agent = initialize_agent(
        tools=[
            wikipedia_search,
            wikimedia_commons_search,
            google_dataset_search,
            github_search,
            neo4j_search
        ],
        llm=llm,
        agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,  # <-- NOT ZERO_SHOT_REACT_DESCRIPTION!
        verbose=True,
    )

    return agent

def parse_output_with_llm(text):
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
    human_msg = HumanMessage(content=f"Extract structured data from this text:\n'''{text}'''")

    response_text = llm.predict_messages([system_msg, human_msg])
    try:
        parsed = json.loads(response_text.content)
        return parsed
    except json.JSONDecodeError as e:
        print("Failed to parse JSON from LLM response:", e)
        print("LLM response was:", response_text)
        return None
    
def extract_search_phrase_from_query(user_query: str) -> str:
    """
    Uses LLM to extract keywords or main concept(s) from a user query.

    The prompt instructs the LLM to return a short phrase or keyword to search in Neo4j.
    """
    system_msg = SystemMessage(
        content=(f"You are an AI assistant that extracts the core subject or concept from a user's query. "
                f"Your job is to return only the key topic the query is about, such as a technical term, library name, or concept."
                F"Return a **single phrase** in its **singular canonical form** only."
)
    )
    human_msg = HumanMessage(
        content=f"Extract the main search topic from this user query:\n'''{user_query}'''"
    )


    response = llm.predict_messages([system_msg, human_msg])
    # The response is the output text containing the search phrase
    search_phrase = response.content.strip().strip('"\'')  # remove quotes if present
    return search_phrase

    