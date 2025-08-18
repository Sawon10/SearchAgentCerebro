from langchain_core.tools import tool
import wikipedia
import requests
from googleapiclient.discovery import build
from github import Github
import os
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()


@tool("neo4j_search", description="Run a Cypher query against the Neo4j database and return the results.")
def neo4j_search(cypher_query: str, params: dict = None):
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USER")
    password = os.getenv("NEO4J_PASSWORD")
    if not uri or not user or not password:
        return "Neo4j connection environment variables not set (.env: NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)"
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        results = []
        with driver.session() as session:
            query_result = session.run(cypher_query, params or {})
            for record in query_result:
                results.append(record.data())
        driver.close()
        return results if results else "No results found."
    except Exception as e:
        return f"Neo4j search error: {e}"


@tool("wikipedia_search", description="Search Wikipedia and return a short summary and links.")
def wikipedia_search(query: str):
    wikipedia.set_lang("en")
    results = []
    try:
        titles = wikipedia.search(query, results=1)
        for title in titles:
            summary = wikipedia.summary(title, sentences=2)
            page_url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
            results.append({
                "source": "wikipedia",
                "summary": summary,
                "links": [page_url],
                "preview": None
            })
    except Exception as e:
        return f"Wikipedia search error: {e}"
    
    return results


### 2. Wikimedia Commons Search Tool ###

@tool("wikimedia_commons_search", description="Search Wikimedia Commons for relevant media files with links and previews.")
def wikimedia_commons_search(query: str):
    endpoint = "https://commons.wikimedia.org/w/api.php"
    params = {
        "action": "query",
        "generator": "search",
        "gsrsearch": query,
        "gsrlimit": 3,
        "prop": "imageinfo",
        "iiprop": "url|extmetadata",
        "format": "json"
    }
    results = []
    try:
        resp = requests.get(endpoint, params=params).json()
        pages = resp.get("query", {}).get("pages", {})
        for page in pages.values():
            imageinfo = page.get("imageinfo", [{}])[0]
            url = imageinfo.get("url")
            metadata = imageinfo.get("extmetadata", {})
            description = metadata.get("ImageDescription", {}).get("value", "")
            description = description or page.get("title", "")
            if url:
                results.append({
                    "source": "wikimedia_commons",
                    "summary": description,
                    "links": [url],
                    "preview": url
                })
    except Exception as e:
        return f"Wikimedia Commons search error: {e}"

    return results


### 3. Google Dataset Search Tool ###

@tool("google_dataset_search", description="Search Google Dataset Search and return dataset summaries with links.")
def google_dataset_search(query: str):
    api_key = os.getenv("GOOGLE_API_KEY")
    cse_id = os.getenv("GOOGLE_CSE_ID_DATASET")
    if not api_key or not cse_id:
        return "Google Dataset Search API key or CSE ID not configured."

    results = []
    try:
        service = build("customsearch", "v1", developerKey=api_key)
        res = service.cse().list(q=query, cx=cse_id, num=3).execute()
        for item in res.get("items", []):
            title = item.get("title", "")
            snippet = item.get("snippet", "")
            link = item.get("link")
            results.append({
                "source": "google_dataset",
                "summary": snippet or title,
                "links": [link] if link else [],
                "preview": None
            })
    except Exception as e:
        return f"Google Dataset search error: {e}"

    return results


### 4. GitHub Search Tool ###

@tool("github_search", description="Search GitHub public repositories and return summaries and links.")
def github_search(query: str):
    token = os.getenv("GITHUB_TOKEN")
    gh = Github(token) if token else Github()
    results = []
    try:
        repos = gh.search_repositories(query=query, sort="stars")
        for repo in repos[:3]:
            results.append({
                "source": "github",
                "summary": repo.description or repo.full_name,
                "links": [repo.html_url],
                "preview": None
            })
    except Exception as e:
        return f"GitHub search error: {e}"

    return results
