from dotenv import load_dotenv
load_dotenv()

import os
from neo4j import GraphDatabase

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def insert_structured_data_to_neo4j(data, driver):
    if data is None:
        print("No structured data to insert.")
        return

    with driver.session() as session:
        concept = data.get("concept")
        if concept:
            desc = concept.get("description", "")
            # Filter for known agent/system messages
            system_phrases = [
                "The agent was stopped",
                "iteration limit",
                "time limit"
            ]
            if any(phrase in desc for phrase in system_phrases):
                print(f"Skipping system/noise concept: {desc}")
            else:
                session.run(
                    """
                    MERGE (c:Concept {name: $name})
                    SET c.description = $description
                    """,
                    name=concept.get("name", "Unknown Concept"),
                    description=concept.get("description", ""),
                )

        wikipedia_data = data.get("wikipedia")
        if wikipedia_data:
            title = wikipedia_data.get("title")
            if title:  # Only insert if title is not None and not empty
                session.run(
                    """
                    MERGE (w:WikipediaArticle {title: $title})
                    SET w.summary = $summary, w.url = $url
                    MERGE (c:Concept {name: $concept_name})
                    MERGE (c)-[:HAS_WIKIPEDIA_SUMMARY]->(w)
                    """,
                    title=title,
                    summary=wikipedia_data.get("summary", ""),
                    url=wikipedia_data.get("url", ""),
                    concept_name=concept.get("name", "Unknown Concept"),
                )
            else:
                print("Skipped WikipediaArticle insert: missing or empty 'title'.")

        for repo in data.get("repositories", []):
            session.run(
                """
                MERGE (r:Repository {name: $repo_name})
                SET r.url = $url, r.description = $desc
                MERGE (c:Concept {name: $concept_name})
                MERGE (c)-[:HAS_IMPLEMENTATION]->(r)
                """,
                repo_name=repo.get("name", ""),
                url=repo.get("url", ""),
                desc=repo.get("description", ""),
                concept_name=concept.get("name", "Unknown Concept"),
            )

        for ds in data.get("datasets", []):
            session.run(
                """
                MERGE (d:Dataset {name: $ds_name})
                SET d.url = $url, d.description = $desc
                MERGE (c:Concept {name: $concept_name})
                MERGE (c)-[:HAS_DATASET]->(d)
                """,
                ds_name=ds.get("name", ""),
                url=ds.get("url", ""),
                desc=ds.get("description", ""),
                concept_name=concept.get("name", "Unknown Concept"),
            )

    print("Data inserted into Neo4j.")

