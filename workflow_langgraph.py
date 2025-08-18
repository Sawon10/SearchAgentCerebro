from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI

from Search_Tools import neo4j_search
from Search_ReAct_Agent import get_search_agent, parse_output_with_llm, extract_keywords_keybert
from Neo4j_setup import insert_structured_data_to_neo4j, driver


########################################
# 2. Node function definitions
########################################

def neo4j_search_node(state):
    # Get the user query
    query = state["user_query"]

    # 🔑 Extract multiple keywords/phrases (semantic)
    keywords = extract_keywords_keybert(query, top_n=5)

    cypher_query = """
        MATCH (c:Concept)
        WITH c,
             [kw IN $keywords WHERE toLower(c.name) CONTAINS toLower(kw)] AS matched
        WHERE size(matched) > 0
        OPTIONAL MATCH (c)-[r]->(x)
        RETURN c, matched, size(matched) AS score, collect(r) AS rels, collect(x) AS linked
        ORDER BY score DESC
        LIMIT 20
    """

    results = neo4j_search(cypher_query, params={"keywords": [kw.lower() for kw in keywords]})
    formatted = []
    if isinstance(results, list):
        for rec in results:
            concept = rec.get("c", {})
            formatted.append({
                "concept_name": concept.get("name", ""),
                "description": concept.get("description", ""),
                "matched_keywords": rec.get("matched", []),
                "score": rec.get("score", 0),
                "relations": [str(r) for r in rec.get("rels", [])],
                "linked_nodes": [str(x) for x in rec.get("linked", [])],
            })
        state["neo4j_results"] = formatted
        state["neo4j_hit"] = bool(formatted)
    else:
        state["neo4j_results"] = []
        state["neo4j_hit"] = False
        state["error"] = f"Neo4j search error or no results: {results}"
    return state

def neo4j_check_node(state):
    if state.get("neo4j_hit") and state.get("neo4j_results"):
        state["use_neo4j"] = True
        display_str = "Found results in Neo4j:\n"
        display_str += "\n".join([str(record) for record in state["neo4j_results"]])
        state["final_answer"] = display_str
    else:
        state["use_neo4j"] = False
    return state

def agent_search_node(state):
    if not state.get("use_neo4j"):
        agent = get_search_agent()
        result = safe_invoke(agent, {"input": state["user_query"]})
        if isinstance(result, dict):
            state["agent_response"] = result.get("output", "")
        else:
            state["agent_response"] = str(result)
    return state

def safe_invoke(agent, input_message, max_length=2000):
    # Truncate user query
    if isinstance(input_message, dict):
        if "input" in input_message and len(input_message["input"]) > max_length:
            input_message["input"] = input_message["input"][:max_length] + "..."

    # Run the agent
    result = agent.invoke(input_message)

    # If the result is very long, cut it down before returning
    if isinstance(result, dict) and "output" in result:
        if len(result["output"]) > 4000:  # adjust threshold
            result["output"] = result["output"][:4000] + "..."
    return result



def parser_node(state):
    if not state.get("use_neo4j") and state.get("agent_response"):
        parsed = parse_output_with_llm(state["agent_response"])
        state["parsed_data"] = parsed
    return state

def neo4j_insert_node(state):
    if not state.get("use_neo4j") and state.get("parsed_data") is not None:
        insert_structured_data_to_neo4j(state["parsed_data"], driver)
    else:
        # Optionally log or set error state
        if not state.get("parsed_data"):
            print("[Neo4j Insert] No parsed data to insert.")
    return state


def return_node(state):
    if state.get("use_neo4j"):
        return state.get("final_answer")
    else:
        return state.get("agent_response") or "Sorry, no results found."

########################################

# 3. Build the LangGraph workflow graph
builder = StateGraph(dict)  # Our state is a dict

# Add nodes
builder.add_node("neo4j_search", neo4j_search_node)
builder.add_node("neo4j_check", neo4j_check_node)
builder.add_node("agent_search", agent_search_node)
builder.add_node("parser", parser_node)
builder.add_node("neo4j_insert", neo4j_insert_node)
builder.add_node("return", return_node)

# Add edges (flow)
builder.add_edge(START, "neo4j_search")
builder.add_edge("neo4j_search", "neo4j_check")
def check_route(state):
    return "return" if state.get("use_neo4j") else "agent_search"
builder.add_conditional_edges("neo4j_check", check_route)
builder.add_edge("agent_search", "parser")
builder.add_edge("parser", "neo4j_insert")
builder.add_edge("neo4j_insert", "return")
builder.add_edge("return", END)

# Compile the workflow
graph = builder.compile()

########################################

if __name__ == "__main__":
    # Input query from user
    user_query = "What is the latest research on deep learning?"
    state = {"user_query": user_query}
    result = graph.invoke(state)
    print("\n=== Final Response ===\n")
    print(result)
