from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI

from Search_Tools import neo4j_search
from Search_ReAct_Agent import get_search_agent, parse_output_with_llm, extract_search_phrase_from_query
from Neo4j_setup import insert_structured_data_to_neo4j, driver


########################################
# 2. Node function definitions
########################################

def neo4j_search_node(state):
    # Get the user query
    query = state["user_query"]
    search_keyword = extract_search_phrase_from_query(query)
    cypher_query = (
        f"""
        MATCH (c:Concept)
        WHERE toLower(c.name) CONTAINS toLower('{search_keyword}')
        OPTIONAL MATCH (c)-[r]->(x)
        RETURN c, collect(r) as rels, collect(x) as linked
        LIMIT 5
        """
    )
    results = neo4j_search(cypher_query)
    if isinstance(results, list):
        state["neo4j_results"] = results
        state["neo4j_hit"] = bool(results)
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
        result = agent.invoke({"input": state["user_query"]})
        if isinstance(result, dict):
            state["agent_response"] = result.get("output", "")
        else:
            state["agent_response"] = str(result)
    return state

def parser_node(state):
    if not state.get("use_neo4j") and state.get("agent_response"):
        parsed = parse_output_with_llm(state["agent_response"])
        state["parsed_data"] = parsed
    return state

def neo4j_insert_node(state):
    if not state.get("use_neo4j") and state.get("parsed_data"):
        insert_structured_data_to_neo4j(state["parsed_data"], driver)
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
    user_query = "Explain hash maps with examples, find datasets about hash maps, and GitHub repos implementing them."
    state = {"user_query": user_query}
    result = graph.invoke(state)
    print("\n=== Final Response ===\n")
    print(result)
