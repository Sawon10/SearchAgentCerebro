from dataclasses import dataclass, field
from typing import Any, List, Optional, Dict

@dataclass
class SearchAgentState:
    user_query: str = ""
    neo4j_results: Optional[List[Dict[str, Any]]] = field(default_factory=list)
    neo4j_hit: bool = False
    use_neo4j: bool = False
    agent_response: Optional[str] = None
    parsed_data: Optional[Dict[str, Any]] = None
    final_answer: Optional[str] = None
    error: Optional[str] = None
