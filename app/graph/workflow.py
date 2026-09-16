"""
Builds the LangGraph StateGraph for one conversational turn.

    START -> receive_turn -> retrieve_context -> evaluate_retrieval
          -> tool_call_if_required -> generate_response
          -> evaluate_confidence -> decide_handoff
          -> (conditional) finalize_handoff -> END
                          \\-> END directly if no handoff needed

The conditional edge after decide_handoff is what makes the graph
actually control behavior: state.handoff_required, computed from
retrieval/tool/confidence signals earlier in the same graph, changes
which node runs next and therefore what the final response text is.
"""
from __future__ import annotations

from langgraph.graph import END, StateGraph

from app.config import Settings
from app.graph.nodes import build_nodes
from app.graph.state import AgentState
from app.rag.retriever import Retriever
from app.services.llm import LLMService


def route_after_handoff_decision(state: AgentState) -> str:
    return "finalize_handoff" if state.get("handoff_required") else END


def build_graph(retriever: Retriever, llm_service: LLMService, settings: Settings):
    nodes = build_nodes(retriever, llm_service, settings)

    graph = StateGraph(AgentState)
    for name, func in nodes.items():
        graph.add_node(name, func)

    graph.set_entry_point("receive_turn")
    graph.add_edge("receive_turn", "retrieve_context")
    graph.add_edge("retrieve_context", "evaluate_retrieval")
    graph.add_edge("evaluate_retrieval", "tool_call_if_required")
    graph.add_edge("tool_call_if_required", "generate_response")
    graph.add_edge("generate_response", "evaluate_confidence")
    graph.add_edge("evaluate_confidence", "decide_handoff")
    graph.add_conditional_edges(
        "decide_handoff", route_after_handoff_decision, {"finalize_handoff": "finalize_handoff", END: END}
    )
    graph.add_edge("finalize_handoff", END)

    return graph.compile()
