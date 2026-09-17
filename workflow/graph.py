from langgraph.graph import StateGraph, END
from workflow.state import GraphState
from workflow.nodes import (
    reconstruct_node,
    sample_node,
    proposal_node,
    segment_node,
    build_mirror_node,
    check_missing_node,
    recall_node,
    dedup_node,
    orient_node,
    codegen_node,
    execute_node
)

def should_recall(state: GraphState) -> str:
    """Conditional edge logic after checking missing labels."""
    if state.get("missing_labels", []) and state.get("recall_retries", 0) < 3: # limit to 3 retries
        return "recall"
    return "dedup"

def build_workflow() -> StateGraph:
    workflow = StateGraph(GraphState)
    
    # Add nodes
    workflow.add_node("reconstruct", reconstruct_node)
    workflow.add_node("sample", sample_node)
    workflow.add_node("proposal", proposal_node)
    workflow.add_node("segment", segment_node)
    workflow.add_node("build_mirror", build_mirror_node)
    workflow.add_node("check_missing", check_missing_node)
    workflow.add_node("recall", recall_node)
    workflow.add_node("dedup", dedup_node)
    workflow.add_node("orient", orient_node)
    workflow.add_node("codegen", codegen_node)
    workflow.add_node("execute", execute_node)
    
    # Define linear flow
    workflow.set_entry_point("reconstruct")
    workflow.add_edge("reconstruct", "sample")
    workflow.add_edge("sample", "proposal")
    workflow.add_edge("proposal", "segment")
    workflow.add_edge("segment", "build_mirror")
    workflow.add_edge("build_mirror", "check_missing")
    
    # Conditional flow
    workflow.add_conditional_edges(
        "check_missing",
        should_recall,
        {
            "recall": "recall",
            "dedup": "dedup"
        }
    )
    
    # If recall happens, rebuild the mirror
    workflow.add_edge("recall", "build_mirror")
    
    # Continue linear flow
    workflow.add_edge("dedup", "orient")
    workflow.add_edge("orient", "codegen")
    workflow.add_edge("codegen", "execute")
    workflow.add_edge("execute", END)
    
    return workflow.compile()
