import argparse
from typing import List, Dict

import litellm
from pydantic import BaseModel

import config
from inference.user_manager import UserManager, get_user_managers
from inference.user_role import get_user_roles
from utils import file_cache


CYCLE_RESOLUTION_PROMPT = """
You are a corporate organization researcher helping to resolve circular reporting relationships in a company hierarchy.

You have detected a CYCLE in the reporting structure where the following people form a circular reporting chain:

<cycle_members>
{cycle_info}
</cycle_members>

This creates an impossible reporting structure. Your task is to break this cycle by determining the correct reporting relationships.

Here are OTHER people in the organization who could be potential managers:

<possible_managers>
{possible_managers}
</possible_managers>

Consider:
- Job titles and seniority levels
- Project responsibilities and scope
- The reasoning provided for each current assignment
- Typical organizational structures (e.g., external consultants report to internal leaders, junior roles report to senior)
- You can assign managers from OUTSIDE the cycle to break it
- Only set manager to null if someone is truly at the top (e.g., CEO, or external engagement lead with no internal oversight)

Please output the corrected manager assignments for ONLY the people in this cycle. At least one person's manager should be changed to someone outside the cycle or set to null.

Return your response as a JSON array with the corrected manager assignments:
[
  {{"name": "Person A", "manager": "Correct Manager or null", "reason": "Brief explanation"}},
  {{"name": "Person B", "manager": "Correct Manager", "reason": "Brief explanation"}},
  ...
]
"""


class UserManagerList(BaseModel):
    user_managers: List[UserManager]


def build_manager_graph(user_managers: List[UserManager]) -> Dict[str, str | None]:
    """Build a graph mapping each person to their manager."""
    graph = {}
    for um in user_managers:
        graph[um.name] = um.manager
    return graph


def detect_cycles(graph: Dict[str, str | None]) -> List[List[str]]:
    """
    Detect all cycles in the manager graph using DFS.
    Returns a list of cycles, where each cycle is a list of names.
    """
    visited = set()
    rec_stack = set()
    cycles = []
    
    def dfs(node: str, path: List[str]) -> None:
        if node in rec_stack:
            # Found a cycle - extract it from the path
            cycle_start_idx = path.index(node)
            cycle = path[cycle_start_idx:]
            cycles.append(cycle)
            return
        
        if node in visited:
            return
        
        visited.add(node)
        rec_stack.add(node)
        path.append(node)
        
        manager = graph.get(node)
        if manager and manager in graph:  # Only follow if manager is in our graph
            dfs(manager, path[:])
        
        rec_stack.remove(node)
    
    # Try DFS from each node
    for node in graph:
        if node not in visited:
            dfs(node, [])
    
    return cycles


def get_user_context(name: str, user_managers: List[UserManager]) -> Dict[str, str]:
    """Get full context for a user including their role information."""
    # Get manager info
    manager_info = next((um for um in user_managers if um.name == name), None)
    
    # Get role info for title and project
    user_roles = get_user_roles()
    role_info = next((ur for ur in user_roles if ur.name == name), None)
    
    context = {
        "name": name,
        "manager": manager_info.manager if manager_info else None,
        "reason": manager_info.reason if manager_info else "",
        "title": role_info.title if role_info else "Unknown",
        "project": role_info.project if role_info else "Unknown",
    }
    
    return context


def resolve_cycle(cycle: List[str], user_managers: List[UserManager]) -> List[UserManager]:
    """
    Use LLM to resolve a cycle by determining correct reporting relationships.
    """
    # Gather context for all people in the cycle
    cycle_contexts = [get_user_context(name, user_managers) for name in cycle]
    
    # Format the cycle information
    cycle_info_str = "\n\n".join(
        f"Name: {ctx['name']}\n"
        f"Title: {ctx['title']}\n"
        f"Project: {ctx['project']}\n"
        f"Current Manager: {ctx['manager']}\n"
        f"Reason: {ctx['reason']}"
        for ctx in cycle_contexts
    )
    
    # Get all people NOT in the cycle as possible managers
    cycle_set = set(cycle)
    possible_managers = [
        get_user_context(um.name, user_managers) 
        for um in user_managers 
        if um.name not in cycle_set
    ]
    
    # Format possible managers information
    possible_managers_str = "\n".join(
        f"- {ctx['name']}: {ctx['title']}, working on {ctx['project']}"
        for ctx in possible_managers
    )
    
    prompt = CYCLE_RESOLUTION_PROMPT.format(
        cycle_info=cycle_info_str,
        possible_managers=possible_managers_str
    )
    
    response = litellm.completion(
        model=config.DEFAULT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format=UserManagerList,
        metadata={"trace_name": "resolve_manager_cycle"},
    )
    
    resolved = UserManagerList.model_validate_json(
        response["choices"][0]["message"]["content"]
    )
    
    return resolved.user_managers


@file_cache(f"{config.INFERENCE_DATA_ROOT}/normalized_user_managers.json")
def get_normalized_user_managers() -> List[UserManager]:
    """
    Load user managers, detect cycles, resolve them using LLM, and return normalized hierarchy.
    """
    user_managers = get_user_managers()
    
    # Build graph and detect cycles
    graph = build_manager_graph(user_managers)
    cycles = detect_cycles(graph)
    
    if not cycles:
        print("No cycles detected in the manager hierarchy.")
        return user_managers
    
    print(f"Detected {len(cycles)} cycle(s) in the manager hierarchy:")
    for i, cycle in enumerate(cycles, 1):
        print(f"  Cycle {i}: {' -> '.join(cycle)} -> {cycle[0]}")
    
    # Create a dict for easy updates
    manager_dict = {um.name: um for um in user_managers}
    
    # Resolve each cycle
    for i, cycle in enumerate(cycles, 1):
        print(f"\nResolving cycle {i}: {' -> '.join(cycle)}")
        resolved_managers = resolve_cycle(cycle, user_managers)
        
        # Update the manager dictionary with resolved relationships
        for resolved in resolved_managers:
            if resolved.name in manager_dict:
                manager_dict[resolved.name].manager = resolved.manager
                manager_dict[resolved.name].reason = resolved.reason
    
    # Return the updated list
    normalized_managers = list(manager_dict.values())
    
    print(f"\nSuccessfully normalized {len(normalized_managers)} manager relationships.")
    return normalized_managers


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Normalize user managers by detecting and resolving cycles"
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force regeneration of cached inference results",
    )
    args = parser.parse_args()
    
    get_normalized_user_managers(force_refresh=args.force_refresh)

