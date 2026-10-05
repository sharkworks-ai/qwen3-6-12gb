from __future__ import annotations

from appliance.db_ext import WorkbenchDB


def graph(db: WorkbenchDB, candidate_id: str) -> dict:
    center = db.get_candidate(candidate_id)
    edges = db.lineage_for(candidate_id)
    ids = {candidate_id}
    for edge in edges:
        ids.add(edge["child_id"])
        ids.add(edge["parent_id"])

    nodes = []
    for item_id in ids:
        try:
            nodes.append(db.get_candidate(item_id))
        except KeyError:
            nodes.append({"candidate_id": item_id, "stage": "unknown"})

    return {"nodes": nodes, "edges": edges}
