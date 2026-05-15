import asyncio
import re
from typing import Callable, Any

from models import TreeNodeConfig, LeafResponse
from llm_client import (
    get_competence_score,
    get_answer,
    decompose_query,
    synthesize_response,
    classify_document,
    detect_top_level_nodes,
    expand_node_into_leaves,
    check_node_relevance,
    generate_node_system_prompt,
    merge_node_lists,
    DEFAULT_MODEL,
)

Callback = Callable[..., Any]

AUTO_EXPAND_THRESHOLD = 0.7


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower().strip())
    return slug.strip("_")[:60]


class TreeEngine:
    def __init__(self):
        self.root = TreeNodeConfig(
            id="root",
            name="Orchestratore",
            model=DEFAULT_MODEL,
            role="root",
            system_prompt=(
                "Sei l'orchestratore principale del sistema AGI Tree. "
                "Il tuo compito è ricevere le domande dell'utente, decomporle se necessario, "
                "e sintetizzare le risposte dei modelli specializzati in una risposta coerente."
            ),
            children=[],
        )
        self._node_map: dict[str, TreeNodeConfig] = {"root": self.root}
        self._nodes_data: list[dict] = []  # raw text data for each top-level node
        self.processing_log: dict = {
            "documents": [], "total_nodes": 0, "total_leaves": 0,
            "doc_type": None, "doc_description": None,
        }

    def _rebuild_node_map(self):
        self._node_map = {}
        self._walk(self.root)

    def _walk(self, node: TreeNodeConfig):
        self._node_map[node.id] = node
        for child in node.children:
            self._walk(child)

    def get_node(self, node_id: str) -> TreeNodeConfig | None:
        return self._node_map.get(node_id)

    def get_tree_dict(self) -> dict:
        return {"tree": self.root.model_dump()}

    def _get_node_text(self, node_id: str) -> str:
        """Get raw text for a node from stored data."""
        for nd in self._nodes_data:
            if nd.get("id") == node_id:
                return nd.get("text", "")
        return ""

    def is_node_expanded(self, node_id: str) -> bool:
        """Check if a node already has leaves."""
        node = self.get_node(node_id)
        return node is not None and len(node.children) > 0

    # ------------------------------------------------------------------
    # Build tree from document — ONLY top-level nodes, no leaves
    # ------------------------------------------------------------------
    async def build_tree_from_document(
        self, document_text: str, filename: str = "", callback: Callback | None = None
    ):
        doc_log: dict = {"filename": filename, "doc_type": "", "detection_method": "", "nodes": []}

        # --- Step 1: Classify document ---
        if callback:
            await callback("analysis", "root", "Orchestratore", "classifying")

        classification = await classify_document(document_text)
        doc_type = classification.get("doc_type", "other")
        doc_description = classification.get("description", "")
        doc_log["doc_type"] = doc_type

        if callback:
            await callback("analysis", "root", "Orchestratore", "classified", classification)

        # --- Step 2: Detect top-level nodes ---
        if callback:
            await callback("analysis", "root", "Orchestratore", "detecting_nodes")

        nodes_raw, method = await detect_top_level_nodes(document_text, doc_type)
        doc_log["detection_method"] = method

        if callback:
            await callback(
                "analysis", "root", "Orchestratore", "nodes_found",
                {"count": len(nodes_raw), "method": method, "names": [n["name"] for n in nodes_raw]},
            )

        # --- Handle merge with existing nodes ---
        if self._nodes_data:
            existing_names = [n["name"] for n in self._nodes_data]
            new_names = [n["name"] for n in nodes_raw]
            decisions = await merge_node_lists(existing_names, new_names)

            merged = list(self._nodes_data)
            for nd in nodes_raw:
                decision = next((d for d in decisions if d.get("new_node") == nd["name"]), None)
                if decision and decision.get("action") == "merge" and decision.get("merge_with"):
                    target = next((e for e in merged if e["name"] == decision["merge_with"]), None)
                    if target:
                        target["text"] += "\n\n" + nd["text"]
                        continue
                merged.append(nd)
            nodes_raw = merged

        # --- Step 3: Create top-level tree nodes (no leaves) ---
        new_children: list[TreeNodeConfig] = []

        for nd in nodes_raw:
            nd_name = nd["name"]
            nd_text = nd["text"]
            nd_id = f"node_{_slugify(nd_name)}"
            nd["id"] = nd_id  # store id for later lookup

            if callback:
                await callback("building", nd_id, nd_name, "creating")

            node_prompt = await generate_node_system_prompt(nd_name)
            tree_node = TreeNodeConfig(
                id=nd_id, name=nd_name, model=DEFAULT_MODEL, role="node",
                system_prompt=node_prompt, context=nd_text[:2000], children=[],
            )
            new_children.append(tree_node)

            doc_log["nodes"].append({"name": nd_name, "char_count": len(nd_text), "expanded": False})

            if callback:
                await callback("building", nd_id, nd_name, "created")

        self._nodes_data = nodes_raw
        self.root = self.root.model_copy(deep=True)
        self.root.children = new_children
        self._rebuild_node_map()

        self.processing_log["documents"].append(doc_log)
        self.processing_log["total_nodes"] = len(new_children)
        self.processing_log["total_leaves"] = sum(len(c.children) for c in new_children)
        self.processing_log["doc_type"] = doc_type
        self.processing_log["doc_description"] = doc_description

        if callback:
            await callback("analysis", "root", "Orchestratore", "complete", {
                "total_nodes": len(new_children),
                "doc_type": doc_type,
                "doc_description": doc_description,
            })

    # ------------------------------------------------------------------
    # Expand a node into leaves (on demand or auto)
    # ------------------------------------------------------------------
    async def expand_node(
        self, node_id: str, callback: Callback | None = None
    ) -> bool:
        """Expand a top-level node into leaf sub-sections. Returns True if expanded."""
        node = self.get_node(node_id)
        if not node or node.role == "leaf" or node.role == "root":
            return False

        if node.children:
            return False  # already expanded

        node_text = self._get_node_text(node_id)
        if not node_text:
            node_text = node.context  # fallback

        if callback:
            await callback("expanding", node_id, node.name, "start")

        leaves_data = await expand_node_into_leaves(node.name, node_text)

        if callback:
            await callback("expanding", node_id, node.name, "leaves_found",
                           {"count": len(leaves_data), "names": [l["name"] for l in leaves_data]})

        for leaf in leaves_data:
            leaf_name = leaf["name"] if isinstance(leaf, dict) else leaf
            leaf_excerpt = leaf.get("excerpt", "") if isinstance(leaf, dict) else ""
            leaf_id = f"leaf_{_slugify(node.name)}_{_slugify(leaf_name)}"

            leaf_prompt = await generate_node_system_prompt(
                f"{leaf_name} (nella sezione: {node.name})"
            )
            leaf_node = TreeNodeConfig(
                id=leaf_id, name=leaf_name, model=DEFAULT_MODEL, role="leaf",
                system_prompt=leaf_prompt, context=leaf_excerpt, children=[],
            )
            node.children.append(leaf_node)

            if callback:
                await callback("expanding", leaf_id, leaf_name, "leaf_created",
                               {"parent": node.name})

        self._rebuild_node_map()

        # Update processing log
        self.processing_log["total_leaves"] = sum(
            len(c.children) for c in self.root.children
        )

        if callback:
            await callback("expanding", node_id, node.name, "complete",
                           {"leaves_count": len(node.children)})

        return True

    # ------------------------------------------------------------------
    # Score nodes (top-level only, or leaves if expanded)
    # ------------------------------------------------------------------
    async def score_nodes(
        self,
        query: str,
        callback: Callback | None = None,
    ) -> tuple[dict[str, float], dict[str, str]]:
        """Score all scorable nodes. Returns (scores, reasons)."""
        scores: dict[str, float] = {}
        reasons: dict[str, str] = {}

        async def score_single(node: TreeNodeConfig):
            if callback:
                await callback("discovery", node.id, node.name, "start")
            result = await get_competence_score(node.system_prompt, query, node.context, node.model)
            scores[node.id] = result["score"]
            reasons[node.id] = result["reason"]
            if callback:
                await callback("discovery", node.id, node.name, "score",
                               {"score": result["score"], "reason": result["reason"]})

        tasks = []
        for child in self.root.children:
            if child.children:
                for leaf in child.children:
                    tasks.append(score_single(leaf))
            else:
                tasks.append(score_single(child))

        await asyncio.gather(*tasks)
        return scores, reasons

    # ------------------------------------------------------------------
    # Auto-expand logic
    # ------------------------------------------------------------------
    async def maybe_auto_expand(
        self,
        query: str,
        scores: dict[str, float],
        callback: Callback | None = None,
    ) -> tuple[bool, str | None]:
        """
        If no node scores >= AUTO_EXPAND_THRESHOLD, check if the best-scoring
        node is relevant. If yes, expand it. Returns (expanded, node_id).
        """
        max_score = max(scores.values()) if scores else 0.0
        if max_score >= AUTO_EXPAND_THRESHOLD:
            return False, None

        # Find best scoring node (must be unexpanded)
        best_id = max(scores, key=scores.get) if scores else None
        if not best_id:
            return False, None

        # Find parent node if best_id is already a leaf
        best_node = self.get_node(best_id)
        if not best_node:
            return False, None

        # If best_node is a top-level node (not expanded), that's the candidate
        # If best_node is a leaf, its parent is already expanded — no further expansion
        if best_node.role == "leaf":
            return False, None

        # Check relevance before expanding
        if callback:
            await callback("auto_expand", best_id, best_node.name, "checking_relevance")

        is_relevant = await check_node_relevance(
            best_node.name, best_node.context, query
        )

        if not is_relevant:
            if callback:
                await callback("auto_expand", best_id, best_node.name, "not_relevant")
            return False, None

        # Expand!
        if callback:
            await callback("auto_expand", best_id, best_node.name, "expanding")

        expanded = await self.expand_node(best_id, callback=callback)
        return expanded, best_id if expanded else None

    # ------------------------------------------------------------------
    # Select best responders
    # ------------------------------------------------------------------
    def select_responders(
        self, scores: dict[str, float], threshold: float = 0.3, max_count: int = 4
    ) -> list[str]:
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        selected = [nid for nid, s in sorted_items if s >= threshold][:max_count]
        if not selected and sorted_items:
            selected = [sorted_items[0][0]]
        return selected

    # ------------------------------------------------------------------
    # Route and respond
    # ------------------------------------------------------------------
    async def route_and_respond(
        self,
        query: str,
        selected_ids: list[str],
        scores: dict[str, float],
        callback: Callback | None = None,
    ) -> tuple[str, list[str], list[LeafResponse]]:
        sub_queries = await decompose_query(query, self.root.model)

        if callback:
            await callback("routing", "root", self.root.name, "decomposed", sub_queries)

        all_responses: list[LeafResponse] = []

        for sub_query in sub_queries:
            tasks = []
            for nid in selected_ids:
                node = self.get_node(nid)
                if node:
                    tasks.append(self._get_node_response(node, sub_query, scores.get(nid, 0), callback))
            responses = await asyncio.gather(*tasks)
            all_responses.extend(responses)

        if callback:
            await callback("synthesis", "root", self.root.name, "start")

        response_dicts = [
            {"name": r.node_name, "score": r.score, "response": r.response}
            for r in all_responses
        ]
        final = await synthesize_response(query, response_dicts, self.root.model)

        if callback:
            await callback("synthesis", "root", self.root.name, "complete")

        return final, sub_queries, all_responses

    async def _get_node_response(
        self, node: TreeNodeConfig, query: str, score: float, callback: Callback | None = None,
    ) -> LeafResponse:
        if callback:
            await callback("answering", node.id, node.name, "start")

        response = await get_answer(node.system_prompt, query, node.context, node.model)

        if callback:
            await callback("answering", node.id, node.name, "complete")

        return LeafResponse(node_id=node.id, node_name=node.name, response=response, score=score)

    # ------------------------------------------------------------------
    # Full pipeline with auto-expand
    # ------------------------------------------------------------------
    async def process_query(
        self, query: str, callback: Callback | None = None,
    ) -> dict:
        if not self.root.children:
            return {
                "response": "Nessun documento caricato. Carica un documento per iniziare.",
                "scores": {},
                "reasons": {},
                "selected_leaves": [],
                "sub_queries": [],
                "leaf_responses": [],
                "auto_expanded": None,
            }

        # Score all scorable nodes
        scores, reasons = await self.score_nodes(query, callback=callback)

        # Check if auto-expansion is needed
        expanded, expanded_node_id = await self.maybe_auto_expand(query, scores, callback=callback)

        # If we expanded, re-score the new leaves
        if expanded and expanded_node_id:
            scores.pop(expanded_node_id, None)
            reasons.pop(expanded_node_id, None)
            expanded_node = self.get_node(expanded_node_id)
            if expanded_node:
                for leaf in expanded_node.children:
                    if callback:
                        await callback("discovery", leaf.id, leaf.name, "start")
                    result = await get_competence_score(leaf.system_prompt, query, leaf.context, leaf.model)
                    scores[leaf.id] = result["score"]
                    reasons[leaf.id] = result["reason"]
                    if callback:
                        await callback("discovery", leaf.id, leaf.name, "score",
                                       {"score": result["score"], "reason": result["reason"]})

        # Select best responders
        selected = self.select_responders(scores)

        if callback:
            await callback("selection", "root", self.root.name, "selected", selected)

        # Route and respond
        final_response, sub_queries, leaf_responses = await self.route_and_respond(
            query, selected, scores, callback
        )

        return {
            "response": final_response,
            "scores": {k: round(v, 3) for k, v in scores.items()},
            "reasons": reasons,
            "selected_leaves": selected,
            "sub_queries": sub_queries,
            "leaf_responses": [
                {"node_id": r.node_id, "node_name": r.node_name, "response": r.response, "score": round(r.score, 3)}
                for r in leaf_responses
            ],
            "auto_expanded": expanded_node_id,
        }
