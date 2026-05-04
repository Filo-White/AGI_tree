import asyncio
from typing import Callable, Any

from models import TreeNodeConfig, LeafResponse
from llm_client import (
    get_competence_score,
    get_answer,
    decompose_query,
    synthesize_response,
)

Callback = Callable[..., Any]


class TreeEngine:
    def __init__(self, tree_config: dict):
        self.root = TreeNodeConfig(**tree_config["tree"])
        self._node_map: dict[str, TreeNodeConfig] = {}
        self._build_node_map(self.root)

    def _build_node_map(self, node: TreeNodeConfig):
        self._node_map[node.id] = node
        for child in node.children:
            self._build_node_map(child)

    def get_node(self, node_id: str) -> TreeNodeConfig | None:
        return self._node_map.get(node_id)

    def get_all_nodes(self) -> dict[str, TreeNodeConfig]:
        return self._node_map

    # ------------------------------------------------------------------
    # Phase 1 — Discovery (top-down): collect competence scores
    # ------------------------------------------------------------------
    async def phase1_discovery(
        self,
        query: str,
        node: TreeNodeConfig | None = None,
        callback: Callback | None = None,
    ) -> dict[str, float]:
        if node is None:
            node = self.root

        if callback:
            await callback("discovery", node.id, node.name, "start")

        if not node.children:
            score = await get_competence_score(node.system_prompt, query, node.model)
            if callback:
                await callback("discovery", node.id, node.name, "score", score)
            return {node.id: score}

        tasks = [
            self.phase1_discovery(query, child, callback) for child in node.children
        ]
        results = await asyncio.gather(*tasks)

        merged: dict[str, float] = {}
        for result in results:
            merged.update(result)

        if callback:
            await callback("discovery", node.id, node.name, "aggregated", merged)

        return merged

    # ------------------------------------------------------------------
    # Phase 2 — Select best leaves
    # ------------------------------------------------------------------
    async def phase2_select_leaves(
        self,
        scores: dict[str, float],
        threshold: float = 0.3,
        max_leaves: int = 3,
    ) -> list[str]:
        sorted_leaves = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        selected = [lid for lid, s in sorted_leaves if s >= threshold][:max_leaves]

        if not selected and sorted_leaves:
            selected = [sorted_leaves[0][0]]

        return selected

    # ------------------------------------------------------------------
    # Phase 3 — Route sub-queries and collect responses
    # ------------------------------------------------------------------
    async def phase3_route_and_respond(
        self,
        query: str,
        selected_leaf_ids: list[str],
        scores: dict[str, float],
        document_context: str | None = None,
        callback: Callback | None = None,
    ) -> tuple[str, list[str], list[LeafResponse]]:
        sub_queries = await decompose_query(query, self.root.model)

        if callback:
            await callback("routing", "root", self.root.name, "decomposed", sub_queries)

        all_responses: list[LeafResponse] = []

        for sub_query in sub_queries:
            tasks = []
            for leaf_id in selected_leaf_ids:
                leaf = self.get_node(leaf_id)
                if leaf:
                    tasks.append(
                        self._get_leaf_response(
                            leaf, sub_query, scores.get(leaf_id, 0), document_context, callback
                        )
                    )
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

    async def _get_leaf_response(
        self,
        leaf: TreeNodeConfig,
        query: str,
        score: float,
        document_context: str | None = None,
        callback: Callback | None = None,
    ) -> LeafResponse:
        if callback:
            await callback("answering", leaf.id, leaf.name, "start")

        response = await get_answer(leaf.system_prompt, query, document_context, leaf.model)

        if callback:
            await callback("answering", leaf.id, leaf.name, "complete")

        return LeafResponse(
            node_id=leaf.id,
            node_name=leaf.name,
            response=response,
            score=score,
        )

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------
    async def process_query(
        self,
        query: str,
        document_context: str | None = None,
        callback: Callback | None = None,
    ) -> dict:
        scores = await self.phase1_discovery(query, callback=callback)

        selected = await self.phase2_select_leaves(scores)

        if callback:
            await callback("selection", "root", self.root.name, "selected", selected)

        final_response, sub_queries, leaf_responses = await self.phase3_route_and_respond(
            query, selected, scores, document_context, callback
        )

        return {
            "response": final_response,
            "scores": {k: round(v, 3) for k, v in scores.items()},
            "selected_leaves": selected,
            "sub_queries": sub_queries,
            "leaf_responses": [
                {
                    "node_id": r.node_id,
                    "node_name": r.node_name,
                    "response": r.response,
                    "score": round(r.score, 3),
                }
                for r in leaf_responses
            ],
        }
