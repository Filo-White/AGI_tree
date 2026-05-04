import asyncio
import re
from typing import Callable, Any

from models import TreeNodeConfig, LeafResponse
from llm_client import (
    get_competence_score,
    get_answer,
    decompose_query,
    synthesize_response,
    analyze_document_topics,
    generate_node_system_prompt,
    merge_document_into_topics,
    DEFAULT_MODEL,
)

Callback = Callable[..., Any]


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower().strip())
    return slug.strip("_")


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
        self._raw_topics: list[dict] = []

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

    # ------------------------------------------------------------------
    # Build / update tree from document
    # ------------------------------------------------------------------
    async def build_tree_from_document(
        self, document_text: str, callback: Callback | None = None
    ):
        if callback:
            await callback("analysis", "root", "Orchestratore", "start")

        if self._raw_topics:
            topics = await merge_document_into_topics(self._raw_topics, document_text)
        else:
            topics = await analyze_document_topics(document_text)

        self._raw_topics = topics

        if callback:
            await callback("analysis", "root", "Orchestratore", "topics_found", [t["name"] for t in topics])

        new_children: list[TreeNodeConfig] = []

        for topic in topics:
            topic_id = f"node_{_slugify(topic['name'])}"
            existing_node = self._node_map.get(topic_id)

            if callback:
                await callback("building", topic_id, topic["name"], "start")

            if not existing_node:
                topic_prompt = await generate_node_system_prompt(topic["name"])
                topic_node = TreeNodeConfig(
                    id=topic_id,
                    name=topic["name"],
                    model=DEFAULT_MODEL,
                    role="node",
                    system_prompt=topic_prompt,
                    children=[],
                )
            else:
                topic_node = existing_node.model_copy(deep=True)
                topic_node.children = []

            subtopics = topic.get("subtopics", [])
            for sub in subtopics:
                sub_name = sub["name"] if isinstance(sub, dict) else sub
                sub_excerpt = sub.get("excerpt", "") if isinstance(sub, dict) else ""
                leaf_id = f"leaf_{_slugify(topic['name'])}_{_slugify(sub_name)}"

                existing_leaf = self._node_map.get(leaf_id)

                if existing_leaf:
                    merged_context = existing_leaf.context
                    if sub_excerpt and sub_excerpt not in merged_context:
                        merged_context += f"\n\n{sub_excerpt}"
                    leaf_node = existing_leaf.model_copy(deep=True)
                    leaf_node.context = merged_context
                else:
                    leaf_prompt = await generate_node_system_prompt(
                        f"{sub_name} (nell'ambito di {topic['name']})"
                    )
                    leaf_node = TreeNodeConfig(
                        id=leaf_id,
                        name=sub_name,
                        model=DEFAULT_MODEL,
                        role="leaf",
                        system_prompt=leaf_prompt,
                        context=sub_excerpt,
                        children=[],
                    )

                topic_node.children.append(leaf_node)

                if callback:
                    await callback("building", leaf_id, sub_name, "created")

            new_children.append(topic_node)

            if callback:
                await callback("building", topic_id, topic["name"], "complete")

        self.root = self.root.model_copy(deep=True)
        self.root.children = new_children
        self._rebuild_node_map()

        if callback:
            await callback("analysis", "root", "Orchestratore", "complete")

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
            score = await get_competence_score(
                node.system_prompt, query, node.context, node.model
            )
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
        max_leaves: int = 4,
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
                            leaf, sub_query, scores.get(leaf_id, 0), callback
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
        callback: Callback | None = None,
    ) -> LeafResponse:
        if callback:
            await callback("answering", leaf.id, leaf.name, "start")

        response = await get_answer(
            leaf.system_prompt, query, leaf.context, leaf.model
        )

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
        callback: Callback | None = None,
    ) -> dict:
        if not self.root.children:
            return {
                "response": "Nessun documento caricato. Carica un documento per iniziare.",
                "scores": {},
                "selected_leaves": [],
                "sub_queries": [],
                "leaf_responses": [],
            }

        scores = await self.phase1_discovery(query, callback=callback)

        selected = await self.phase2_select_leaves(scores)

        if callback:
            await callback("selection", "root", self.root.name, "selected", selected)

        final_response, sub_queries, leaf_responses = await self.phase3_route_and_respond(
            query, selected, scores, callback
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
