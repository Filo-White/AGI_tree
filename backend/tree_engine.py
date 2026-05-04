import asyncio
import re
from typing import Callable, Any

from models import TreeNodeConfig, LeafResponse
from llm_client import (
    get_competence_score,
    get_answer,
    decompose_query,
    synthesize_response,
    detect_chapters,
    extract_sections_for_chapter,
    generate_node_system_prompt,
    merge_chapter_lists,
    DEFAULT_MODEL,
)

Callback = Callable[..., Any]


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
        self._chapters_data: list[dict] = []
        self.processing_log: dict = {"documents": [], "total_chapters": 0, "total_sections": 0}

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
    # Build / update tree from document (two-phase: chapters → sections)
    # ------------------------------------------------------------------
    async def build_tree_from_document(
        self, document_text: str, filename: str = "", callback: Callback | None = None
    ):
        doc_log: dict = {"filename": filename, "detection_method": "", "chapters": []}

        # --- Phase 1: detect chapters ---
        if callback:
            await callback("analysis", "root", "Orchestratore", "detecting_chapters")

        chapters, method = await detect_chapters(document_text)
        doc_log["detection_method"] = method

        if callback:
            await callback(
                "analysis", "root", "Orchestratore", "chapters_found",
                {"count": len(chapters), "method": method, "names": [c["name"] for c in chapters]},
            )

        # --- Handle merge with existing chapters ---
        if self._chapters_data:
            existing_names = [c["name"] for c in self._chapters_data]
            new_names = [c["name"] for c in chapters]
            decisions = await merge_chapter_lists(existing_names, new_names)

            merged_chapters = list(self._chapters_data)
            for ch in chapters:
                decision = next((d for d in decisions if d["new_chapter"] == ch["name"]), None)
                if decision and decision.get("action") == "merge" and decision.get("merge_with"):
                    target = next((ec for ec in merged_chapters if ec["name"] == decision["merge_with"]), None)
                    if target:
                        target["text"] += "\n\n" + ch["text"]
                        target.pop("sections", None)
                        continue
                merged_chapters.append(ch)
            chapters = merged_chapters

        self._chapters_data = chapters

        # --- Phase 2: extract sections per chapter ---
        new_children: list[TreeNodeConfig] = []

        for ch_idx, chapter in enumerate(chapters):
            ch_name = chapter["name"]
            ch_text = chapter["text"]
            ch_id = f"ch_{_slugify(ch_name)}"

            if callback:
                await callback("building", ch_id, ch_name, "extracting_sections",
                               {"chapter_index": ch_idx + 1, "total_chapters": len(chapters)})

            if "sections" not in chapter:
                sections = await extract_sections_for_chapter(ch_name, ch_text)
                chapter["sections"] = sections
            else:
                sections = chapter["sections"]

            ch_log = {"name": ch_name, "char_count": len(ch_text), "sections": []}

            if callback:
                await callback("building", ch_id, ch_name, "sections_found",
                               {"count": len(sections), "names": [s["name"] for s in sections]})

            chapter_prompt = await generate_node_system_prompt(ch_name)
            chapter_node = TreeNodeConfig(
                id=ch_id, name=ch_name, model=DEFAULT_MODEL, role="node",
                system_prompt=chapter_prompt, context=ch_text[:500], children=[],
            )

            for sec in sections:
                sec_name = sec["name"] if isinstance(sec, dict) else sec
                sec_excerpt = sec.get("excerpt", "") if isinstance(sec, dict) else ""
                leaf_id = f"leaf_{_slugify(ch_name)}_{_slugify(sec_name)}"

                leaf_prompt = await generate_node_system_prompt(
                    f"{sec_name} (nel capitolo: {ch_name})"
                )
                leaf_node = TreeNodeConfig(
                    id=leaf_id, name=sec_name, model=DEFAULT_MODEL, role="leaf",
                    system_prompt=leaf_prompt, context=sec_excerpt, children=[],
                )
                chapter_node.children.append(leaf_node)

                ch_log["sections"].append({
                    "name": sec_name,
                    "excerpt_preview": sec_excerpt[:150] + ("..." if len(sec_excerpt) > 150 else ""),
                })

                if callback:
                    await callback("building", leaf_id, sec_name, "created",
                                   {"chapter": ch_name, "excerpt_len": len(sec_excerpt)})

            new_children.append(chapter_node)
            doc_log["chapters"].append(ch_log)

            if callback:
                await callback("building", ch_id, ch_name, "complete")

        self.root = self.root.model_copy(deep=True)
        self.root.children = new_children
        self._rebuild_node_map()

        self.processing_log["documents"].append(doc_log)
        self.processing_log["total_chapters"] = len(new_children)
        self.processing_log["total_sections"] = sum(len(c.children) for c in new_children)

        if callback:
            await callback("analysis", "root", "Orchestratore", "complete", {
                "total_chapters": self.processing_log["total_chapters"],
                "total_sections": self.processing_log["total_sections"],
            })

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
