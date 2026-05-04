import json
import os

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from tree_engine import TreeEngine
from document_processor import process_document

load_dotenv()

app = FastAPI(title="AGI Tree")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TREE_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "tree_config.json")


def load_tree_config() -> dict:
    with open(TREE_CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_tree_config(config: dict):
    with open(TREE_CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


engine: TreeEngine = TreeEngine(load_tree_config())
document_store: dict[str, str | None] = {"context": None, "filename": None}


@app.get("/api/tree")
async def get_tree():
    return load_tree_config()


@app.put("/api/tree")
async def update_tree(config: dict):
    try:
        global engine
        test_engine = TreeEngine(config)
        save_tree_config(config)
        engine = test_engine
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    contents = await file.read()
    text = process_document(file.filename, contents)

    max_chars = 15000
    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n[... documento troncato ...]"

    document_store["context"] = text
    document_store["filename"] = file.filename
    return {"status": "ok", "filename": file.filename, "char_count": len(text)}


@app.get("/api/document")
async def get_document_info():
    if document_store["context"]:
        return {
            "filename": document_store["filename"],
            "char_count": len(document_store["context"]),
        }
    return {"filename": None, "char_count": 0}


@app.delete("/api/document")
async def clear_document():
    document_store["context"] = None
    document_store["filename"] = None
    return {"status": "ok"}


@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            query = data.get("query", "").strip()

            if not query:
                await websocket.send_json({"type": "error", "message": "Query vuota"})
                continue

            async def progress_callback(phase, node_id, node_name, status, data=None):
                payload = {
                    "type": "progress",
                    "phase": phase,
                    "node_id": node_id,
                    "node_name": node_name,
                    "status": status,
                }
                if data is not None:
                    payload["data"] = data
                await websocket.send_json(payload)

            try:
                result = await engine.process_query(
                    query,
                    document_context=document_store.get("context"),
                    callback=progress_callback,
                )
                await websocket.send_json({"type": "result", **result})
            except Exception as e:
                await websocket.send_json({"type": "error", "message": str(e)})

    except WebSocketDisconnect:
        pass
    except Exception:
        pass
