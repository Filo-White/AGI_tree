from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
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

engine = TreeEngine()
uploaded_files: list[str] = []
_active_ws: WebSocket | None = None


@app.get("/api/tree")
async def get_tree():
    return engine.get_tree_dict()


@app.get("/api/documents")
async def get_documents():
    return {"files": uploaded_files}


@app.delete("/api/documents")
async def clear_documents():
    global engine
    engine = TreeEngine()
    uploaded_files.clear()
    return {"status": "ok"}


@app.get("/api/processing-log")
async def get_processing_log():
    return engine.processing_log


@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    global _active_ws
    contents = await file.read()
    text = process_document(file.filename, contents)

    uploaded_files.append(file.filename)

    async def ws_callback(phase, node_id, node_name, status, data=None):
        if _active_ws:
            payload = {
                "type": "progress",
                "phase": phase,
                "node_id": node_id,
                "node_name": node_name,
                "status": status,
            }
            if data is not None:
                payload["data"] = data
            try:
                await _active_ws.send_json(payload)
            except Exception:
                pass

    try:
        await engine.build_tree_from_document(text, filename=file.filename, callback=ws_callback)
    except Exception as e:
        return {"status": "error", "message": str(e)}

    tree_data = engine.get_tree_dict()
    if _active_ws:
        try:
            await _active_ws.send_json({"type": "tree_update", **tree_data})
        except Exception:
            pass

    return {
        "status": "ok",
        "filename": file.filename,
        "char_count": len(text),
        "tree": tree_data,
    }


@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    global _active_ws
    await websocket.accept()
    _active_ws = websocket
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
                    callback=progress_callback,
                )
                await websocket.send_json({"type": "result", **result})
            except Exception as e:
                await websocket.send_json({"type": "error", "message": str(e)})

    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        if _active_ws is websocket:
            _active_ws = None
