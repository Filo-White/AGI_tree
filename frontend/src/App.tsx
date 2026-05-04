import { useState, useEffect, useRef, useCallback } from "react";
import { TreeView } from "./components/TreeView";
import { ChatPanel } from "./components/ChatPanel";
import { NodeDetail } from "./components/NodeDetail";
import {
  TreeConfig,
  TreeNodeConfig,
  ChatMessage,
  LogEntry,
  NodeState,
  WSMessage,
} from "./types";
import {
  TreesIcon,
  Upload,
  X,
  FileText,
  Activity,
  Loader2,
  Trash2,
} from "lucide-react";

export default function App() {
  const [treeConfig, setTreeConfig] = useState<TreeConfig | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [nodeStates, setNodeStates] = useState<Record<string, NodeState>>({});
  const [logEntries, setLogEntries] = useState<LogEntry[]>([]);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const [uploadedFiles, setUploadedFiles] = useState<string[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [rightTab, setRightTab] = useState<"tree" | "log">("tree");
  const wsRef = useRef<WebSocket | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    fetch("/api/tree")
      .then((r) => r.json())
      .then((data: TreeConfig) => setTreeConfig(data))
      .catch(console.error);

    fetch("/api/documents")
      .then((r) => r.json())
      .then((data) => {
        if (data.files?.length) setUploadedFiles(data.files);
      })
      .catch(() => {});
  }, []);

  useEffect(() => {
    const connect = () => {
      const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
      const ws = new WebSocket(`${protocol}//${window.location.host}/ws/chat`);

      ws.onopen = () => setIsConnected(true);
      ws.onclose = () => {
        setIsConnected(false);
        setTimeout(connect, 3000);
      };

      ws.onmessage = (event) => {
        const msg: WSMessage = JSON.parse(event.data);

        if (msg.type === "progress") {
          setLogEntries((prev) => [
            ...prev,
            {
              id: `${Date.now()}-${Math.random()}`,
              phase: msg.phase,
              nodeId: msg.node_id,
              nodeName: msg.node_name,
              status: msg.status,
              data: msg.data,
              timestamp: Date.now(),
            },
          ]);

          if (msg.phase === "building") {
            setNodeStates((prev) => ({
              ...prev,
              [msg.node_id]: { visualState: "building" },
            }));
          } else if (msg.phase === "analysis" && msg.status === "topics_found") {
            // topics found — tree will be updated via tree_update
          } else if (msg.phase === "discovery" && msg.status === "start") {
            setNodeStates((prev) => ({
              ...prev,
              [msg.node_id]: { visualState: "scoring" },
            }));
          } else if (msg.phase === "discovery" && msg.status === "score") {
            setNodeStates((prev) => ({
              ...prev,
              [msg.node_id]: { visualState: "scored", score: msg.data },
            }));
          } else if (msg.phase === "selection" && msg.status === "selected") {
            const selected = msg.data as string[];
            setNodeStates((prev) => {
              const next = { ...prev };
              selected.forEach((id) => {
                next[id] = { ...next[id], visualState: "selected" };
              });
              return next;
            });
          } else if (msg.phase === "answering" && msg.status === "start") {
            setNodeStates((prev) => ({
              ...prev,
              [msg.node_id]: {
                ...prev[msg.node_id],
                visualState: "answering",
              },
            }));
          } else if (msg.phase === "answering" && msg.status === "complete") {
            setNodeStates((prev) => ({
              ...prev,
              [msg.node_id]: {
                ...prev[msg.node_id],
                visualState: "complete",
              },
            }));
          }
        } else if (msg.type === "tree_update") {
          setTreeConfig({ tree: msg.tree });
          setNodeStates({});
        } else if (msg.type === "result") {
          setMessages((prev) => [
            ...prev,
            {
              id: `${Date.now()}`,
              role: "assistant",
              content: msg.response,
              timestamp: Date.now(),
              metadata: {
                scores: msg.scores,
                selectedLeaves: msg.selected_leaves,
                subQueries: msg.sub_queries,
                leafResponses: msg.leaf_responses,
              },
            },
          ]);
          setIsProcessing(false);
          setTimeout(() => setNodeStates({}), 4000);
        } else if (msg.type === "error") {
          setMessages((prev) => [
            ...prev,
            {
              id: `${Date.now()}`,
              role: "system",
              content: `Errore: ${msg.message}`,
              timestamp: Date.now(),
            },
          ]);
          setIsProcessing(false);
          setNodeStates({});
        }
      };

      wsRef.current = ws;
    };

    connect();
    return () => wsRef.current?.close();
  }, []);

  const sendQuery = useCallback(
    (query: string) => {
      if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
      setIsProcessing(true);
      setLogEntries([]);
      setNodeStates({});
      setMessages((prev) => [
        ...prev,
        {
          id: `${Date.now()}`,
          role: "user",
          content: query,
          timestamp: Date.now(),
        },
      ]);
      wsRef.current.send(JSON.stringify({ query }));
    },
    []
  );

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setIsUploading(true);
    setLogEntries([]);
    const formData = new FormData();
    formData.append("file", file);
    try {
      const res = await fetch("/api/upload", { method: "POST", body: formData });
      const data = await res.json();
      if (data.status === "ok") {
        setUploadedFiles((prev) => [...prev, data.filename]);
        if (data.tree) {
          setTreeConfig(data.tree);
        }
      }
    } catch (err) {
      console.error("Upload failed:", err);
    }
    setIsUploading(false);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const handleClearAll = async () => {
    try {
      await fetch("/api/documents", { method: "DELETE" });
      setUploadedFiles([]);
      setTreeConfig(null);
      setMessages([]);
      setLogEntries([]);
      setNodeStates({});
      setSelectedNodeId(null);
      // Re-fetch the empty tree
      const res = await fetch("/api/tree");
      const data = await res.json();
      setTreeConfig(data);
    } catch (err) {
      console.error(err);
    }
  };

  const findNode = (
    node: TreeNodeConfig,
    id: string
  ): TreeNodeConfig | null => {
    if (node.id === id) return node;
    for (const child of node.children) {
      const found = findNode(child, id);
      if (found) return found;
    }
    return null;
  };

  const selectedNode =
    selectedNodeId && treeConfig
      ? findNode(treeConfig.tree, selectedNodeId)
      : null;

  const hasTree = treeConfig && treeConfig.tree.children.length > 0;

  return (
    <div className="h-screen flex flex-col bg-slate-950 text-slate-100">
      {/* Header */}
      <header className="flex items-center justify-between px-5 py-3 border-b border-slate-800 bg-slate-900/80 backdrop-blur-sm">
        <div className="flex items-center gap-3">
          <TreesIcon className="w-6 h-6 text-emerald-400" />
          <h1 className="text-lg font-semibold tracking-tight">AGI Tree</h1>
          <span
            className={`ml-2 inline-block w-2 h-2 rounded-full ${
              isConnected ? "bg-emerald-400" : "bg-red-400"
            }`}
          />
        </div>
        <div className="flex items-center gap-3">
          {uploadedFiles.length > 0 && (
            <div className="flex items-center gap-1.5">
              {uploadedFiles.map((f, i) => (
                <span
                  key={i}
                  className="flex items-center gap-1 text-xs bg-slate-800 px-2.5 py-1.5 rounded-full text-slate-300"
                >
                  <FileText className="w-3 h-3 text-blue-400" />
                  {f}
                </span>
              ))}
              <button
                onClick={handleClearAll}
                className="ml-1 p-1 hover:text-red-400 transition-colors text-slate-500"
                title="Rimuovi tutti i documenti e resetta l'albero"
              >
                <Trash2 className="w-3.5 h-3.5" />
              </button>
            </div>
          )}
          <input
            ref={fileInputRef}
            type="file"
            accept=".pdf,.txt,.md,.csv"
            className="hidden"
            onChange={handleUpload}
          />
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={isUploading}
            className="flex items-center gap-1.5 text-xs bg-slate-800 hover:bg-slate-700 px-3 py-1.5 rounded-lg transition-colors disabled:opacity-50"
          >
            {isUploading ? (
              <Loader2 className="w-3.5 h-3.5 animate-spin" />
            ) : (
              <Upload className="w-3.5 h-3.5" />
            )}
            {isUploading ? "Analisi in corso..." : "Carica documento"}
          </button>
        </div>
      </header>

      {/* Main content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left: Chat */}
        <div className="w-[420px] min-w-[360px] border-r border-slate-800 flex flex-col">
          {!hasTree ? (
            <div className="flex-1 flex flex-col items-center justify-center p-8 text-center">
              <TreesIcon className="w-16 h-16 text-slate-700 mb-4" />
              <h2 className="text-lg font-semibold text-slate-400 mb-2">
                Carica un documento per iniziare
              </h2>
              <p className="text-sm text-slate-500 mb-6 max-w-xs">
                L'albero di conoscenza verrà costruito automaticamente
                analizzando gli argomenti del documento.
              </p>
              <button
                onClick={() => fileInputRef.current?.click()}
                disabled={isUploading}
                className="flex items-center gap-2 px-5 py-2.5 bg-emerald-600 hover:bg-emerald-500 rounded-lg transition-colors text-sm font-medium disabled:opacity-50"
              >
                {isUploading ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <Upload className="w-4 h-4" />
                )}
                {isUploading ? "Analisi in corso..." : "Carica documento"}
              </button>
            </div>
          ) : (
            <ChatPanel
              messages={messages}
              isProcessing={isProcessing}
              onSend={sendQuery}
            />
          )}
        </div>

        {/* Right: Tree + Details */}
        <div className="flex-1 flex flex-col overflow-hidden">
          {/* Tabs */}
          <div className="flex border-b border-slate-800">
            <button
              onClick={() => setRightTab("tree")}
              className={`flex items-center gap-1.5 px-4 py-2.5 text-sm font-medium transition-colors ${
                rightTab === "tree"
                  ? "text-emerald-400 border-b-2 border-emerald-400"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              <TreesIcon className="w-4 h-4" />
              Albero
            </button>
            <button
              onClick={() => setRightTab("log")}
              className={`flex items-center gap-1.5 px-4 py-2.5 text-sm font-medium transition-colors ${
                rightTab === "log"
                  ? "text-emerald-400 border-b-2 border-emerald-400"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              <Activity className="w-4 h-4" />
              Log
              {logEntries.length > 0 && (
                <span className="ml-1 text-xs bg-slate-700 px-1.5 py-0.5 rounded-full">
                  {logEntries.length}
                </span>
              )}
            </button>
          </div>

          {rightTab === "tree" ? (
            <div className="flex-1 flex overflow-hidden">
              <div className="flex-1 overflow-auto p-6">
                {hasTree ? (
                  <TreeView
                    tree={treeConfig.tree}
                    nodeStates={nodeStates}
                    selectedNodeId={selectedNodeId}
                    onSelectNode={setSelectedNodeId}
                  />
                ) : (
                  <div className="flex flex-col items-center justify-center h-full text-slate-600">
                    <TreesIcon className="w-20 h-20 mb-3 opacity-30" />
                    <p className="text-sm">
                      {isUploading
                        ? "Costruzione albero in corso..."
                        : "L'albero apparirà dopo il caricamento di un documento"}
                    </p>
                    {isUploading && (
                      <Loader2 className="w-5 h-5 animate-spin mt-3 text-emerald-500" />
                    )}
                  </div>
                )}
              </div>
              {selectedNode && (
                <NodeDetail
                  node={selectedNode}
                  state={nodeStates[selectedNode.id]}
                  onClose={() => setSelectedNodeId(null)}
                />
              )}
            </div>
          ) : (
            <div className="flex-1 overflow-auto p-4 space-y-1.5">
              {logEntries.length === 0 && (
                <p className="text-slate-500 text-sm text-center mt-8">
                  Carica un documento o invia una domanda per vedere il log.
                </p>
              )}
              {logEntries.map((entry) => (
                <LogItem key={entry.id} entry={entry} />
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function LogItem({ entry }: { entry: LogEntry }) {
  const phaseColors: Record<string, string> = {
    analysis: "text-cyan-400 bg-cyan-400/10",
    building: "text-orange-400 bg-orange-400/10",
    discovery: "text-amber-400 bg-amber-400/10",
    selection: "text-emerald-400 bg-emerald-400/10",
    routing: "text-blue-400 bg-blue-400/10",
    answering: "text-purple-400 bg-purple-400/10",
    synthesis: "text-pink-400 bg-pink-400/10",
  };

  const color = phaseColors[entry.phase] || "text-slate-400 bg-slate-400/10";

  let detail = "";
  if (entry.status === "score" && typeof entry.data === "number") {
    detail = `→ ${entry.data.toFixed(2)}`;
  } else if (entry.status === "selected" && Array.isArray(entry.data)) {
    detail = `→ [${entry.data.join(", ")}]`;
  } else if (entry.status === "decomposed" && Array.isArray(entry.data)) {
    detail = `→ ${entry.data.length} sub-queries`;
  } else if (entry.status === "topics_found" && Array.isArray(entry.data)) {
    detail = `→ ${entry.data.join(", ")}`;
  } else if (entry.status === "created") {
    detail = "→ nodo creato";
  }

  return (
    <div className="flex items-center gap-2 text-xs font-mono">
      <span className={`px-1.5 py-0.5 rounded text-[10px] uppercase font-semibold ${color}`}>
        {entry.phase}
      </span>
      <span className="text-slate-300">{entry.nodeName}</span>
      <span className="text-slate-500">{entry.status}</span>
      {detail && <span className="text-slate-400">{detail}</span>}
    </div>
  );
}
