import { X, Crown, GitBranch, Leaf, Cpu, FileCode } from "lucide-react";
import { TreeNodeConfig, NodeState } from "../types";

interface Props {
  node: TreeNodeConfig;
  state?: NodeState;
  onClose: () => void;
}

export function NodeDetail({ node, state, onClose }: Props) {
  return (
    <div className="w-80 border-l border-slate-800 bg-slate-900/50 flex flex-col overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-slate-800">
        <div className="flex items-center gap-2">
          <RoleIcon role={node.role} />
          <h3 className="font-semibold text-sm">{node.name}</h3>
        </div>
        <button
          onClick={onClose}
          className="text-slate-400 hover:text-white transition-colors"
        >
          <X className="w-4 h-4" />
        </button>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {/* ID */}
        <Field label="ID" value={node.id} />

        {/* Role */}
        <div>
          <label className="text-[10px] uppercase font-bold text-slate-500 tracking-wider">
            Ruolo
          </label>
          <div className="mt-1">
            <span
              className={`text-xs font-semibold px-2 py-1 rounded ${getRoleBadge(
                node.role
              )}`}
            >
              {node.role === "root"
                ? "Radice"
                : node.role === "node"
                ? "Nodo"
                : "Foglia"}
            </span>
          </div>
        </div>

        {/* Model */}
        <div>
          <label className="text-[10px] uppercase font-bold text-slate-500 tracking-wider">
            Modello
          </label>
          <div className="mt-1 flex items-center gap-1.5 text-sm text-slate-300">
            <Cpu className="w-3.5 h-3.5 text-slate-500" />
            <span className="font-mono text-xs">{node.model}</span>
          </div>
        </div>

        {/* Score */}
        {state?.score !== undefined && (
          <div>
            <label className="text-[10px] uppercase font-bold text-slate-500 tracking-wider">
              Score competenza
            </label>
            <div className="mt-1.5">
              <div className="flex items-center gap-2">
                <div className="flex-1 h-2 bg-slate-800 rounded-full overflow-hidden">
                  <div
                    className={`h-full rounded-full transition-all duration-500 ${
                      state.score >= 0.7
                        ? "bg-emerald-500"
                        : state.score >= 0.4
                        ? "bg-amber-500"
                        : "bg-red-500"
                    }`}
                    style={{ width: `${state.score * 100}%` }}
                  />
                </div>
                <span className="text-sm font-bold text-slate-200">
                  {state.score.toFixed(2)}
                </span>
              </div>
            </div>
          </div>
        )}

        {/* State */}
        {state?.visualState && state.visualState !== "idle" && (
          <div>
            <label className="text-[10px] uppercase font-bold text-slate-500 tracking-wider">
              Stato
            </label>
            <div className="mt-1">
              <span
                className={`text-xs font-semibold px-2 py-1 rounded ${getStateBadge(
                  state.visualState
                )}`}
              >
                {stateLabels[state.visualState] || state.visualState}
              </span>
            </div>
          </div>
        )}

        {/* System Prompt */}
        <div>
          <label className="text-[10px] uppercase font-bold text-slate-500 tracking-wider flex items-center gap-1">
            <FileCode className="w-3 h-3" />
            System Prompt
          </label>
          <div className="mt-1.5 bg-slate-800/80 rounded-lg p-3 text-xs text-slate-300 leading-relaxed max-h-48 overflow-y-auto">
            {node.system_prompt}
          </div>
        </div>

        {/* Children count */}
        {node.children.length > 0 && (
          <Field
            label="Figli"
            value={`${node.children.length} (${node.children
              .map((c) => c.name)
              .join(", ")})`}
          />
        )}
      </div>
    </div>
  );
}

function Field({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <label className="text-[10px] uppercase font-bold text-slate-500 tracking-wider">
        {label}
      </label>
      <p className="mt-0.5 text-sm text-slate-300 font-mono">{value}</p>
    </div>
  );
}

function RoleIcon({ role }: { role: string }) {
  const cls = "w-4 h-4";
  switch (role) {
    case "root":
      return <Crown className={`${cls} text-emerald-400`} />;
    case "node":
      return <GitBranch className={`${cls} text-blue-400`} />;
    case "leaf":
      return <Leaf className={`${cls} text-purple-400`} />;
    default:
      return null;
  }
}

function getRoleBadge(role: string): string {
  switch (role) {
    case "root":
      return "bg-emerald-500/20 text-emerald-400";
    case "node":
      return "bg-blue-500/20 text-blue-400";
    case "leaf":
      return "bg-purple-500/20 text-purple-400";
    default:
      return "bg-slate-700 text-slate-400";
  }
}

function getStateBadge(state: string): string {
  switch (state) {
    case "scoring":
      return "bg-amber-500/20 text-amber-400";
    case "scored":
      return "bg-amber-500/10 text-amber-300";
    case "selected":
      return "bg-emerald-500/20 text-emerald-400";
    case "answering":
      return "bg-blue-500/20 text-blue-400";
    case "complete":
      return "bg-emerald-500/20 text-emerald-400";
    default:
      return "bg-slate-700 text-slate-400";
  }
}

const stateLabels: Record<string, string> = {
  scoring: "Valutazione in corso...",
  scored: "Valutato",
  selected: "Selezionato",
  answering: "Risposta in corso...",
  complete: "Completato",
};
