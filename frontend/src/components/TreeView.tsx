import { TreeNodeConfig, NodeState, NodeVisualState } from "../types";
import { Crown, GitBranch, Leaf } from "lucide-react";

interface Props {
  tree: TreeNodeConfig;
  nodeStates: Record<string, NodeState>;
  selectedNodeId: string | null;
  onSelectNode: (id: string) => void;
}

export function TreeView({ tree, nodeStates, selectedNodeId, onSelectNode }: Props) {
  return (
    <div className="flex justify-center min-w-fit">
      <TreeNodeComponent
        node={tree}
        nodeStates={nodeStates}
        selectedNodeId={selectedNodeId}
        onSelectNode={onSelectNode}
      />
    </div>
  );
}

interface NodeProps {
  node: TreeNodeConfig;
  nodeStates: Record<string, NodeState>;
  selectedNodeId: string | null;
  onSelectNode: (id: string) => void;
}

function TreeNodeComponent({ node, nodeStates, selectedNodeId, onSelectNode }: NodeProps) {
  const state = nodeStates[node.id];
  const isSelected = selectedNodeId === node.id;

  return (
    <div className="flex flex-col items-center">
      {/* Node card */}
      <button
        onClick={() => onSelectNode(node.id)}
        className={`
          relative px-4 py-3 rounded-xl border-2 transition-all duration-300 cursor-pointer
          min-w-[140px] text-left
          ${getNodeStyle(node.role, state?.visualState, isSelected)}
        `}
      >
        {/* Score badge */}
        {state?.score !== undefined && (
          <span
            className={`absolute -top-2.5 -right-2.5 text-[10px] font-bold px-1.5 py-0.5 rounded-full
              ${
                state.score >= 0.7
                  ? "bg-emerald-500 text-white"
                  : state.score >= 0.4
                  ? "bg-amber-500 text-white"
                  : "bg-slate-600 text-slate-300"
              }`}
          >
            {state.score.toFixed(2)}
          </span>
        )}

        <div className="flex items-center gap-2 mb-1">
          <RoleIcon role={node.role} />
          <span className="text-xs font-semibold truncate">{node.name}</span>
        </div>

        <div className="flex items-center gap-1.5">
          <span
            className={`text-[9px] uppercase font-bold px-1.5 py-0.5 rounded
              ${getRoleBadgeColor(node.role)}`}
          >
            {node.role}
          </span>
        </div>

        {/* Processing indicator */}
        {(state?.visualState === "scoring" || state?.visualState === "answering") && (
          <div className="absolute inset-0 rounded-xl border-2 border-transparent animate-pulse-slow pointer-events-none" />
        )}
      </button>

      {/* Children */}
      {node.children.length > 0 && (
        <div className="flex flex-col items-center">
          {/* Vertical connector from parent */}
          <div className="w-px h-8 bg-slate-600/60" />

          {/* Children row */}
          <div className="relative flex">
            {/* Horizontal connector line */}
            {node.children.length > 1 && (
              <div
                className="absolute top-0 h-px bg-slate-600/60"
                style={{
                  left: `calc(${100 / (node.children.length * 2)}%)`,
                  right: `calc(${100 / (node.children.length * 2)}%)`,
                }}
              />
            )}

            {node.children.map((child, i) => (
              <div key={child.id} className="flex flex-col items-center px-4">
                {/* Vertical connector to horizontal line */}
                <div className="w-px h-8 bg-slate-600/60" />
                <TreeNodeComponent
                  node={child}
                  nodeStates={nodeStates}
                  selectedNodeId={selectedNodeId}
                  onSelectNode={onSelectNode}
                />
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function getNodeStyle(
  role: string,
  visualState?: NodeVisualState,
  isSelected?: boolean
): string {
  const base = "bg-slate-800/80 backdrop-blur-sm";

  if (isSelected) {
    return `${base} border-white/60 ring-2 ring-white/20`;
  }

  switch (visualState) {
    case "scoring":
      return `${base} border-amber-400/80 glow-amber animate-pulse-slow`;
    case "scored":
      return `${base} border-amber-400/50`;
    case "selected":
      return `${base} border-emerald-400/80 glow-emerald`;
    case "answering":
      return `${base} border-blue-400/80 glow-blue animate-pulse-slow`;
    case "complete":
      return `${base} border-emerald-400/60 glow-emerald`;
    default:
      break;
  }

  switch (role) {
    case "root":
      return `${base} border-emerald-500/40`;
    case "node":
      return `${base} border-blue-500/30`;
    case "leaf":
      return `${base} border-purple-500/30`;
    default:
      return `${base} border-slate-700`;
  }
}

function getRoleBadgeColor(role: string): string {
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

function RoleIcon({ role }: { role: string }) {
  const cls = "w-3.5 h-3.5";
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
