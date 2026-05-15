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
    <div className="flex flex-col items-center min-w-fit py-4">
      {/* Root circle */}
      <CircleNode
        node={tree}
        state={nodeStates[tree.id]}
        isSelected={selectedNodeId === tree.id}
        onClick={() => onSelectNode(tree.id)}
        size="lg"
      />

      {/* Chapter level */}
      {tree.children.length > 0 && (
        <>
          <div className="w-px h-6 bg-slate-600/50" />
          <div className="relative flex items-start">
            {/* Horizontal line across all chapters */}
            {tree.children.length > 1 && (
              <div
                className="absolute top-0 h-px bg-slate-600/50"
                style={{
                  left: `calc(${100 / (tree.children.length * 2)}% + 1px)`,
                  right: `calc(${100 / (tree.children.length * 2)}% + 1px)`,
                }}
              />
            )}

            {tree.children.map((chapter) => (
              <div key={chapter.id} className="flex flex-col items-center px-3">
                {/* Connector down from h-line to chapter */}
                <div className="w-px h-5 bg-slate-600/50" />

                <CircleNode
                  node={chapter}
                  state={nodeStates[chapter.id]}
                  isSelected={selectedNodeId === chapter.id}
                  onClick={() => onSelectNode(chapter.id)}
                  size="md"
                />

                {/* Section leaves below each chapter */}
                {chapter.children.length > 0 && (
                  <div className="flex flex-col items-center mt-1">
                    <div className="w-px h-3 bg-slate-600/30" />
                    <div className="flex flex-col gap-1">
                      {chapter.children.map((section) => (
                        <div key={section.id} className="flex items-center gap-1.5">
                          <div className="w-3 h-px bg-slate-600/30" />
                          <CircleNode
                            node={section}
                            state={nodeStates[section.id]}
                            isSelected={selectedNodeId === section.id}
                            onClick={() => onSelectNode(section.id)}
                            size="sm"
                          />
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ */

interface CircleProps {
  node: TreeNodeConfig;
  state?: NodeState;
  isSelected: boolean;
  onClick: () => void;
  size: "lg" | "md" | "sm";
}

const sizeMap = {
  lg: { circle: "w-[72px] h-[72px]", icon: "w-7 h-7", label: "text-[11px] max-w-[90px]", badge: "w-5 h-5 text-[9px] -top-1 -right-1" },
  md: { circle: "w-[52px] h-[52px]", icon: "w-4 h-4", label: "text-[10px] max-w-[76px]", badge: "w-4 h-4 text-[8px] -top-0.5 -right-0.5" },
  sm: { circle: "w-[36px] h-[36px]", icon: "w-3 h-3", label: "text-[9px] max-w-[68px]", badge: "w-4 h-4 text-[8px] -top-0.5 -right-0.5" },
};

function CircleNode({ node, state, isSelected, onClick, size }: CircleProps) {
  const s = sizeMap[size];
  const borderStyle = getCircleStyle(node.role, state?.visualState, isSelected);
  const isAnimating = state?.visualState === "scoring" || state?.visualState === "answering" || state?.visualState === "building" || state?.visualState === "expanding";

  return (
    <div className="flex flex-col items-center gap-0.5">
      <button
        onClick={onClick}
        title={node.name}
        className={`
          relative rounded-full border-2 flex items-center justify-center
          transition-all duration-300 cursor-pointer shrink-0
          ${s.circle} ${borderStyle}
          ${isAnimating ? "animate-pulse-slow" : ""}
        `}
      >
        <RoleIcon role={node.role} className={s.icon} />

        {/* Score badge */}
        {state?.score !== undefined && (
          <span
            className={`absolute flex items-center justify-center rounded-full font-bold ${s.badge}
              ${state.score >= 0.7 ? "bg-emerald-500 text-white"
                : state.score >= 0.4 ? "bg-amber-500 text-white"
                : "bg-slate-600 text-slate-300"}`}
          >
            {state.score.toFixed(1)}
          </span>
        )}
      </button>
      <span className={`text-center leading-tight text-slate-400 ${s.label} line-clamp-2`}>
        {node.name}
      </span>
    </div>
  );
}

/* ------------------------------------------------------------------ */

function getCircleStyle(role: string, visualState?: NodeVisualState, isSelected?: boolean): string {
  const base = "bg-slate-800/90 backdrop-blur-sm";

  if (isSelected) return `${base} border-white/70 ring-2 ring-white/20 shadow-lg`;

  switch (visualState) {
    case "building":   return `${base} border-orange-400/80 shadow-orange-500/20 shadow-md`;
    case "expanding":  return `${base} border-cyan-400/80 shadow-cyan-500/20 shadow-md`;
    case "scoring":    return `${base} border-amber-400/80 shadow-amber-500/20 shadow-md`;
    case "scored":     return `${base} border-amber-400/50`;
    case "selected":   return `${base} border-emerald-400/80 shadow-emerald-500/20 shadow-md`;
    case "answering":  return `${base} border-blue-400/80 shadow-blue-500/20 shadow-md`;
    case "complete":   return `${base} border-emerald-400/60 shadow-emerald-500/20 shadow-sm`;
  }

  switch (role) {
    case "root": return `${base} border-emerald-500/50`;
    case "node": return `${base} border-blue-500/40`;
    case "leaf": return `${base} border-purple-500/40`;
    default:     return `${base} border-slate-700`;
  }
}

function RoleIcon({ role, className }: { role: string; className?: string }) {
  const cls = className || "w-4 h-4";
  switch (role) {
    case "root": return <Crown className={`${cls} text-emerald-400`} />;
    case "node": return <GitBranch className={`${cls} text-blue-400`} />;
    case "leaf": return <Leaf className={`${cls} text-purple-400`} />;
    default:     return null;
  }
}
