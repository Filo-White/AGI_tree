export interface TreeNodeConfig {
  id: string;
  name: string;
  model: string;
  role: "root" | "node" | "leaf";
  system_prompt: string;
  children: TreeNodeConfig[];
}

export interface TreeConfig {
  tree: TreeNodeConfig;
}

export type NodeVisualState =
  | "idle"
  | "scoring"
  | "scored"
  | "selected"
  | "answering"
  | "complete";

export interface NodeState {
  visualState: NodeVisualState;
  score?: number;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  timestamp: number;
  metadata?: {
    scores?: Record<string, number>;
    selectedLeaves?: string[];
    subQueries?: string[];
    leafResponses?: LeafResponseData[];
  };
}

export interface LeafResponseData {
  node_id: string;
  node_name: string;
  response: string;
  score: number;
}

export interface LogEntry {
  id: string;
  phase: string;
  nodeId: string;
  nodeName: string;
  status: string;
  data?: any;
  timestamp: number;
}

export interface ProgressMessage {
  type: "progress";
  phase: string;
  node_id: string;
  node_name: string;
  status: string;
  data?: any;
}

export interface ResultMessage {
  type: "result";
  response: string;
  scores: Record<string, number>;
  selected_leaves: string[];
  sub_queries: string[];
  leaf_responses: LeafResponseData[];
}

export interface ErrorMessage {
  type: "error";
  message: string;
}

export type WSMessage = ProgressMessage | ResultMessage | ErrorMessage;
