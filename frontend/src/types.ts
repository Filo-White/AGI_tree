export interface TreeNodeConfig {
  id: string;
  name: string;
  model: string;
  role: "root" | "node" | "leaf";
  system_prompt: string;
  context: string;
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
  | "complete"
  | "building"
  | "expanding";

export interface NodeState {
  visualState: NodeVisualState;
  score?: number;
  reason?: string;
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

export interface TreeUpdateMessage {
  type: "tree_update";
  tree: TreeNodeConfig;
}

export interface ErrorMessage {
  type: "error";
  message: string;
}

export type WSMessage = ProgressMessage | ResultMessage | TreeUpdateMessage | ErrorMessage;

// Processing log types
export interface NodeLog {
  name: string;
  char_count: number;
  expanded: boolean;
}

export interface DocumentLog {
  filename: string;
  doc_type: string;
  detection_method: string;
  nodes: NodeLog[];
}

export interface ProcessingLog {
  documents: DocumentLog[];
  total_nodes: number;
  total_leaves: number;
  doc_type: string | null;
  doc_description: string | null;
}
