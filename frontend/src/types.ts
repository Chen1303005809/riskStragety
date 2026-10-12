export type ChatRole = 'user' | 'assistant';

export interface ChatTurn {
  role: ChatRole;
  content: string;
}

export interface Artifact {
  artifact_id: string;
  filename: string;
}

export interface ChatMessage extends ChatTurn {
  id: string;
  status: 'complete' | 'streaming' | 'error' | 'stopped';
  artifacts: Artifact[];
}

export interface ModelOptions {
  temperature: number;
  top_p: number;
  num_ctx: number;
}

export interface ChatRequest {
  account_id: string;
  conversation_id: string;
  message: string;
  /** Accepted for source compatibility; streamChat intentionally omits this legacy field. */
  history?: ChatTurn[];
  model?: string;
  options: ModelOptions;
  think: boolean;
}

export interface DoneEvent {
  content: string;
  model: string;
  artifacts: Artifact[];
  usage: Record<string, number>;
}

export interface Page<T> {
  items: T[];
  offset: number;
  limit: number;
  total: number;
}

export interface AccountSummary {
  account_id: string;
  created_at: string;
  updated_at: string;
  conversation_count: number;
  message_count: number;
  artifact_count: number;
}

export interface ConversationSummary {
  conversation_id: string;
  title: string;
  status: 'idle' | 'active' | 'completed' | 'failed' | 'interrupted';
  created_at: string;
  updated_at: string;
  message_count: number;
  artifact_count: number;
}

export interface ArtifactRecord extends Artifact {
  created_at: string;
}

export interface MonitorConversationsPage extends Page<ConversationSummary> {
  unassigned_artifacts: ArtifactRecord[];
  unassigned_offset: number;
  unassigned_limit: number;
  unassigned_total: number;
}

export interface MonitorMessage extends ChatTurn {
  id: number;
  sequence: number;
  status: 'pending' | 'complete' | 'failed' | 'interrupted';
  created_at: string;
  completed_at: string | null;
}

export interface ConversationDetail {
  conversation: ConversationSummary & { account_id: string };
  messages: MonitorMessage[];
  message_offset: number;
  message_limit: number;
  message_total: number;
  artifacts: ArtifactRecord[];
  artifact_offset: number;
  artifact_limit: number;
  artifact_total: number;
}
