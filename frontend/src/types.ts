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

export interface ChatRequest {
  message: string;
  history: ChatTurn[];
  model?: string;
}

export interface DoneEvent {
  content: string;
  model: string;
  artifacts: Artifact[];
  usage: Record<string, number>;
}
