export type Decision = "accepted" | "rejected" | "invalid" | null;

export interface Page<T> {
  items: T[];
  total: number;
  limit: number;
  offset: number;
}

export interface Run {
  id: string;
  namespace: string;
  objective: string;
  state: string;
  memory_enabled: boolean;
  planner_mode: string;
  seed: number;
  max_experiments: number;
  max_duration_seconds: number;
  max_retries: number;
  stop_requested: boolean;
  parent_run_id: string | null;
  start_baseline_experiment_id: string | null;
  end_baseline_experiment_id: string | null;
  stop_reason: string | null;
  error: string | null;
  initial_configuration?: Record<string, unknown>;
  summary: Record<string, unknown> | null;
  created_at: string;
  finished_at: string | null;
}

export interface Experiment {
  id: string;
  run_id: string;
  sequence: number;
  parent_experiment_id: string | null;
  baseline_experiment_id: string | null;
  proposed_by: string;
  hypothesis: string;
  expected_effects: { metric: string; direction: string }[];
  reasoning_summary: string;
  plan_source: string;
  referenced_memory_ids: string[];
  model_type: string;
  configuration: Record<string, unknown>;
  config_diff: Record<string, { from: unknown; to: unknown }>;
  moves: string[];
  status: string;
  attempts: number;
  error: string | null;
  decision: Decision;
  decision_reason: string | null;
  primary_delta: number | null;
  secondary_deltas: Record<string, { candidate: number; baseline: number; delta: number; relative: number }>;
  acceptance_threshold: number;
  hypothesis_outcome: string | null;
  hypothesis_details: Record<string, unknown>[];
  git_commit: string | null;
  dataset_version: string;
  protocol_version: string;
  created_at: string;
  metrics: Record<string, number>;
}

export interface Overview {
  total_runs: number;
  total_experiments: number;
  accepted: number;
  rejected: number;
  invalid: number;
  best_validation_f1: number | null;
  current_baseline: Experiment | null;
  latest_run: Run | null;
  total_memory_records: number;
  busy: boolean;
  f1_series: { index: number; experiment_id: string; f1: number | null; decision: Decision; model_type: string }[];
}

export interface MemoryRecord {
  id: string;
  category: "episodic" | "semantic" | "procedural";
  kind: string;
  title: string;
  content: string;
  source_run_id: string | null;
  source_experiment_id: string | null;
  finding_id: string | null;
  created_by: string;
  evidence: Record<string, unknown>;
  verification_status: "verified" | "unverified" | "contradicted";
  tags: string[];
  created_at: string;
}

export interface SearchResult {
  memory_id: string;
  category: string;
  kind: string;
  title: string;
  content: string;
  verification_status: string;
  source_experiment_id: string | null;
  score: number;
  score_components: Record<string, number>;
  reason: string;
  evidence: Record<string, unknown>;
}

export interface SearchResponse {
  parsed_query: Record<string, unknown>;
  ranking_formula: string;
  results: SearchResult[];
  skipped: { memory_id: string; reason: string }[];
  latency_ms: number;
}

export interface GraphNode {
  id: string;
  type: string;
  label: string;
  [key: string]: unknown;
}

export interface GraphEdge {
  id: string;
  source: string;
  target: string;
  relation: string;
  evidence_ref: string;
}

export interface GraphResponse {
  nodes: GraphNode[];
  edges: GraphEdge[];
  dangling: unknown[];
  run_id: string | null;
  stats: { nodes: number; edges: number; dangling: number };
}

export interface AgentInfo {
  id: string;
  name: string;
  role: string;
  description: string;
  tools: string[];
  action_count: number;
}

export interface AgentAction {
  id: string;
  run_id: string | null;
  experiment_id: string | null;
  action_type: string;
  status: string;
  output: Record<string, unknown> | null;
  error: string | null;
  retrieved_memory_ids: string[];
  retrieval_count: number;
  retrieval_ms: number;
  duration_ms: number;
  created_at: string;
}

export interface AgentMessage {
  id: string;
  sender: string;
  recipient: string;
  message_type: string;
  experiment_id: string | null;
  payload: Record<string, unknown>;
  created_at: string;
}

export interface EvaluationRow {
  id: string;
  status: string;
  params: Record<string, unknown>;
  results: {
    trials: Record<string, unknown>[];
    aggregate: Record<string, Record<string, number | null>>;
    note?: string;
  } | null;
  error: string | null;
  created_at: string;
}
