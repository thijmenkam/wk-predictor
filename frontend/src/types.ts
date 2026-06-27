export type Match = {
  match_id: string | number;
  group: string;
  match_round: number;
  kickoff_at: string | null;
  location: string | null;
  team_a: string;
  team_b: string;
  elo_a: number;
  elo_b: number;
  lambda_a: number;
  lambda_b: number;
  p_win_a: number;
  p_draw: number;
  p_win_b: number;
  most_likely_score: string;
  recommended_score: string;
  expected_pool_points: number;
  strategy: string;
  recommendation_reason: string | null;
  best_ev_score?: string;
  best_ev?: number;
  recommended_ev?: number;
  ev_loss_vs_best?: number;
  score_selection_strategy?: string;
  candidate_scores_within_tolerance?: number;
  selection_reason?: string | null;
  realism_score?: number;
  score_rank_by_ev?: number;
  fixture_id?: string;
  recommended_goals_a?: number;
  recommended_goals_b?: number;
  recommendation?: {
    score: string;
    goals_a: number;
    goals_b: number;
    expected_pool_points: number;
    source: string;
    score_probability_source: string;
    selection_strategy: string | null;
    selection_reason: string | null;
    best_ev_score?: string;
    best_ev?: number;
    recommended_ev?: number;
    ev_loss_vs_best?: number;
    realism_score?: number;
    score_rank_by_ev?: number;
  };
  model?: ProbabilityView & {
    lambda_a: number;
    lambda_b: number;
    recommended_score: string;
    expected_pool_points: number;
  };
  market_1x2?: ProbabilityView & {
    available: boolean;
    confidence: string | null;
    raw_p_win_a: number | null;
    raw_p_draw: number | null;
    raw_p_win_b: number | null;
    source_used: string | null;
    market_slug: string | null;
  };
  hybrid_1x2?: ProbabilityView & {
    available: boolean;
    market_weight: number | null;
    source_used: string | null;
  };
  market_delta?: {
    home: number | null;
    draw: number | null;
    away: number | null;
    largest_outcome: "home" | "draw" | "away" | null;
    largest_abs_delta: number | null;
  };
  exact_score_market?: {
    available: boolean;
    score_probability_source: string;
    market_score_weight: number | null;
    scores_count: number;
    raw_probability_sum: number | null;
    top_scores: ExactScore[];
  };
  score_recommendations?: {
    final: { score: string; source: string };
  };
  warnings?: string[];
};

export type KnockoutMatch = {
  match_id: string;
  stage: string;
  slot_a: string;
  slot_b: string;
  team_a: string | null;
  team_b: string | null;
  label_a: string;
  label_b: string;
  odds_source: string | null;
  lambda_a: number | null;
  lambda_b: number | null;
  p_win_a: number | null;
  p_draw: number | null;
  p_win_b: number | null;
  p_advance_a: number | null;
  p_advance_b: number | null;
  most_likely_score: string | null;
};

type ProbabilityView = {
  p_win_a: number | null;
  p_draw: number | null;
  p_win_b: number | null;
};

export type ExactScore = {
  score: string;
  goals_a: number;
  goals_b: number;
  raw_probability: number | null;
  normalized_probability: number | null;
  confidence: string | null;
  market_slug: string | null;
};

export type Team = {
  team: string;
  group: string;
  elo: number;
  p_round_of_32: number;
  p_round_of_16: number;
  p_quarter_final: number;
  p_semi_final: number;
  p_final: number;
  p_champion: number;
  p_top4: number;
};

export type TopScorer = {
  player: string;
  team: string;
  position: string | null;
  expected_goals: number;
  p_top_scorer: number;
  p_top_3_goals: number;
  recommended_score_value: number;
  is_recommended: boolean;
};

export type FinalStandingRow = {
  position: string;
  team: string;
  elo: number;
  p_top4: number;
  p_exact_position: number;
  expected_points_component_marginal: number;
  ev_method: string;
};

export type FinalStandingCandidate = Record<string, string | number | null>;

export type FrontendData = {
  schema_version?: string;
  generated_at?: string;
  source_run_dir?: string;
  metadata: Record<string, unknown> & {
    seed?: number;
    num_simulations?: number;
    limitations?: string[];
  };
  coverage?: {
    moneyline: CoverageValue;
    exact_score: CoverageValue;
    model_fallback: { count: number };
    source_used_counts: Record<string, number>;
  };
  round_1_predictions?: Match[];
  matches: Match[];
  round_of_32?: KnockoutMatch[];
  teams: Team[];
  top_scorers: TopScorer[];
  final_standings: {
    gold: string;
    silver: string;
    bronze: string;
    fourth: string;
    recommendation: FinalStandingRow[];
    candidates: FinalStandingCandidate[];
  };
  market_comparison: Record<string, unknown>[];
  warnings?: string[];
};

type CoverageValue = {
  available: number;
  total: number;
  coverage_pct: number;
};
