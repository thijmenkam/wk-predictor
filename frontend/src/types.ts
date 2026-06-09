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
  metadata: Record<string, unknown> & {
    seed?: number;
    num_simulations?: number;
    limitations?: string[];
  };
  matches: Match[];
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
};
