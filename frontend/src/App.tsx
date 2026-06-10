import { useEffect, useMemo, useState } from "react";
import type { FrontendData, Match, Team, TopScorer } from "./types";

type Tab = "basic" | "matches" | "teams" | "scorers" | "market";
type TeamSort = "p_champion" | "p_top4" | "p_final";
type ScorerSort = "expected_goals" | "recommended_score_value";

const percentage = new Intl.NumberFormat("en", {
  style: "percent",
  maximumFractionDigits: 1,
});
const decimal = new Intl.NumberFormat("en", { maximumFractionDigits: 2 });

function pct(value: number) {
  return percentage.format(value);
}

function optionalPct(value: number | null | undefined) {
  return value == null ? "—" : pct(value);
}

function formatKickoff(value: string | null) {
  if (!value) return "TBD";
  const date = new Date(value);
  return Number.isNaN(date.getTime())
    ? value
    : new Intl.DateTimeFormat("en", {
        day: "2-digit",
        month: "short",
        hour: "2-digit",
        minute: "2-digit",
      }).format(date);
}

function TableShell({ children }: { children: React.ReactNode }) {
  return <div className="table-shell">{children}</div>;
}

function EmptyState({ children }: { children: React.ReactNode }) {
  return <div className="empty-state">{children}</div>;
}

function sourceLabel(match: Match) {
  const recommendationSource = match.recommendation?.source;
  const scoreSource = match.score_recommendations?.final.source;
  const probabilitySource = match.hybrid_1x2?.source_used ?? match.market_1x2?.source_used;
  const source = recommendationSource ?? probabilitySource ?? scoreSource ?? "model";
  if (source.startsWith("model_fallback")) {
    return ["Model fallback", "fallback"];
  }
  if (source === "market" || source === "market_only" || scoreSource === "market_exact_score") {
    return ["Market", "market"];
  }
  if (source === "hybrid" || scoreSource === "hybrid_exact_score") {
    return ["Hybrid", "hybrid"];
  }
  return ["Model", "model"];
}

function ProbabilityLine({ match, values }: { match: Match; values: { p_win_a: number | null; p_draw: number | null; p_win_b: number | null } }) {
  return <div className="probability-line"><span>{match.team_a} <b>{optionalPct(values.p_win_a)}</b></span><span>Draw <b>{optionalPct(values.p_draw)}</b></span><span>{match.team_b} <b>{optionalPct(values.p_win_b)}</b></span></div>;
}

function MatchesTable({ matches, showExactScore }: { matches: Match[]; showExactScore: boolean }) {
  return (
    <div className="match-list">
      {matches.map((match) => {
        const [label, kind] = sourceLabel(match);
        const model = match.model ?? { lambda_a: match.lambda_a, lambda_b: match.lambda_b, p_win_a: match.p_win_a, p_draw: match.p_draw, p_win_b: match.p_win_b, recommended_score: match.most_likely_score, expected_pool_points: match.expected_pool_points };
        const market = match.market_1x2 ?? { available: false, confidence: null, p_win_a: null, p_draw: null, p_win_b: null, raw_p_win_a: null, raw_p_draw: null, raw_p_win_b: null, source_used: null, market_slug: null };
        const exact = match.exact_score_market ?? { available: false, score_probability_source: "model_score_grid", market_score_weight: null, scores_count: 0, raw_probability_sum: null, top_scores: [] };
        const recommendation = match.recommendation;
        const bestEvScore = recommendation?.best_ev_score ?? match.best_ev_score;
        const evLoss = recommendation?.ev_loss_vs_best ?? match.ev_loss_vs_best;
        const selectionReason = recommendation?.selection_reason ?? match.selection_reason ?? match.recommendation_reason;
        return (
          <article key={match.match_id} className="match-card">
            <div className="match-summary">
              <div><strong>{match.team_a} <span className="versus">vs</span> {match.team_b}</strong><small>Group {match.group} · Round {match.match_round} · {formatKickoff(match.kickoff_at)}</small></div>
              <span className={`source-badge ${kind}`}>{label}</span>
              <div className="recommended"><small>Recommended</small><span className="score">{match.recommendation?.score ?? match.recommended_score}</span></div>
              <div><small>Expected points</small><strong>{decimal.format(match.recommendation?.expected_pool_points ?? match.expected_pool_points)}</strong></div>
            </div>
            <details>
              <summary>Prediction details</summary>
              <div className="detail-grid">
                <section><h3>Model</h3><p>Lambda: {decimal.format(model.lambda_a)} / {decimal.format(model.lambda_b)}</p><ProbabilityLine match={match} values={model} /><p>Recommended: <b>{model.recommended_score}</b></p></section>
                <section><h3>Score selection</h3><p>Best EV score: <b>{bestEvScore ?? match.recommended_score}</b></p><p>Chosen score: <b>{recommendation?.score ?? match.recommended_score}</b></p><p>EV loss: <b>{decimal.format(evLoss ?? 0)}</b></p><p>{selectionReason ?? "Beste score op expected pool points."}</p></section>
                <section><h3>Polymarket 1X2</h3>{market.available ? <><ProbabilityLine match={match} values={market} /><p>Confidence: <b>{market.confidence ?? "—"}</b></p><p>Largest delta: <b>{match.market_delta?.largest_outcome ?? "—"} {optionalPct(match.market_delta?.largest_abs_delta)}</b></p></> : <p className="fallback-copy">Geen Polymarket 1X2 beschikbaar, model fallback.</p>}</section>
                {match.hybrid_1x2?.available ? <section><h3>Hybrid</h3><ProbabilityLine match={match} values={match.hybrid_1x2} /><p>Market weight: {optionalPct(match.hybrid_1x2.market_weight)}</p><p>Source: {match.hybrid_1x2.source_used}</p></section> : null}
                {showExactScore ? <section><h3>Exact-score market</h3>{exact.available ? <div className="exact-scores">{exact.top_scores.map((item) => <div key={item.score} className={item.score === match.recommended_score ? "selected" : ""}><b>{item.score}</b><span>{optionalPct(item.normalized_probability)} normalized</span><span>{optionalPct(item.raw_probability)} raw · {item.confidence ?? "—"}</span></div>)}</div> : <p className="fallback-copy">Geen exact-score markt gevonden.</p>}</section> : null}
              </div>
              {match.warnings?.length ? <ul className="warnings">{match.warnings.map((warning) => <li key={warning}>{warning}</li>)}</ul> : null}
            </details>
          </article>
        );
      })}
    </div>
  );
}

function TeamsTable({ teams }: { teams: Team[] }) {
  return (
    <TableShell>
      <table>
        <thead>
          <tr><th>Team</th><th>Elo</th><th>R32</th><th>R16</th><th>Quarter</th><th>Semi</th><th>Final</th><th>Top 4</th><th>Champion</th></tr>
        </thead>
        <tbody>
          {teams.map((team, index) => (
            <tr key={team.team}>
              <td><span className="rank">{index + 1}</span><strong>{team.team}</strong><small>Group {team.group}</small></td>
              <td>{Math.round(team.elo)}</td>
              <td>{pct(team.p_round_of_32)}</td>
              <td>{pct(team.p_round_of_16)}</td>
              <td>{pct(team.p_quarter_final)}</td>
              <td>{pct(team.p_semi_final)}</td>
              <td>{pct(team.p_final)}</td>
              <td>{pct(team.p_top4)}</td>
              <td><strong>{pct(team.p_champion)}</strong></td>
            </tr>
          ))}
        </tbody>
      </table>
    </TableShell>
  );
}

function ScorersTable({ scorers }: { scorers: TopScorer[] }) {
  return (
    <TableShell>
      <table>
        <thead><tr><th>Player</th><th>Team</th><th>Position</th><th>Expected goals</th><th>Top scorer</th><th>Top 3 goals</th><th>EV</th></tr></thead>
        <tbody>
          {scorers.map((player) => (
            <tr key={`${player.player}-${player.team}`}>
              <td><strong>{player.player}</strong>{player.is_recommended ? <span className="tag">Pick</span> : null}</td>
              <td>{player.team}</td>
              <td>{player.position ?? "—"}</td>
              <td>{decimal.format(player.expected_goals)}</td>
              <td>{pct(player.p_top_scorer)}</td>
              <td>{pct(player.p_top_3_goals)}</td>
              <td><strong>{decimal.format(player.recommended_score_value)}</strong></td>
            </tr>
          ))}
        </tbody>
      </table>
    </TableShell>
  );
}

function CoverageCard({ data, matches }: { data: FrontendData; matches: Match[] }) {
  const moneyline = data.coverage?.moneyline;
  const exact = data.coverage?.exact_score;
  const probabilitySource = String(data.metadata.probability_source ?? "model");
  const marketWeight = Number(data.metadata.market_weight ?? 0);
  return <section className="coverage-card"><div><p className="eyebrow">Data coverage</p><h2>What drives these picks</h2></div><dl>
    <div><dt>Polymarket 1X2</dt><dd>{moneyline ? `${moneyline.available}/${moneyline.total}` : "Unknown"}</dd></div>
    <div><dt>Exact-score markets</dt><dd>{exact ? `${exact.available}/${exact.total}` : "Unknown"}</dd></div>
    <div><dt>Probability source</dt><dd>{probabilitySource}{probabilitySource === "hybrid" ? `, ${Math.round(marketWeight * 100)}% market` : ""}</dd></div>
    <div><dt>Exact scores</dt><dd>{String(data.metadata.score_probability_source ?? "model_score_grid").replaceAll("_", " ")}</dd></div>
    <div><dt>Final standings</dt><dd>Simulation</dd></div><div><dt>Top scorers</dt><dd>Simulation</dd></div>
  </dl>{data.warnings?.map((warning) => <p className="info-banner" key={warning}>{warning}</p>)}
  {!data.coverage && matches.length ? <p className="legacy-note">Coverage is not available in this older export.</p> : null}</section>;
}

function LargestDeltas({ matches }: { matches: Match[] }) {
  const rows = matches.filter((match) => match.market_delta?.largest_abs_delta != null).sort((a, b) => (b.market_delta?.largest_abs_delta ?? 0) - (a.market_delta?.largest_abs_delta ?? 0)).slice(0, 5);
  if (!rows.length) return null;
  return <section><div className="section-heading"><div><p className="eyebrow">Model versus market</p><h2>Waar wijkt Polymarket het meest af?</h2></div></div><div className="delta-list">{rows.map((match) => {
    const outcome = match.market_delta?.largest_outcome ?? "home";
    const modelValue = outcome === "home" ? match.model?.p_win_a ?? match.p_win_a : outcome === "draw" ? match.model?.p_draw ?? match.p_draw : match.model?.p_win_b ?? match.p_win_b;
    const marketValue = outcome === "home" ? match.market_1x2?.p_win_a : outcome === "draw" ? match.market_1x2?.p_draw : match.market_1x2?.p_win_b;
    const delta = match.market_delta?.[outcome];
    return <article key={match.match_id}><strong>{match.team_a} - {match.team_b}</strong><span>{outcome}</span><span>Model {optionalPct(modelValue)}</span><span>Market {optionalPct(marketValue)}</span><b>{delta == null ? "—" : `${delta >= 0 ? "+" : ""}${decimal.format(delta * 100)} pp`}</b></article>;
  })}</div></section>;
}

function App() {
  const [data, setData] = useState<FrontendData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState<Tab>("basic");
  const [group, setGroup] = useState("all");
  const [round, setRound] = useState("all");
  const [teamSort, setTeamSort] = useState<TeamSort>("p_champion");
  const [scorerSort, setScorerSort] = useState<ScorerSort>("recommended_score_value");
  const [fallbackOnly, setFallbackOnly] = useState(false);

  useEffect(() => {
    const controller = new AbortController();
    fetch("/frontend_data.json", { signal: controller.signal })
      .then((response) => {
        if (!response.ok) throw new Error(`Could not load predictions (${response.status})`);
        return response.json() as Promise<FrontendData>;
      })
      .then((payload) => setData({ ...payload, schema_version: payload.schema_version ?? "1.0", matches: payload.round_1_predictions ?? payload.matches ?? [], teams: payload.teams ?? [], market_comparison: payload.market_comparison ?? [] }))
      .catch((reason: unknown) => {
        if (reason instanceof DOMException && reason.name === "AbortError") return;
        setError(reason instanceof Error ? reason.message : "Could not load predictions");
      });
    return () => controller.abort();
  }, []);

  const groups = useMemo(
    () => [...new Set(data?.matches?.map((match) => match.group) ?? [])].sort(),
    [data],
  );
  const rounds = useMemo(
    () => [...new Set(data?.matches?.map((match) => match.match_round) ?? [])].sort((a, b) => a - b),
    [data],
  );
  const filteredMatches = useMemo(
    () => data?.matches?.filter((match) =>
      (group === "all" || match.group === group) &&
      (round === "all" || match.match_round === Number(round)) &&
      (!fallbackOnly || sourceLabel(match)[1] === "fallback")) ?? [],
    [data, group, round, fallbackOnly],
  );
  const sortedTeams = useMemo(
    () => [...(data?.teams ?? [])].sort((a, b) => b[teamSort] - a[teamSort]),
    [data, teamSort],
  );
  const sortedScorers = useMemo(
    () => [...(data?.top_scorers ?? [])].sort((a, b) => b[scorerSort] - a[scorerSort]),
    [data, scorerSort],
  );

  if (error) return <main className="center-state"><span className="status-dot error" /><h1>Data unavailable</h1><p>{error}</p><button onClick={() => location.reload()}>Try again</button></main>;
  if (!data) return <main className="center-state"><span className="loader" /><h1>Loading predictions</h1><p>Reading the latest simulation export.</p></main>;

  const picks = data.top_scorers.filter((player) => player.is_recommended);
  const matches = data.matches ?? [];
  const showExactScore = (data.coverage?.exact_score.available ?? 1) > 0;
  const tabs: { id: Tab; label: string }[] = [
    { id: "basic", label: "Basic predictions" },
    { id: "matches", label: "Matches" },
    { id: "teams", label: "Teams" },
    { id: "scorers", label: "Top scorers" },
    ...(data.market_comparison.length ? [{ id: "market" as Tab, label: "Market" }] : []),
  ];

  return (
    <div className="app">
      <header>
        <div className="brand"><span>26</span><div><strong>World Cup</strong><small>Prediction model</small></div></div>
        <div className="run-meta"><span>Seed {data.metadata.seed ?? "—"}</span><span>{Number(data.metadata.num_simulations ?? 0).toLocaleString()} simulations</span></div>
      </header>
      <nav aria-label="Dashboard sections">
        {tabs.map((item) => <button key={item.id} className={tab === item.id ? "active" : ""} onClick={() => setTab(item.id)}>{item.label}</button>)}
      </nav>

      <main>
        <CoverageCard data={data} matches={matches} />
        {tab === "basic" ? (
          <>
            <section className="hero">
              <div><p className="eyebrow">WK 2026 · Model outlook</p><h1>The tournament,<br /><em>before it happens.</em></h1><p className="intro">A data-driven view of every opening match, title contender and top scorer pick.</p></div>
              <div className="trophy-card">
                <small>Projected champion</small><strong>{data.final_standings.gold}</strong>
                <span>{pct(data.teams.find((team) => team.team === data.final_standings.gold)?.p_champion ?? 0)} title probability</span>
              </div>
            </section>
            <section>
              <div className="section-heading"><div><p className="eyebrow">Podium forecast</p><h2>Final standings</h2></div></div>
              <div className="podium">
                {(["gold", "silver", "bronze", "fourth"] as const).map((position, index) => (
                  <article key={position} className={`place place-${index + 1}`}><span>0{index + 1}</span><small>{position}</small><strong>{data.final_standings[position]}</strong></article>
                ))}
              </div>
            </section>
            <section>
              <div className="section-heading"><div><p className="eyebrow">Pool picks</p><h2>Opening round recommendations</h2></div><button className="text-button" onClick={() => setTab("matches")}>View all matches →</button></div>
              <MatchesTable matches={matches.slice(0, 6)} showExactScore={showExactScore} />
            </section>
            <LargestDeltas matches={matches} />
            <section>
              <div className="section-heading"><div><p className="eyebrow">Golden boot</p><h2>Recommended scorers</h2></div></div>
              <div className="scorer-cards">{picks.map((player, index) => <article key={player.player}><span>0{index + 1}</span><div><strong>{player.player}</strong><small>{player.team} · {decimal.format(player.expected_goals)} expected goals</small></div><b>{decimal.format(player.recommended_score_value)} EV</b></article>)}</div>
            </section>
            <section className="metadata">
              <div><p className="eyebrow">About this run</p><h2>Model metadata</h2></div>
              <dl>{Object.entries(data.metadata).filter(([key, value]) => key !== "limitations" && typeof value !== "object").map(([key, value]) => <div key={key}><dt>{key.replaceAll("_", " ")}</dt><dd>{String(value)}</dd></div>)}</dl>
              {data.metadata.limitations?.length ? <div className="limitations"><h3>Limitations</h3><ul>{data.metadata.limitations.map((item) => <li key={item}>{item}</li>)}</ul></div> : null}
            </section>
          </>
        ) : null}

        {tab === "matches" ? <section><div className="page-title"><p className="eyebrow">Fixture model</p><h1>Match predictions</h1><p>Recommended pool scores and model probabilities for every round-one fixture.</p></div><div className="controls"><label>Group<select value={group} onChange={(event) => setGroup(event.target.value)}><option value="all">All groups</option>{groups.map((value) => <option key={value}>{value}</option>)}</select></label><label>Round<select value={round} onChange={(event) => setRound(event.target.value)}><option value="all">All rounds</option>{rounds.map((value) => <option key={value}>{value}</option>)}</select></label><label className="checkbox"><input type="checkbox" checked={fallbackOnly} onChange={(event) => setFallbackOnly(event.target.checked)} />Fallback only</label><span>{filteredMatches.length} matches</span></div><MatchesTable matches={filteredMatches} showExactScore={showExactScore} /><LargestDeltas matches={filteredMatches} /></section> : null}

        {tab === "teams" ? <section><div className="page-title"><p className="eyebrow">Tournament forecast</p><h1>Team probabilities</h1><p>Progression odds across every stage of the competition.</p></div><div className="controls"><label>Rank teams by<select value={teamSort} onChange={(event) => setTeamSort(event.target.value as TeamSort)}><option value="p_champion">Champion probability</option><option value="p_top4">Top 4 probability</option><option value="p_final">Final probability</option></select></label></div><TeamsTable teams={sortedTeams} /></section> : null}

        {tab === "scorers" ? <section><div className="page-title"><p className="eyebrow">Golden boot model</p><h1>Top scorers</h1><p>Expected goals, finish probabilities and recommendation value.</p></div><div className="controls"><label>Rank players by<select value={scorerSort} onChange={(event) => setScorerSort(event.target.value as ScorerSort)}><option value="recommended_score_value">Expected value</option><option value="expected_goals">Expected goals</option></select></label></div><ScorersTable scorers={sortedScorers} /></section> : null}

        {tab === "market" ? <section><div className="page-title"><p className="eyebrow">Model versus market</p><h1>Market comparison</h1></div>{data.market_comparison.length ? <pre>{JSON.stringify(data.market_comparison, null, 2)}</pre> : <EmptyState>No market comparison is available for this run.</EmptyState>}</section> : null}
      </main>
      <footer><span>WK 2026 prediction model</span><span>Generated data · No live market calls</span></footer>
    </div>
  );
}

export default App;
