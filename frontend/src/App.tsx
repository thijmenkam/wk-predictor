import { useEffect, useMemo, useState } from "react";
import type { FrontendData, Match, Team } from "./types";

type Tab = "basic" | "matches" | "teams" | "market";
type TeamSort = "p_champion" | "p_top4" | "p_final";

const percentage = new Intl.NumberFormat("en", {
  style: "percent",
  maximumFractionDigits: 1,
});
const decimal = new Intl.NumberFormat("en", { maximumFractionDigits: 2 });

const teamFlags: Record<string, string> = {
  Algeria: "🇩🇿",
  Argentina: "🇦🇷",
  Australia: "🇦🇺",
  Austria: "🇦🇹",
  Belgium: "🇧🇪",
  Bosnia: "🇧🇦",
  Brazil: "🇧🇷",
  Canada: "🇨🇦",
  "Cape Verde": "🇨🇻",
  Colombia: "🇨🇴",
  Croatia: "🇭🇷",
  Curaçao: "🇨🇼",
  Czechia: "🇨🇿",
  "DR Congo": "🇨🇩",
  Ecuador: "🇪🇨",
  Egypt: "🇪🇬",
  England: "\u{1F3F4}\u{E0067}\u{E0062}\u{E0065}\u{E006E}\u{E0067}\u{E007F}",
  France: "🇫🇷",
  Germany: "🇩🇪",
  Ghana: "🇬🇭",
  Haiti: "🇭🇹",
  Iran: "🇮🇷",
  Iraq: "🇮🇶",
  "Ivory Coast": "🇨🇮",
  Japan: "🇯🇵",
  Jordan: "🇯🇴",
  Mexico: "🇲🇽",
  Morocco: "🇲🇦",
  Netherlands: "🇳🇱",
  "New Zealand": "🇳🇿",
  Norway: "🇳🇴",
  Panama: "🇵🇦",
  Paraguay: "🇵🇾",
  Portugal: "🇵🇹",
  Qatar: "🇶🇦",
  "Saudi Arabia": "🇸🇦",
  Scotland: "\u{1F3F4}\u{E0067}\u{E0062}\u{E0073}\u{E0063}\u{E0074}\u{E007F}",
  Senegal: "🇸🇳",
  "South Africa": "🇿🇦",
  "South Korea": "🇰🇷",
  Spain: "🇪🇸",
  Sweden: "🇸🇪",
  Switzerland: "🇨🇭",
  Tunisia: "🇹🇳",
  Türkiye: "🇹🇷",
  Uruguay: "🇺🇾",
  USA: "🇺🇸",
  Uzbekistan: "🇺🇿",
};

function teamName(team: string) {
  const flag = teamFlags[team];
  return flag ? `${flag} ${team}` : team;
}

function englishText(value: string) {
  const translations: Record<string, string> = {
    "Beste score op expected pool points.": "Best score by expected pool points.",
    "Alternatief gekozen binnen EV-tolerantie voor realistischer scorebeeld.": "Alternative selected within the EV tolerance for a more realistic score.",
    "Draw gekozen binnen EV-tolerantie vanwege hoge draw probability.": "Draw selected within the EV tolerance because of the high draw probability.",
    "Draw gekozen binnen EV-tolerantie om realistische draw-rate te bereiken.": "Draw selected within the EV tolerance to reach a realistic draw rate.",
    "Geen Polymarket 1X2-markt gevonden, model fallback gebruikt.": "No Polymarket 1X2 market found; model fallback used.",
    "Geen exact-score markt gevonden.": "No exact-score market found.",
  };
  return translations[value] ?? value;
}

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
  return <div className="probability-line"><span>{teamName(match.team_a)} <b>{optionalPct(values.p_win_a)}</b></span><span>Draw <b>{optionalPct(values.p_draw)}</b></span><span>{teamName(match.team_b)} <b>{optionalPct(values.p_win_b)}</b></span></div>;
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
        const warnings = match.warnings
          ?.filter((warning) => warning.trim() && warning.toLowerCase() !== "nan")
          .map(englishText) ?? [];
        return (
          <article key={match.match_id} className="match-card">
            <div className="match-summary">
              <div><strong>{teamName(match.team_a)} <span className="versus">vs</span> {teamName(match.team_b)}</strong><small>Group {match.group} · Round {match.match_round} · {formatKickoff(match.kickoff_at)}</small></div>
              <span className={`source-badge ${kind}`}>{label}</span>
              <div className="recommended"><small>Recommended</small><span className="score">{match.recommendation?.score ?? match.recommended_score}</span></div>
              <div><small>Expected points</small><strong>{decimal.format(match.recommendation?.expected_pool_points ?? match.expected_pool_points)}</strong></div>
            </div>
            <details>
              <summary>Prediction details</summary>
              <div className="detail-grid">
                <section><h3>Model</h3><p>Lambda: {decimal.format(model.lambda_a)} / {decimal.format(model.lambda_b)}</p><ProbabilityLine match={match} values={model} /><p>Recommended: <b>{model.recommended_score}</b></p></section>
                <section><h3>Score selection</h3><p>Best EV score: <b>{bestEvScore ?? match.recommended_score}</b></p><p>Chosen score: <b>{recommendation?.score ?? match.recommended_score}</b></p><p>EV loss: <b>{decimal.format(evLoss ?? 0)}</b></p><p>{englishText(selectionReason ?? "Best score by expected pool points.")}</p></section>
                <section><h3>Polymarket 1X2</h3>{market.available ? <><ProbabilityLine match={match} values={market} /><p>Confidence: <b>{market.confidence ?? "—"}</b></p><p>Largest delta: <b>{match.market_delta?.largest_outcome ?? "—"} {optionalPct(match.market_delta?.largest_abs_delta)}</b></p></> : <p className="fallback-copy">No Polymarket 1X2 market available; using the model fallback.</p>}</section>
                {match.hybrid_1x2?.available ? <section><h3>Hybrid</h3><ProbabilityLine match={match} values={match.hybrid_1x2} /><p>Market weight: {optionalPct(match.hybrid_1x2.market_weight)}</p><p>Source: {match.hybrid_1x2.source_used}</p></section> : null}
                {showExactScore ? <section><h3>Exact-score market</h3>{exact.available ? <div className="exact-scores">{exact.top_scores.map((item) => <div key={item.score} className={item.score === match.recommended_score ? "selected" : ""}><b>{item.score}</b><span>{optionalPct(item.normalized_probability)} normalized</span><span>{optionalPct(item.raw_probability)} raw · {item.confidence ?? "—"}</span></div>)}</div> : <p className="fallback-copy">No exact-score market found.</p>}</section> : null}
              </div>
              {warnings.length ? <ul className="warnings">{warnings.map((warning) => <li key={warning}>{warning}</li>)}</ul> : null}
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
              <td><span className="rank">{index + 1}</span><strong>{teamName(team.team)}</strong><small>Group {team.group}</small></td>
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

function LargestDeltas({ matches }: { matches: Match[] }) {
  const rows = matches.filter((match) => match.market_delta?.largest_abs_delta != null).sort((a, b) => (b.market_delta?.largest_abs_delta ?? 0) - (a.market_delta?.largest_abs_delta ?? 0)).slice(0, 5);
  if (!rows.length) return null;
  return <section><div className="section-heading"><div><h2>Where does Polymarket differ most?</h2></div></div><div className="delta-list">{rows.map((match) => {
    const outcome = match.market_delta?.largest_outcome ?? "home";
    const modelValue = outcome === "home" ? match.model?.p_win_a ?? match.p_win_a : outcome === "draw" ? match.model?.p_draw ?? match.p_draw : match.model?.p_win_b ?? match.p_win_b;
    const marketValue = outcome === "home" ? match.market_1x2?.p_win_a : outcome === "draw" ? match.market_1x2?.p_draw : match.market_1x2?.p_win_b;
    const delta = match.market_delta?.[outcome];
    return <article key={match.match_id}><strong>{teamName(match.team_a)} - {teamName(match.team_b)}</strong><span>{outcome}</span><span>Model {optionalPct(modelValue)}</span><span>Market {optionalPct(marketValue)}</span><b>{delta == null ? "—" : `${delta >= 0 ? "+" : ""}${decimal.format(delta * 100)} pp`}</b></article>;
  })}</div></section>;
}

function App() {
  const [data, setData] = useState<FrontendData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState<Tab>("basic");
  const [group, setGroup] = useState("all");
  const [round, setRound] = useState("all");
  const [teamSort, setTeamSort] = useState<TeamSort>("p_champion");

  useEffect(() => {
    const controller = new AbortController();
    fetch(`${import.meta.env.BASE_URL}frontend_data.json`, { signal: controller.signal })
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
      (round === "all" || match.match_round === Number(round))) ?? [],
    [data, group, round],
  );
  const sortedTeams = useMemo(
    () => [...(data?.teams ?? [])].sort((a, b) => b[teamSort] - a[teamSort]),
    [data, teamSort],
  );
  if (error) return <main className="center-state"><span className="status-dot error" /><h1>Data unavailable</h1><p>{error}</p><button onClick={() => location.reload()}>Try again</button></main>;
  if (!data) return <main className="center-state"><span className="loader" /><h1>Loading predictions</h1><p>Reading the latest simulation export.</p></main>;

  const matches = data.matches ?? [];
  const showExactScore = (data.coverage?.exact_score.available ?? 1) > 0;
  const tabs: { id: Tab; label: string }[] = [
    { id: "basic", label: "Basic predictions" },
    { id: "matches", label: "Matches" },
    { id: "teams", label: "Teams" },
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
        {tab === "basic" ? (
          <>
            <section className="hero">
              <div><h1>The tournament,<br /><em>before it happens.</em></h1><p className="intro">A data-driven view of every opening match and title contender.</p></div>
              <div className="trophy-card">
                <small>Projected champion</small><strong>{teamName(data.final_standings.gold)}</strong>
                <span>{pct(data.teams.find((team) => team.team === data.final_standings.gold)?.p_champion ?? 0)} title probability</span>
              </div>
            </section>
            <section>
              <div className="section-heading"><div><h2>Final standings</h2></div></div>
              <div className="podium">
                {(["gold", "silver", "bronze", "fourth"] as const).map((position, index) => (
                  <article key={position} className={`place place-${index + 1}`}><span>0{index + 1}</span><small>{position}</small><strong>{teamName(data.final_standings[position])}</strong></article>
                ))}
              </div>
            </section>
            <section>
              <div className="section-heading"><div><h2>Opening round recommendations</h2></div><button className="text-button" onClick={() => setTab("matches")}>View all matches →</button></div>
              <MatchesTable matches={matches.slice(0, 6)} showExactScore={showExactScore} />
            </section>
            <LargestDeltas matches={matches} />
            <section className="metadata">
              <div><h2>How it works</h2></div>
              <div className="method-grid">
                <article>
                  <h3>Team strength</h3>
                  <p>Elo ratings estimate the relative strength of all 48 teams and provide the basis for each match prediction.</p>
                </article>
                <article>
                  <h3>Match probabilities</h3>
                  <p>A Poisson score model calculates likely results. For 1X2 probabilities, this run blends the model with {Math.round(Number(data.metadata.market_weight ?? 0) * 100)}% local Polymarket data.</p>
                </article>
                <article>
                  <h3>Tournament forecast</h3>
                  <p>{Number(data.metadata.num_simulations ?? 0).toLocaleString()} simulations play out the group stage and knockout bracket to estimate progression and title chances.</p>
                </article>
              </div>
            </section>
          </>
        ) : null}

        {tab === "matches" ? <section><div className="page-title"><h1>Match predictions</h1><p>Recommended pool scores and model probabilities for every round-one fixture.</p></div><div className="controls"><label>Group<select value={group} onChange={(event) => setGroup(event.target.value)}><option value="all">All groups</option>{groups.map((value) => <option key={value}>{value}</option>)}</select></label><label>Round<select value={round} onChange={(event) => setRound(event.target.value)}><option value="all">All rounds</option>{rounds.map((value) => <option key={value}>{value}</option>)}</select></label><span>{filteredMatches.length} matches</span></div><MatchesTable matches={filteredMatches} showExactScore={showExactScore} /><LargestDeltas matches={filteredMatches} /></section> : null}

        {tab === "teams" ? <section><div className="page-title"><h1>Team probabilities</h1><p>Progression odds across every stage of the competition.</p></div><div className="controls"><label>Rank teams by<select value={teamSort} onChange={(event) => setTeamSort(event.target.value as TeamSort)}><option value="p_champion">Champion probability</option><option value="p_top4">Top 4 probability</option><option value="p_final">Final probability</option></select></label></div><TeamsTable teams={sortedTeams} /></section> : null}

        {tab === "market" ? <section><div className="page-title"><h1>Market comparison</h1></div>{data.market_comparison.length ? <pre>{JSON.stringify(data.market_comparison, null, 2)}</pre> : <EmptyState>No market comparison is available for this run.</EmptyState>}</section> : null}
      </main>
      <footer><span>World Cup 2026 prediction model</span><span>Made by Thijmen with Codex</span></footer>
    </div>
  );
}

export default App;
