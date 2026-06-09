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

function MatchesTable({ matches }: { matches: Match[] }) {
  return (
    <TableShell>
      <table>
        <thead>
          <tr>
            <th>Match</th>
            <th>Kickoff</th>
            <th>Probabilities</th>
            <th>Model score</th>
            <th>Recommended</th>
            <th>Expected points</th>
          </tr>
        </thead>
        <tbody>
          {matches.map((match) => (
            <tr key={match.match_id} className="match-row">
              <td data-label="Match">
                <strong>{match.team_a}</strong>
                <span className="versus">vs</span>
                <strong>{match.team_b}</strong>
                <small>Group {match.group} · Round {match.match_round}</small>
              </td>
              <td data-label="Kickoff">
                {formatKickoff(match.kickoff_at)}
                <small>{match.location ?? "Location TBD"}</small>
              </td>
              <td data-label="Odds" className="probabilities">
                <span><small>{match.team_a}</small>{pct(match.p_win_a)}</span>
                <span><small>Draw</small>{pct(match.p_draw)}</span>
                <span><small>{match.team_b}</small>{pct(match.p_win_b)}</span>
              </td>
              <td data-label="Model score">{match.most_likely_score}</td>
              <td data-label="Recommended"><span className="score">{match.recommended_score}</span></td>
              <td data-label="Expected points">{decimal.format(match.expected_pool_points)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </TableShell>
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

function App() {
  const [data, setData] = useState<FrontendData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState<Tab>("basic");
  const [group, setGroup] = useState("all");
  const [round, setRound] = useState("all");
  const [teamSort, setTeamSort] = useState<TeamSort>("p_champion");
  const [scorerSort, setScorerSort] = useState<ScorerSort>("recommended_score_value");

  useEffect(() => {
    const controller = new AbortController();
    fetch("/data/frontend_data.json", { signal: controller.signal })
      .then((response) => {
        if (!response.ok) throw new Error(`Could not load predictions (${response.status})`);
        return response.json() as Promise<FrontendData>;
      })
      .then(setData)
      .catch((reason: unknown) => {
        if (reason instanceof DOMException && reason.name === "AbortError") return;
        setError(reason instanceof Error ? reason.message : "Could not load predictions");
      });
    return () => controller.abort();
  }, []);

  const groups = useMemo(
    () => [...new Set(data?.matches.map((match) => match.group) ?? [])].sort(),
    [data],
  );
  const rounds = useMemo(
    () => [...new Set(data?.matches.map((match) => match.match_round) ?? [])].sort((a, b) => a - b),
    [data],
  );
  const filteredMatches = useMemo(
    () => data?.matches.filter((match) =>
      (group === "all" || match.group === group) &&
      (round === "all" || match.match_round === Number(round))) ?? [],
    [data, group, round],
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
              <MatchesTable matches={data.matches.slice(0, 6)} />
            </section>
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

        {tab === "matches" ? <section><div className="page-title"><p className="eyebrow">Fixture model</p><h1>Match predictions</h1><p>Recommended pool scores and model probabilities for every round-one fixture.</p></div><div className="controls"><label>Group<select value={group} onChange={(event) => setGroup(event.target.value)}><option value="all">All groups</option>{groups.map((value) => <option key={value}>{value}</option>)}</select></label><label>Round<select value={round} onChange={(event) => setRound(event.target.value)}><option value="all">All rounds</option>{rounds.map((value) => <option key={value}>{value}</option>)}</select></label><span>{filteredMatches.length} matches</span></div><MatchesTable matches={filteredMatches} /></section> : null}

        {tab === "teams" ? <section><div className="page-title"><p className="eyebrow">Tournament forecast</p><h1>Team probabilities</h1><p>Progression odds across every stage of the competition.</p></div><div className="controls"><label>Rank teams by<select value={teamSort} onChange={(event) => setTeamSort(event.target.value as TeamSort)}><option value="p_champion">Champion probability</option><option value="p_top4">Top 4 probability</option><option value="p_final">Final probability</option></select></label></div><TeamsTable teams={sortedTeams} /></section> : null}

        {tab === "scorers" ? <section><div className="page-title"><p className="eyebrow">Golden boot model</p><h1>Top scorers</h1><p>Expected goals, finish probabilities and recommendation value.</p></div><div className="controls"><label>Rank players by<select value={scorerSort} onChange={(event) => setScorerSort(event.target.value as ScorerSort)}><option value="recommended_score_value">Expected value</option><option value="expected_goals">Expected goals</option></select></label></div><ScorersTable scorers={sortedScorers} /></section> : null}

        {tab === "market" ? <section><div className="page-title"><p className="eyebrow">Model versus market</p><h1>Market comparison</h1></div>{data.market_comparison.length ? <pre>{JSON.stringify(data.market_comparison, null, 2)}</pre> : <EmptyState>No market comparison is available for this run.</EmptyState>}</section> : null}
      </main>
      <footer><span>WK 2026 prediction model</span><span>Generated data · No live market calls</span></footer>
    </div>
  );
}

export default App;
