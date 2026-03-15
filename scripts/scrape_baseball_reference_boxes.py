from __future__ import annotations

import argparse
import io
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup, Comment
from curl_cffi import requests as curl_requests
from curl_cffi.requests.exceptions import RequestException as CurlRequestException

TEAM_CODES_2024_AND_EARLIER = [
    "ARI", "ATL", "BAL", "BOS", "CHC", "CHW", "CIN", "CLE", "COL", "DET",
    "HOU", "KCR", "LAA", "LAD", "MIA", "MIL", "MIN", "NYM", "NYY", "OAK",
    "PHI", "PIT", "SDP", "SEA", "SFG", "STL", "TBR", "TEX", "TOR", "WSN",
]

TEAM_CODES_2025_AND_LATER = [
    "ARI", "ATL", "BAL", "BOS", "CHC", "CHW", "CIN", "CLE", "COL", "DET",
    "HOU", "KCR", "LAA", "LAD", "MIA", "MIL", "MIN", "NYM", "NYY", "ATH",
    "PHI", "PIT", "SDP", "SEA", "SFG", "STL", "TBR", "TEX", "TOR", "WSN",
]

BR_TEAM_CODES = {
    "ANA": "ANA",
    "ARI": "ARI",
    "ATH": "ATH",
    "ATL": "ATL",
    "BAL": "BAL",
    "BOS": "BOS",
    "CHC": "CHN",
    "CHW": "CHA",
    "CIN": "CIN",
    "CLE": "CLE",
    "COL": "COL",
    "DET": "DET",
    "HOU": "HOU",
    "KCR": "KCA",
    "LAA": "ANA",
    "LAD": "LAN",
    "MIA": "MIA",
    "MIL": "MIL",
    "MIN": "MIN",
    "NYM": "NYN",
    "NYY": "NYA",
    "OAK": "OAK",
    "PHI": "PHI",
    "PIT": "PIT",
    "SDP": "SDN",
    "SEA": "SEA",
    "SFG": "SFN",
    "STL": "SLN",
    "TBR": "TBA",
    "TEX": "TEX",
    "TOR": "TOR",
    "WSN": "WAS",
}

BR_TEAM_TABLE_NAMES = {
    "ARI": "ArizonaDiamondbacks",
    "ATL": "AtlantaBraves",
    "BAL": "BaltimoreOrioles",
    "BOS": "BostonRedSox",
    "CHC": "ChicagoCubs",
    "CHW": "ChicagoWhiteSox",
    "CIN": "CincinnatiReds",
    "CLE": "ClevelandGuardians",
    "COL": "ColoradoRockies",
    "DET": "DetroitTigers",
    "HOU": "HoustonAstros",
    "KCR": "KansasCityRoyals",
    "LAA": "LosAngelesAngels",
    "LAD": "LosAngelesDodgers",
    "MIA": "MiamiMarlins",
    "MIL": "MilwaukeeBrewers",
    "MIN": "MinnesotaTwins",
    "NYM": "NewYorkMets",
    "NYY": "NewYorkYankees",
    "ATH": "Athletics",
    "PHI": "PhiladelphiaPhillies",
    "PIT": "PittsburghPirates",
    "SDP": "SanDiegoPadres",
    "SEA": "SeattleMariners",
    "SFG": "SanFranciscoGiants",
    "STL": "StLouisCardinals",
    "TBR": "TampaBayRays",
    "TEX": "TexasRangers",
    "TOR": "TorontoBlueJays",
    "WSN": "WashingtonNationals",
    "ANA": "LosAngelesAngels",
    "OAK": "OaklandAthletics",
}

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/127.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.baseball-reference.com/",
}

IMPERSIONATION = "chrome"
DEFAULT_MAX_RETRIES = 4


def team_codes_for_season(season: int) -> list[str]:
    if season >= 2025:
        return TEAM_CODES_2025_AND_LATER
    return TEAM_CODES_2024_AND_EARLIER


class RateLimiter:
    def __init__(self, min_interval_seconds: float) -> None:
        self.min_interval_seconds = min_interval_seconds
        self._last_request_at = 0.0

    def wait(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_request_at
        if elapsed < self.min_interval_seconds:
            time.sleep(self.min_interval_seconds - elapsed)
        self._last_request_at = time.monotonic()


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            "_".join(str(part) for part in col if str(part) not in {"", "nan"}).strip("_")
            for col in df.columns
        ]
    else:
        df.columns = [str(col).strip() for col in df.columns]
    return df


def normalize_schedule_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for column in list(out.columns):
        if not str(column).startswith("Unnamed:"):
            continue

        values = out[column].fillna("").astype(str).str.strip()
        distinct = {value for value in values.unique().tolist() if value}
        if distinct and distinct.issubset({"@"}):
            out = out.rename(columns={column: "dest"})
        elif distinct and distinct.issubset({"boxscore"}):
            out = out.rename(columns={column: "boxscore"})

    drop_columns = [column for column in out.columns if str(column).startswith("Unnamed:")]
    if drop_columns:
        out = out.drop(columns=drop_columns)

    return out


def parse_table_from_html(html: str, table_id: str) -> pd.DataFrame:
    soup = BeautifulSoup(html, "lxml")

    direct = soup.find(id=table_id)
    if direct is not None:
        try:
            return flatten_columns(pd.read_html(io.StringIO(str(direct)))[0])
        except ValueError:
            pass

    wrapper = soup.find(id=f"all_{table_id}")
    if wrapper is not None:
        for node in wrapper.descendants:
            if isinstance(node, Comment) and "<table" in node:
                inner = BeautifulSoup(str(node), "lxml").find(id=table_id)
                if inner is not None:
                    return flatten_columns(pd.read_html(io.StringIO(str(inner)))[0])

    for node in soup.find_all(string=lambda text: isinstance(text, Comment)):
        if table_id not in node:
            continue
        inner = BeautifulSoup(str(node), "lxml").find(id=table_id)
        if inner is not None:
            return flatten_columns(pd.read_html(io.StringIO(str(inner)))[0])

    raise ValueError(f"Could not find table '{table_id}' in page source")


def list_table_ids(html: str) -> list[str]:
    soup = BeautifulSoup(html, "lxml")
    ids = {tag.get("id") for tag in soup.find_all("table", id=True)}
    for node in soup.find_all(string=lambda text: isinstance(text, Comment)):
        if "<table" not in node:
            continue
        inner = BeautifulSoup(str(node), "lxml")
        ids.update(tag.get("id") for tag in inner.find_all("table", id=True))
    return sorted(table_id for table_id in ids if table_id)


def fetch_html(
    session: requests.Session,
    limiter: RateLimiter,
    url: str,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> str:
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        limiter.wait()
        try:
            response = curl_requests.get(
                url,
                headers=REQUEST_HEADERS,
                impersonate=IMPERSIONATION,
                timeout=30,
            )
            response.raise_for_status()
            return response.text
        except CurlRequestException as exc:
            last_error = exc
            print(f"curl_cffi attempt {attempt}/{max_retries} failed for {url}: {exc}")

        limiter.wait()
        try:
            response = session.get(url, headers=REQUEST_HEADERS, timeout=30)
            response.raise_for_status()
            return response.text
        except requests.RequestException as exc:
            last_error = exc
            print(f"requests attempt {attempt}/{max_retries} failed for {url}: {exc}")

        if attempt < max_retries:
            time.sleep(min(5 * attempt, 20))

    if last_error is None:
        raise RuntimeError(f"Failed to fetch {url} for an unknown reason")
    raise last_error


def scrape_team_schedule(
    session: requests.Session,
    limiter: RateLimiter,
    team_code: str,
    season: int,
) -> pd.DataFrame:
    url = (
        f"https://www.baseball-reference.com/teams/{team_code}/"
        f"{season}-schedule-scores.shtml#all_team_schedule"
    )
    html = fetch_html(session, limiter, url)
    df = parse_table_from_html(html, "team_schedule")
    df = normalize_schedule_columns(df)
    df["schedule_team"] = team_code
    df["source_url"] = url
    return df


def build_game_ids_from_schedule(df: pd.DataFrame, season: int) -> pd.DataFrame:
    work = df.copy()
    work["dest"] = work["dest"].fillna("")
    work = work[work["Date"].notna()].copy()
    work = work[work["Date"].astype(str).str.strip() != "Date"].copy()
    work = work[work["Tm"].astype(str).str.strip() != "Tm"].copy()
    work = work[work["Opp"].astype(str).str.strip() != "Opp"].copy()
    work = work[~work["Date"].astype(str).str.contains("Postponed", na=False)].copy()

    work["date_clean"] = work["Date"].astype(str).str.replace(r"\s+\((\d+)\)$", "", regex=True)
    work["game_number"] = work["Date"].astype(str).str.extract(r"\((\d+)\)$", expand=False).fillna("0")
    work["home_team"] = work.apply(
        lambda row: row["Opp"].strip() if row["dest"] == "@" else row["Tm"].strip(),
        axis=1,
    )
    work["br_home_code"] = work["home_team"].map(BR_TEAM_CODES)

    missing = sorted(work.loc[work["br_home_code"].isna(), "home_team"].dropna().unique())
    if missing:
        raise ValueError(f"Missing BR code mapping for: {missing}")

    work["game_date"] = pd.to_datetime(
        work["date_clean"].str.strip() + f" {season}",
        format="%A, %b %d %Y",
        errors="coerce",
    )
    work = work[work["game_date"].notna()].copy()

    work["game_id"] = (
        work["br_home_code"]
        + work["game_date"].dt.strftime("%Y%m%d")
        + work["game_number"]
    )

    return work[[
        "game_id", "game_date", "game_number", "Tm", "Opp", "dest", "home_team",
        "br_home_code", "schedule_team", "source_url",
    ]]


def build_season_game_ids(
    session: requests.Session,
    limiter: RateLimiter,
    season: int,
) -> pd.DataFrame:
    frames = []
    for team_code in team_codes_for_season(season):
        print(f"Scraping schedule for {team_code} ({season})")
        frames.append(build_game_ids_from_schedule(scrape_team_schedule(session, limiter, team_code, season), season))

    all_rows = pd.concat(frames, ignore_index=True)
    return (
        all_rows
        .sort_values(["game_date", "game_id", "schedule_team"])
        .drop_duplicates(subset=["game_id"])
        .reset_index(drop=True)
    )


def write_game_ids_csv(game_ids_df: pd.DataFrame, output_root: Path, season: int) -> Path:
    game_ids_path = output_root / f"br_game_ids_{season}.csv"
    game_ids_df.to_csv(game_ids_path, index=False)
    return game_ids_path


def boxscore_url(game_id: str) -> str:
    return f"https://www.baseball-reference.com/boxes/{game_id[:3]}/{game_id}.shtml"


def find_team_table_ids(table_ids: list[str], suffix: str, season: int) -> dict[str, str]:
    matches: dict[str, str] = {}
    valid_team_codes = set(team_codes_for_season(season))
    for team_code, table_name in BR_TEAM_TABLE_NAMES.items():
        candidate = f"{table_name}{suffix}"
        if candidate in table_ids and team_code in valid_team_codes:
            matches[team_code] = candidate
    return matches


def add_common_columns(
    df: pd.DataFrame,
    game_id: str,
    source_url: str,
    table_id: str,
    team_code: str | None = None,
) -> pd.DataFrame:
    out = df.copy()
    out.insert(0, "table_id", table_id)
    out.insert(0, "source_url", source_url)
    if team_code is not None:
        out.insert(0, "team_code", team_code)
    out.insert(0, "game_id", game_id)
    return out


def ensure_output_dirs(root: Path) -> dict[str, Path]:
    out_dirs = {
        "batting": root / "batting",
        "pitching": root / "pitching",
        "pbp": root / "pbp",
    }
    for path in out_dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return out_dirs


def scrape_game(
    session: requests.Session,
    limiter: RateLimiter,
    season: int,
    game_id: str,
    out_dirs: dict[str, Path],
    overwrite: bool,
) -> None:
    url = boxscore_url(game_id)
    html = fetch_html(session, limiter, url)
    table_ids = list_table_ids(html)

    batting_tables = find_team_table_ids(table_ids, "batting", season)
    pitching_tables = find_team_table_ids(table_ids, "pitching", season)

    if len(batting_tables) != 2:
        raise ValueError(f"{game_id}: expected 2 batting tables, found {len(batting_tables)}")
    if len(pitching_tables) != 2:
        raise ValueError(f"{game_id}: expected 2 pitching tables, found {len(pitching_tables)}")
    if "play_by_play" not in table_ids:
        raise ValueError(f"{game_id}: missing play_by_play table")

    for team_code, table_id in sorted(batting_tables.items()):
        path = out_dirs["batting"] / f"{game_id}_{team_code}.csv"
        if overwrite or not path.exists():
            df = parse_table_from_html(html, table_id)
            add_common_columns(df, game_id, url, table_id, team_code).to_csv(path, index=False)

    for team_code, table_id in sorted(pitching_tables.items()):
        path = out_dirs["pitching"] / f"{game_id}_{team_code}.csv"
        if overwrite or not path.exists():
            df = parse_table_from_html(html, table_id)
            add_common_columns(df, game_id, url, table_id, team_code).to_csv(path, index=False)

    pbp_path = out_dirs["pbp"] / f"{game_id}.csv"
    if overwrite or not pbp_path.exists():
        pbp_df = parse_table_from_html(html, "play_by_play")
        add_common_columns(pbp_df, game_id, url, "play_by_play").to_csv(pbp_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scrape Baseball Reference batting, pitching, and play-by-play box score tables."
    )
    parser.add_argument(
        "--build-game-ids-only",
        action="store_true",
        help="Only scrape schedules and write the deduplicated season game-id CSV.",
    )
    parser.add_argument("--season", type=int, default=2025, help="MLB season to scrape.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data"),
        help="Root directory for outputs. Creates batting/, pitching/, and pbp/ under this path.",
    )
    parser.add_argument(
        "--game-ids-csv",
        type=Path,
        default=None,
        help="Optional existing CSV of game ids. If omitted, schedules are scraped to build it.",
    )
    parser.add_argument(
        "--max-games",
        type=int,
        default=None,
        help="Optional limit for partial runs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing CSVs.",
    )
    parser.add_argument(
        "--min-interval-seconds",
        type=float,
        default=10.0,
        help="Minimum delay between HTTP requests. Default 10s to respect BR's 6 req/min limit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    out_dirs = ensure_output_dirs(output_root)

    session = requests.Session()
    limiter = RateLimiter(min_interval_seconds=args.min_interval_seconds)

    if args.game_ids_csv is not None:
        game_ids_df = pd.read_csv(args.game_ids_csv)
    else:
        game_ids_df = build_season_game_ids(session, limiter, args.season)
        game_ids_path = write_game_ids_csv(game_ids_df, output_root, args.season)
        print(f"Wrote {len(game_ids_df):,} game ids to {game_ids_path}")

    if args.build_game_ids_only:
        return

    if "game_id" not in game_ids_df.columns:
        raise ValueError("Input game ids CSV must contain a 'game_id' column")

    game_ids = game_ids_df["game_id"].dropna().astype(str).tolist()
    if args.max_games is not None:
        game_ids = game_ids[:args.max_games]

    total = len(game_ids)
    for index, game_id in enumerate(game_ids, start=1):
        print(f"[{index}/{total}] Scraping {game_id}")
        scrape_game(session, limiter, args.season, game_id, out_dirs, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
