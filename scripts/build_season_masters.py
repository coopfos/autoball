from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path


SCRAPE_TYPES = ("pbp", "batting", "pitching")
PBP_DATA_COLUMNS = [
    "Inn", "Score", "Out", "RoB", "Pit(cnt)", "R/O", "@Bat", "Batter",
    "Pitcher", "wWPA", "wWE", "Play Description",
]
PBP_HALF_INNING_CODE = re.compile(r"^[tb]\d+$")
PBP_SUMMARY_PATTERN = re.compile(
    r"^\d+\s+run[s]?,\s+\d+\s+hit[s]?,\s+\d+\s+error[s]?,\s+\d+\s+LOB\.",
)
BATTING_DATA_COLUMNS = [
    "Batting", "AB", "R", "H", "RBI", "BB", "SO", "PA", "BA", "OBP", "SLG",
    "OPS", "Pit", "Str", "WPA", "aLI", "WPA+", "WPA-", "cWPA", "acLI",
    "RE24", "PO", "A", "Details",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Roll up per-game Baseball Reference scrape CSVs into season master CSVs."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Root data directory containing season subdirectories.",
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        default=("2024", "2025"),
        help="Season directories to roll up.",
    )
    return parser.parse_args()


def load_game_ids(path: Path) -> set[str]:
    with path.open(newline="") as handle:
        return {row["game_id"] for row in csv.DictReader(handle)}


def list_csvs(path: Path) -> list[Path]:
    return sorted(file for file in path.glob("*.csv") if file.is_file())


def game_id_from_path(path: Path) -> str:
    return path.stem.split("_", 1)[0]


def validate_files(files: list[Path], expected_ids: set[str], scrape_type: str) -> None:
    expected_count = 1 if scrape_type == "pbp" else 2
    counter = Counter(game_id_from_path(path) for path in files)

    missing = sorted(expected_ids - set(counter))
    extra = sorted(set(counter) - expected_ids)
    wrong_count = sorted((game_id, count) for game_id, count in counter.items() if count != expected_count)

    if missing or extra or wrong_count:
        parts: list[str] = []
        if missing:
            parts.append(f"missing={missing[:10]}")
        if extra:
            parts.append(f"extra={extra[:10]}")
        if wrong_count:
            parts.append(f"wrong_count={wrong_count[:10]}")
        raise ValueError(f"{scrape_type} validation failed: {'; '.join(parts)}")


def collect_fieldnames(files: list[Path]) -> list[str]:
    fieldnames: list[str] = []
    seen: set[str] = set()
    for path in files:
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                continue
            for name in reader.fieldnames:
                if name not in seen:
                    seen.add(name)
                    fieldnames.append(name)
    return fieldnames


def build_master(season_dir: Path, scrape_type: str, expected_ids: set[str]) -> tuple[Path, int]:
    source_dir = season_dir / scrape_type
    files = list_csvs(source_dir)
    validate_files(files, expected_ids, scrape_type)

    fieldnames = collect_fieldnames(files)
    output_path = season_dir / f"{scrape_type}_master.csv"
    row_count = 0

    with output_path.open("w", newline="") as out_handle:
        writer = csv.DictWriter(out_handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for path in files:
            with path.open(newline="") as in_handle:
                reader = csv.DictReader(in_handle)
                for row in reader:
                    if should_skip_row(scrape_type, row):
                        continue
                    writer.writerow({name: row.get(name, "") for name in fieldnames})
                    row_count += 1

    return output_path, row_count


def should_skip_row(scrape_type: str, row: dict[str, str]) -> bool:
    if scrape_type == "pbp":
        return is_pbp_artifact(row)
    if scrape_type == "batting":
        return is_empty_batting_row(row)
    return False


def is_pbp_artifact(row: dict[str, str]) -> bool:
    inn = (row.get("Inn") or "").strip()
    play_description = (row.get("Play Description") or "").strip()
    non_empty_values = [value.strip() for column in PBP_DATA_COLUMNS if (value := row.get(column) or "").strip()]

    # Baseball Reference adds a repeated full-sentence half-inning header across every play-by-play column.
    if inn and not PBP_HALF_INNING_CODE.fullmatch(inn):
        return len(set(non_empty_values)) == 1

    # Each half inning also ends with a plain-language summary row rather than an event row.
    if PBP_SUMMARY_PATTERN.match(play_description):
        return True

    return False


def is_empty_batting_row(row: dict[str, str]) -> bool:
    return all(not (row.get(column) or "").strip() for column in BATTING_DATA_COLUMNS)


def main() -> None:
    args = parse_args()
    for season in args.seasons:
        season_dir = args.data_root / season
        game_ids_path = args.data_root / f"br_game_ids_{season}.csv"
        if not season_dir.exists():
            raise FileNotFoundError(f"Missing season directory: {season_dir}")
        if not game_ids_path.exists():
            raise FileNotFoundError(f"Missing game-id manifest: {game_ids_path}")

        expected_ids = load_game_ids(game_ids_path)
        print(f"[{season}] game_ids={len(expected_ids)}")
        for scrape_type in SCRAPE_TYPES:
            output_path, row_count = build_master(season_dir, scrape_type, expected_ids)
            print(f"  wrote {output_path} rows={row_count}")


if __name__ == "__main__":
    main()
