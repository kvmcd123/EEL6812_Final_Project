from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import meteostat as ms
import numpy as np
import pandas as pd


FLORIDA_BOUNDS = {
    "min_lat": 24.0,
    "max_lat": 31.5,
    "min_lon": -87.7,
    "max_lon": -79.8,
}

SEARCH_LAT_VALUES = [15, 20, 25, 30, 35, 40, 45]
SEARCH_LON_VALUES = [-90, -80, -70, -60, -50, -40]

FEATURE_COLUMNS = ["temp", "rhum", "prcp", "wspd"]
FEATURE_FILE_NAMES = {
    "temp": "temp_data.csv",
    "rhum": "humidity_data.csv",
    "prcp": "rain_data.csv",
    "wspd": "wind_speed_data.csv",
}


def read_station_catalog(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None

    stations = pd.read_csv(path)
    required = {"id", "latitude", "longitude"}
    if stations.empty or not required.issubset(stations.columns):
        return None
    return stations


def discover_florida_stations() -> pd.DataFrame:
    all_stations: list[pd.DataFrame] = []

    for lat in SEARCH_LAT_VALUES:
        for lon in SEARCH_LON_VALUES:
            point = ms.Point(lat, lon)
            try:
                nearby = ms.stations.nearby(point, radius=2_000_000, limit=5_000)
            except Exception as exc:  # pragma: no cover - network/API behavior
                print(f"Skipped point ({lat}, {lon}): {exc}")
                continue

            if nearby is not None and not nearby.empty:
                all_stations.append(nearby.reset_index())

    if not all_stations:
        raise RuntimeError("No Florida stations were discovered from Meteostat.")

    stations = pd.concat(all_stations, ignore_index=True)
    stations["latitude"] = pd.to_numeric(stations["latitude"], errors="coerce")
    stations["longitude"] = pd.to_numeric(stations["longitude"], errors="coerce")

    stations = stations[
        stations["latitude"].between(FLORIDA_BOUNDS["min_lat"], FLORIDA_BOUNDS["max_lat"])
        & stations["longitude"].between(FLORIDA_BOUNDS["min_lon"], FLORIDA_BOUNDS["max_lon"])
    ]

    stations = stations.drop_duplicates(subset="id").sort_values("id").reset_index(drop=True)
    return stations


def load_or_create_station_catalog(
    primary_path: Path,
    fallback_path: Path | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    if not refresh:
        primary = read_station_catalog(primary_path)
        if primary is not None:
            print(f"Loaded {len(primary)} stations from {primary_path}")
            return primary

        if fallback_path is not None:
            fallback = read_station_catalog(fallback_path)
            if fallback is not None:
                print(f"Loaded {len(fallback)} stations from {fallback_path}")
                return fallback

    stations = discover_florida_stations()
    primary_path.parent.mkdir(parents=True, exist_ok=True)
    stations.to_csv(primary_path, index=False)
    print(f"Discovered and saved {len(stations)} stations to {primary_path}")
    return stations


def fetch_hourly_weather(stations: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    ms.config.block_large_requests = False
    station_ids = stations["id"].astype(str)

    ts = ms.hourly(
        station_ids,
        start,
        end,
        parameters=[ms.Parameter.TEMP, ms.Parameter.PRCP, ms.Parameter.WSPD, ms.Parameter.RHUM],
    )
    weather = ts.fetch()

    weather["temp"] = weather["temp"] * 9 / 5 + 32
    weather["wspd"] = weather["wspd"] * 0.6213711922
    return weather


def build_station_coord_lookup(stations: pd.DataFrame) -> pd.DataFrame:
    station_coords = stations.copy()
    station_coords["id"] = station_coords["id"].astype(str)
    station_coords["latitude"] = pd.to_numeric(station_coords["latitude"], errors="coerce")
    station_coords["longitude"] = pd.to_numeric(station_coords["longitude"], errors="coerce")
    station_coords = station_coords.dropna(subset=["latitude", "longitude"])
    station_coords = station_coords[["id", "latitude", "longitude"]].drop_duplicates("id")
    return station_coords.set_index("id")[["latitude", "longitude"]]


def fill_hourly_group_from_nearby(
    group: pd.DataFrame,
    coord_lookup: pd.DataFrame,
    numeric_cols: list[str],
    k: int = 3,
) -> pd.DataFrame:
    filled = group.copy()
    station_ids = (
        filled.index.get_level_values("station")
        if "station" in filled.index.names
        else filled.index.astype(str)
    )

    latitudes = station_ids.map(coord_lookup["latitude"])
    longitudes = station_ids.map(coord_lookup["longitude"])
    coords = np.column_stack([latitudes.to_numpy(dtype=float), longitudes.to_numpy(dtype=float)])

    for col in numeric_cols:
        values = np.array(pd.to_numeric(filled[col], errors="coerce"), dtype=float, copy=True)
        missing_idx = np.where(np.isnan(values))[0]
        known_idx = np.where(
            ~np.isnan(values)
            & ~np.isnan(coords[:, 0])
            & ~np.isnan(coords[:, 1])
        )[0]

        if len(known_idx) == 0:
            continue

        for i in missing_idx:
            if np.isnan(coords[i, 0]) or np.isnan(coords[i, 1]):
                continue

            dists = np.sqrt(
                (coords[known_idx, 0] - coords[i, 0]) ** 2
                + (coords[known_idx, 1] - coords[i, 1]) ** 2
            )
            nearest = known_idx[np.argsort(dists)[:k]]
            if len(nearest) > 0:
                values[i] = np.nanmean(values[nearest])

        filled[col] = values

    return filled


def fill_hourly_weather(weather: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    coord_lookup = build_station_coord_lookup(stations)
    numeric_cols = weather.select_dtypes(include=[np.number]).columns.tolist()

    if isinstance(weather.index, pd.MultiIndex) and "time" in weather.index.names:
        weather = weather.groupby(level="time", group_keys=False).apply(
            fill_hourly_group_from_nearby,
            coord_lookup=coord_lookup,
            numeric_cols=numeric_cols,
        )
    else:
        weather = fill_hourly_group_from_nearby(weather, coord_lookup, numeric_cols)

    if (
        isinstance(weather.index, pd.MultiIndex)
        and {"station", "time"}.issubset(weather.index.names)
        and "prcp" in weather.columns
    ):
        weather = weather.sort_index()
        weather["prcp"] = weather["prcp"].groupby(level="station", group_keys=True).apply(
            lambda s: pd.to_numeric(s.droplevel("station"), errors="coerce")
            .sort_index()
            .interpolate(method="time")
            .ffill()
            .bfill()
        )

    return weather


def fill_daily_matrix(
    daily_matrix: pd.DataFrame,
    station_coords: np.ndarray,
    k: int = 3,
) -> pd.DataFrame:
    filled = daily_matrix.copy()

    for col in filled.columns:
        values = np.array(pd.to_numeric(filled[col], errors="coerce"), dtype=float, copy=True)
        missing_idx = np.where(np.isnan(values))[0]
        known_idx = np.where(
            ~np.isnan(values)
            & ~np.isnan(station_coords[:, 0])
            & ~np.isnan(station_coords[:, 1])
        )[0]

        if len(known_idx) == 0:
            continue

        for i in missing_idx:
            if np.isnan(station_coords[i, 0]) or np.isnan(station_coords[i, 1]):
                continue

            dists = np.sqrt(
                (station_coords[known_idx, 0] - station_coords[i, 0]) ** 2
                + (station_coords[known_idx, 1] - station_coords[i, 1]) ** 2
            )
            nearest = known_idx[np.argsort(dists)[:k]]
            if len(nearest) > 0:
                values[i] = np.nanmean(values[nearest])

        filled[col] = values

    filled = filled.apply(
        lambda row: pd.Series(row, index=filled.columns).interpolate(limit_direction="both"),
        axis=1,
    )
    return filled


def export_daily_feature_datasets(
    weather: pd.DataFrame,
    stations: pd.DataFrame,
    output_root: Path,
) -> dict[str, dict[date, pd.DataFrame]]:
    if not isinstance(weather.index, pd.MultiIndex) or not {"station", "time"}.issubset(weather.index.names):
        raise ValueError("weather must have a MultiIndex with 'station' and 'time' levels.")

    all_station_ids = pd.Index(stations["id"].astype(str).drop_duplicates(), name="station")
    station_lookup = build_station_coord_lookup(stations).reindex(all_station_ids)
    station_coords = station_lookup[["latitude", "longitude"]].to_numpy(dtype=float)

    daily_feature_datasets: dict[str, dict[date, pd.DataFrame]] = {}

    for feature in FEATURE_COLUMNS:
        if feature not in weather.columns:
            continue

        feature_df = weather[[feature]].reset_index().copy()
        feature_df["station"] = feature_df["station"].astype(str)
        feature_df["date"] = feature_df["time"].dt.date
        feature_df["hour"] = feature_df["time"].dt.hour

        daily_feature_datasets[feature] = {}

        for date_value, group in feature_df.groupby("date"):
            daily_matrix = (
                group.pivot_table(
                    index="station",
                    columns="hour",
                    values=feature,
                    aggfunc="first",
                )
                .reindex(columns=range(24))
                .reindex(all_station_ids)
                .sort_index()
            )
            daily_matrix.columns = [f"hour_{hour:02d}" for hour in daily_matrix.columns]
            daily_matrix = fill_daily_matrix(daily_matrix, station_coords)
            daily_feature_datasets[feature][date_value] = daily_matrix

            date_folder = output_root / f"{date_value.month}_{date_value.day}_{str(date_value.year)[-2:]}"
            date_folder.mkdir(parents=True, exist_ok=True)
            output_path = date_folder / FEATURE_FILE_NAMES.get(feature, f"{feature}_data.csv")
            daily_matrix.to_csv(output_path)

    return daily_feature_datasets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch Meteostat hourly weather for Florida stations and export daily datasets."
    )
    parser.add_argument(
        "--stations-csv",
        type=Path,
        default=Path("florida_meteostat_stations.csv"),
        help="Primary station catalog path.",
    )
    parser.add_argument(
        "--fallback-stations-csv",
        type=Path,
        default=Path("florida_stations.csv"),
        help="Fallback station catalog path if the primary file is missing or empty.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("WeatherData"),
        help="Directory where daily datasets will be written.",
    )
    parser.add_argument("--start-year", type=int, default=2018)
    parser.add_argument("--start-month", type=int, default=1)
    parser.add_argument("--start-day", type=int, default=1)
    parser.add_argument("--end-year", type=int, default=2018)
    parser.add_argument("--end-month", type=int, default=12)
    parser.add_argument("--end-day", type=int, default=31)
    parser.add_argument(
        "--refresh-stations",
        action="store_true",
        help="Rediscover Florida stations from Meteostat and overwrite the primary station CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = date(args.start_year, args.start_month, args.start_day)
    end = date(args.end_year, args.end_month, args.end_day)

    stations = load_or_create_station_catalog(
        primary_path=args.stations_csv,
        fallback_path=args.fallback_stations_csv,
        refresh=args.refresh_stations,
    )
    print(f"Using {len(stations)} Florida stations")

    weather = fetch_hourly_weather(stations, start, end)
    print("Initial missing values:")
    print(weather.isna().sum())

    weather = fill_hourly_weather(weather, stations)
    print("\nMissing values after hourly fill:")
    print(weather.isna().sum())

    daily_feature_datasets = export_daily_feature_datasets(weather, stations, args.output_dir)

    print(f"\nSaved daily datasets under: {args.output_dir.resolve()}")
    print(f"Total stations expected in each dataset: {stations['id'].astype(str).nunique()}")
    for feature, date_dict in daily_feature_datasets.items():
        remaining_missing = sum(int(matrix.isna().sum().sum()) for matrix in date_dict.values())
        print(
            f"{feature}: {len(date_dict)} daily datasets, "
            f"remaining missing values = {remaining_missing}"
        )


if __name__ == "__main__":
    main()
