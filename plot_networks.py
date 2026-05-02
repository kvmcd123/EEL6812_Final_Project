import io
import math
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent / "GraphData"
WEB_MERCATOR_RADIUS = 6_378_137.0
MAX_MERCATOR_LAT = 85.05112878
OSM_TILE_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
OSM_USER_AGENT = "EEL6812-weather-network-plotter/1.0"


DATA_FOLDER = SCRIPT_DIR
OUTPUT_FOLDER = DATA_FOLDER / "network_plots"

NODE_CSV = None
EDGE_CSV = None
PLOT_NAME = "custom"

ZOOM_TO_NETWORK = False
MAX_NODES = None
NODE_SELECTION = "first"
MAX_EDGES = None
COUNT_LABEL = "nodes"
DISPLAY_COUNT = None

DPI = 180
TILE_ZOOM = 5
TILE_CACHE_FOLDER = DATA_FOLDER / "tile_cache" / "osm"
SHOW_PLOTS_AFTER_SAVING = False


class NetworkSpec:
    def __init__(self, name, node_path, edge_path):
        self.name = name
        self.node_path = node_path
        self.edge_path = edge_path


def lonlat_to_mercator(lon, lat):
    lon_array = np.asarray(lon, dtype=float)
    lat_array = np.clip(np.asarray(lat, dtype=float), -MAX_MERCATOR_LAT, MAX_MERCATOR_LAT)
    x = WEB_MERCATOR_RADIUS * np.radians(lon_array)
    y = WEB_MERCATOR_RADIUS * np.log(np.tan(np.pi / 4.0 + np.radians(lat_array) / 2.0))
    return x, y


def lonlat_to_tile(lon, lat, zoom):
    lat = float(np.clip(lat, -MAX_MERCATOR_LAT, MAX_MERCATOR_LAT))
    lat_rad = math.radians(lat)
    n = 2**zoom
    x_tile = int((lon + 180.0) / 360.0 * n)
    y_tile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return max(0, min(n - 1, x_tile)), max(0, min(n - 1, y_tile))


def tile_bounds_mercator(x_tile, y_tile, zoom):
    n = 2**zoom
    lon_left = x_tile / n * 360.0 - 180.0
    lon_right = (x_tile + 1) / n * 360.0 - 180.0
    lat_top = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * y_tile / n))))
    lat_bottom = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * (y_tile + 1) / n))))
    x_left, y_bottom = lonlat_to_mercator(lon_left, lat_bottom)
    x_right, y_top = lonlat_to_mercator(lon_right, lat_top)
    return float(x_left), float(x_right), float(y_bottom), float(y_top)


def load_osm_tile(x_tile, y_tile, zoom, cache_dir):
    tile_path = cache_dir / str(zoom) / str(x_tile) / f"{y_tile}.png"
    if not tile_path.exists():
        tile_path.parent.mkdir(parents=True, exist_ok=True)
        url = OSM_TILE_URL.format(z=zoom, x=x_tile, y=y_tile)
        request = Request(url, headers={"User-Agent": OSM_USER_AGENT})
        try:
            with urlopen(request, timeout=30) as response:
                tile_path.write_bytes(response.read())
        except (HTTPError, URLError, TimeoutError) as exc:
            raise RuntimeError(f"Could not fetch OpenStreetMap tile {zoom}/{x_tile}/{y_tile}: {exc}") from exc

    return mpimg.imread(io.BytesIO(tile_path.read_bytes()), format="png")


def draw_osm_basemap(ax, lon_min, lon_max, lat_min, lat_max, zoom, cache_dir):
    x_min_tile, y_max_tile = lonlat_to_tile(lon_min, lat_min, zoom)
    x_max_tile, y_min_tile = lonlat_to_tile(lon_max, lat_max, zoom)
    ax.set_facecolor("#dcecf5")

    for x_tile in range(x_min_tile, x_max_tile + 1):
        for y_tile in range(y_min_tile, y_max_tile + 1):
            image = load_osm_tile(x_tile, y_tile, zoom, cache_dir)
            extent = tile_bounds_mercator(x_tile, y_tile, zoom)
            ax.imshow(image, extent=extent, origin="upper", zorder=0, interpolation="bilinear")


def clean_id_series(series):
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all():
        return numeric.astype(int).astype(str)
    return series.astype(str)


def load_network(spec):
    nodes = pd.read_csv(spec.node_path)
    edges = pd.read_csv(spec.edge_path)

    required_node_cols = {"id", "latitude", "longitude"}
    required_edge_cols = {"source", "target"}
    if not required_node_cols.issubset(nodes.columns):
        raise ValueError(f"{spec.node_path} must contain columns {sorted(required_node_cols)}")
    if not required_edge_cols.issubset(edges.columns):
        raise ValueError(f"{spec.edge_path} must contain columns {sorted(required_edge_cols)}")

    nodes = nodes.copy()
    edges = edges.copy()
    nodes["plot_id"] = clean_id_series(nodes["id"])
    nodes["plot_sort"] = pd.to_numeric(nodes["id"], errors="coerce")
    edges["plot_source"] = clean_id_series(edges["source"])
    edges["plot_target"] = clean_id_series(edges["target"])
    nodes["latitude"] = pd.to_numeric(nodes["latitude"], errors="coerce")
    nodes["longitude"] = pd.to_numeric(nodes["longitude"], errors="coerce")
    nodes = nodes.dropna(subset=["latitude", "longitude"]).drop_duplicates("plot_id")
    return nodes, edges


def lonlat_bounds(nodes, full_us):
    if full_us:
        return -126.0, -66.0, 23.0, 50.5

    lon = nodes["longitude"]
    lat = nodes["latitude"]
    lon_pad = max(float(lon.max() - lon.min()) * 0.10, 1.0)
    lat_pad = max(float(lat.max() - lat.min()) * 0.10, 1.0)
    return (
        float(lon.min() - lon_pad),
        float(lon.max() + lon_pad),
        float(lat.min() - lat_pad),
        float(lat.max() + lat_pad),
    )


def set_map_extent(ax, bounds):
    lon_min, lon_max, lat_min, lat_max = bounds
    x_min, y_min = lonlat_to_mercator(lon_min, lat_min)
    x_max, y_max = lonlat_to_mercator(lon_max, lat_max)
    ax.set_xlim(float(x_min), float(x_max))
    ax.set_ylim(float(y_min), float(y_max))
    ax.set_aspect("equal", adjustable="box")


def plot_network(spec):
    nodes, edges = load_network(spec)
    if MAX_NODES is not None and len(nodes) > MAX_NODES:
        if NODE_SELECTION == "first":
            nodes = nodes.sort_values(["plot_sort", "plot_id"], na_position="last").head(MAX_NODES).copy()
        elif NODE_SELECTION == "random":
            nodes = nodes.sample(MAX_NODES, random_state=42).sort_values("plot_id").copy()
        else:
            raise ValueError(f"Unknown node_selection: {NODE_SELECTION}")

    nodes["x_mercator"], nodes["y_mercator"] = lonlat_to_mercator(
        nodes["longitude"].to_numpy(dtype=float),
        nodes["latitude"].to_numpy(dtype=float),
    )
    pos = {
        row.plot_id: (float(row.x_mercator), float(row.y_mercator))
        for row in nodes[["plot_id", "x_mercator", "y_mercator"]].itertuples(index=False)
    }

    drawable_edges = edges[
        edges["plot_source"].isin(pos) & edges["plot_target"].isin(pos)
    ].copy()
    if MAX_EDGES is not None and len(drawable_edges) > MAX_EDGES:
        drawable_edges = drawable_edges.sample(MAX_EDGES, random_state=42)

    fig, ax = plt.subplots(figsize=(12, 7.2))
    bounds = lonlat_bounds(nodes, full_us=not ZOOM_TO_NETWORK)
    draw_osm_basemap(
        ax,
        lon_min=bounds[0],
        lon_max=bounds[1],
        lat_min=bounds[2],
        lat_max=bounds[3],
        zoom=TILE_ZOOM,
        cache_dir=TILE_CACHE_FOLDER,
    )

    for edge in drawable_edges.itertuples(index=False):
        source = edge.plot_source
        target = edge.plot_target
        xs = [pos[source][0], pos[target][0]]
        ys = [pos[source][1], pos[target][1]]
        ax.plot(xs, ys, color="#56616f", linewidth=0.45, alpha=0.28, zorder=1)

    ax.scatter(
        nodes["x_mercator"],
        nodes["y_mercator"],
        s=18,
        c="#d1495b",
        edgecolors="black",
        linewidths=0.25,
        alpha=0.95,
        zorder=2,
    )

    set_map_extent(ax, bounds)
    ax.set_title(f"Weather Station Network", fontsize=16, pad=12)
    ax.set_xlabel("Web Mercator X")
    ax.set_ylabel("Web Mercator Y")
    ax.grid(False)

    plotted_count = len(nodes)
    count_to_show = plotted_count if DISPLAY_COUNT is None else DISPLAY_COUNT
    stats = f"{count_to_show:,} {COUNT_LABEL} | {len(drawable_edges):,} shown edges"
    if DISPLAY_COUNT is not None and DISPLAY_COUNT != plotted_count:
        stats += f" | {plotted_count:,} plotted nodes"
    ax.text(
        0.01,
        0.02,
        stats,
        transform=ax.transAxes,
        fontsize=10,
        bbox={"facecolor": "white", "edgecolor": "0.7", "alpha": 0.8, "pad": 4},
    )
    ax.text(
        0.99,
        0.02,
        "Map data © OpenStreetMap contributors",
        transform=ax.transAxes,
        ha="right",
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "0.75", "alpha": 0.8, "pad": 3},
    )

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_FOLDER / f"{spec.name}_network_us.png"
    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI)
    plt.close(fig)
    return output_path


def discover_networks(search_dirs):
    specs = []
    seen = set()

    for directory in search_dirs:
        for node_path in sorted(directory.glob("*nodeList.csv")):
            prefix = node_path.name[: -len("nodeList.csv")]
            edge_path = directory / f"{prefix}edgeList.csv"
            if not edge_path.exists():
                continue
            key = (node_path.resolve(), edge_path.resolve())
            if key in seen:
                continue
            seen.add(key)
            name = prefix.rstrip("_").replace("_meteo", "_meteo").replace(" ", "_")
            specs.append(NetworkSpec(name=name, node_path=node_path, edge_path=edge_path))

    return specs


def get_networks_to_plot():
    if (NODE_CSV is None) != (EDGE_CSV is None):
        raise ValueError("Set both NODE_CSV and EDGE_CSV, or leave both as None.")

    if NODE_CSV is not None and EDGE_CSV is not None:
        return [
            NetworkSpec(
                name=PLOT_NAME,
                node_path=NODE_CSV.resolve(),
                edge_path=EDGE_CSV.resolve(),
            )
        ]

    return discover_networks([DATA_FOLDER])


def show_saved_plot(image_path):
    image = mpimg.imread(image_path)
    fig, ax = plt.subplots(figsize=(12, 7.2))
    ax.imshow(image)
    ax.set_title(image_path.name)
    ax.axis("off")
    plt.show()


def run_plots():
    specs = get_networks_to_plot()
    if not specs:
        raise ValueError("No nodeList/edgeList CSV pairs were found.")

    print(f"Found {len(specs)} network(s) to plot.")
    saved_files = []

    for spec in specs:
        try:
            output_path = plot_network(spec)
            saved_files.append(output_path)
            print(f"Saved {spec.name} -> {output_path}")

            if SHOW_PLOTS_AFTER_SAVING:
                show_saved_plot(output_path)
        except Exception as exc:
            print(f"Skipped {spec.name}: {exc}")

    return saved_files


saved_plot_files = run_plots()
