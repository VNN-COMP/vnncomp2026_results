"""Helpers for interpreting benchmark instance fields from result CSV files."""

import ast
from pathlib import Path


def parse_network_field(network_field):
    """Return ``(network_name, path)`` pairs from a result CSV network field."""

    try:
        parsed = ast.literal_eval(network_field)
    except (SyntaxError, ValueError):
        return [(None, network_field)]

    if not isinstance(parsed, list):
        return [(None, network_field)]

    networks = []
    for entry in parsed:
        if not isinstance(entry, (tuple, list)) or len(entry) != 2:
            raise ValueError(f"expected network entries to be pairs, got {entry!r}")
        name, path = entry
        networks.append((str(name), str(path)))

    if not networks:
        raise ValueError("network list must not be empty")

    return networks


def network_stem(network_field):
    """Return the counterexample filename stem used by the benchmark runner."""

    networks = parse_network_field(network_field)
    if networks[0][0] is None:
        return Path(networks[0][1]).stem
    return "_".join(name for name, _ in networks)


def property_stem(property_field):
    return Path(property_field).stem


def benchmark_version_from_network_field(category, network_field):
    """Return the version directory present in a result CSV network field."""

    assert category[4] == "_", f"expected year at start of category: {category}"
    category_without_year = category[5:]

    for _, network_path in parse_network_field(network_field):
        parts = Path(network_path).parts
        for index, part in enumerate(parts):
            if (
                part == "benchmarks"
                and index + 2 < len(parts)
                and parts[index + 1] == category_without_year
            ):
                candidate = parts[index + 2]
                if candidate not in ("onnx", "vnnlib"):
                    return candidate

    return None


def resolve_benchmark_path(benchmark_dir, result_path, expected_directory):
    """Resolve a path stored in results.csv against a benchmark version directory."""

    path = Path(result_path)
    if path.is_file():
        return path

    parts = path.parts
    if expected_directory in parts:
        index = parts.index(expected_directory)
        candidate = benchmark_dir.joinpath(*parts[index:])
    else:
        candidate = benchmark_dir / expected_directory / path.name

    if candidate.is_file():
        return candidate

    gz_candidate = Path(f"{candidate}.gz")
    if gz_candidate.is_file():
        return gz_candidate

    raise FileNotFoundError(f"benchmark file not found: {candidate} or {gz_candidate}")
