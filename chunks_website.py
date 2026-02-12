#!/usr/bin/env python3
"""Chunk SIEM logs by website and time window for RAG ingestion."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

# Tunable defaults for quickly generating chunk variations.
DEFAULT_INPUT_GLOB = "Logs/*.log*"
DEFAULT_OUTPUT_DIRNAME = "LogChunksByWebsite"
DEFAULT_INTERVAL_MINUTES = 15
DEFAULT_INTERVALS = "10,5,2"
DEFAULT_LEVELS = "error,crit,alert,emerg"
DEFAULT_YEAR = datetime.now().year
DEFAULT_MAX_EVENTS_PER_CHUNK = 200
DEFAULT_MAX_TEXT_CHARS = 12000

# Nginx error log
NGINX_ERROR_RE = re.compile(
    r"^(?P<ts>\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}) \[(?P<level>[^\]]+)\] (?P<rest>.*)$"
)
SERVER_RE = re.compile(r"server:\s*([^,]+)")
HOST_RE = re.compile(r'host:\s*"([^"]+)"')
CLIENT_RE = re.compile(r"client:\s*([^,]+)")
REQUEST_RE = re.compile(r'request:\s*"([^"]+)"')
MODSEC_HOST_RE = re.compile(r'\[hostname\s+"([^"]+)"\]')
MODSEC_ID_RE = re.compile(r'\[id\s+"([^"]+)"\]')
MODSEC_MSG_RE = re.compile(r'\[msg\s+"([^"]+)"\]')
CODE_RE = re.compile(r"\bcode\s+(\d{3})\b", re.IGNORECASE)

# Nginx access log
ACCESS_RE = re.compile(
    r'^(?P<client>\S+)\s+\S+\s+\S+\s+\[(?P<ts>[^\]]+)\]\s+"(?P<request>[^"]*)"\s+'
    r'(?P<status>\d{3})\s+\S+\s+"[^"]*"\s+"[^"]*"\s+"(?P<host>[^"]*)"'
)

# BSD syslog format, e.g. "Feb 11 00:14:00 tracerSENSE ..."
SYSLOG_RE = re.compile(
    r"^(?P<mon>[A-Z][a-z]{2})\s+(?P<day>\d{1,2})\s+(?P<time>\d{2}:\d{2}:\d{2})\s+(?P<host>\S+)\s+(?P<rest>.*)$"
)

# ModSecurity audit log markers
MODSEC_MARKER_RE = re.compile(r"^---([A-Za-z0-9]+)---([A-Z])--$")
MODSEC_A_LINE_RE = re.compile(
    r"^\[(?P<dt>[^\]]+)\]\s+\S+\s+(?P<src_ip>\S+)\s+\S+\s+(?P<dst_ip>\S+)\s+\S+"
)

IPV4_RE = re.compile(r"^(\d{1,3}\.){3}\d{1,3}$")
HOSTNAME_RE = re.compile(r"^[a-z0-9.-]+$")

MONTHS = {
    "Jan": 1,
    "Feb": 2,
    "Mar": 3,
    "Apr": 4,
    "May": 5,
    "Jun": 6,
    "Jul": 7,
    "Aug": 8,
    "Sep": 9,
    "Oct": 10,
    "Nov": 11,
    "Dec": 12,
}


@dataclass
class ParsedEvent:
    timestamp: datetime
    level: str
    raw_line: str
    website: str
    website_raw: str | None
    client_ip: str | None
    request: str | None
    source_type: str
    method: str | None = None
    path: str | None = None
    status: int | None = None
    rule_id: str | None = None
    signature: str | None = None
    src_ip: str | None = None
    dst_ip: str | None = None


@dataclass
class ChunkStats:
    methods: set[str] = field(default_factory=set)
    paths: set[str] = field(default_factory=set)
    statuses: set[int] = field(default_factory=set)
    rule_ids: set[str] = field(default_factory=set)
    signatures: set[str] = field(default_factory=set)
    src_ips: set[str] = field(default_factory=set)
    dst_ips: set[str] = field(default_factory=set)


def slugify(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", value.strip().lower())
    return cleaned.strip("._-") or "unknown"


def normalize_server(server: str | None) -> str | None:
    if not server:
        return None
    s = server.strip().strip('"')
    if not s:
        return None
    if s.startswith("0.0.0.0:"):
        return None
    if s.startswith("[::]:"):
        return None
    return s


def canonicalize_website(website: str | None) -> tuple[str, str | None]:
    if website is None:
        return "unknown", None

    raw = website.strip().strip('"')
    if not raw or raw == "-":
        return "unknown", raw or None

    low = raw.lower()

    # Collapse obvious injection/test payload host values.
    if "${" in low or "jndi:" in low or "ldap://" in low:
        return "suspicious_host", raw
    if "/" in raw and not raw.startswith("["):
        return "suspicious_host", raw
    if " " in raw:
        return "suspicious_host", raw

    host_only = low
    if ":" in host_only and host_only.count(":") == 1:
        maybe_host, maybe_port = host_only.rsplit(":", 1)
        if maybe_port.isdigit():
            host_only = maybe_host

    if IPV4_RE.match(host_only) or HOSTNAME_RE.match(host_only):
        return host_only, raw

    return "unknown", raw


def parse_request_line(request: str | None) -> tuple[str | None, str | None]:
    if not request:
        return None, None
    parts = request.split()
    if len(parts) < 2:
        return None, None
    return parts[0], parts[1]


def parse_access_time(ts: str) -> datetime | None:
    # Example: 09/Feb/2026:00:00:05 -0700
    try:
        return datetime.strptime(ts, "%d/%b/%Y:%H:%M:%S %z").replace(tzinfo=None)
    except ValueError:
        return None


def parse_syslog_time(mon: str, day: str, hhmmss: str, default_year: int) -> datetime | None:
    month = MONTHS.get(mon)
    if not month:
        return None
    try:
        day_int = int(day)
        hh, mm, ss = [int(x) for x in hhmmss.split(":")]
        return datetime(default_year, month, day_int, hh, mm, ss)
    except (ValueError, TypeError):
        return None


def extract_error_fields(rest: str) -> dict:
    server_match = SERVER_RE.search(rest)
    host_match = HOST_RE.search(rest)
    client_match = CLIENT_RE.search(rest)
    request_match = REQUEST_RE.search(rest)
    modsec_host_match = MODSEC_HOST_RE.search(rest)
    modsec_id_match = MODSEC_ID_RE.search(rest)
    modsec_msg_match = MODSEC_MSG_RE.search(rest)
    code_match = CODE_RE.search(rest)

    server = normalize_server(server_match.group(1).strip()) if server_match else None
    host = host_match.group(1).strip() if host_match else None
    modsec_host = modsec_host_match.group(1).strip() if modsec_host_match else None

    website_raw = server or host or modsec_host
    website, website_raw_out = canonicalize_website(website_raw)

    request = request_match.group(1).strip() if request_match else None
    method, path = parse_request_line(request)

    return {
        "website": website,
        "website_raw": website_raw_out,
        "client_ip": client_match.group(1).strip() if client_match else None,
        "request": request,
        "method": method,
        "path": path,
        "status": int(code_match.group(1)) if code_match else None,
        "rule_id": modsec_id_match.group(1) if modsec_id_match else None,
        "signature": modsec_msg_match.group(1) if modsec_msg_match else None,
    }


def parse_nginx_error_line(line: str, include_levels: set[str]) -> ParsedEvent | None:
    m = NGINX_ERROR_RE.match(line)
    if not m:
        return None

    level = m.group("level").strip().lower()
    if include_levels and level not in include_levels:
        return None

    timestamp = datetime.strptime(m.group("ts"), "%Y/%m/%d %H:%M:%S")
    fields = extract_error_fields(m.group("rest"))

    return ParsedEvent(
        timestamp=timestamp,
        level=level,
        raw_line=line.rstrip("\n"),
        website=fields["website"],
        website_raw=fields["website_raw"],
        client_ip=fields["client_ip"],
        request=fields["request"],
        source_type="nginx_error",
        method=fields["method"],
        path=fields["path"],
        status=fields["status"],
        rule_id=fields["rule_id"],
        signature=fields["signature"],
    )


def parse_access_line(line: str) -> ParsedEvent | None:
    m = ACCESS_RE.match(line)
    if not m:
        return None

    timestamp = parse_access_time(m.group("ts"))
    if not timestamp:
        return None

    website, website_raw = canonicalize_website(m.group("host").strip())
    request = m.group("request").strip() or None
    method, path = parse_request_line(request)

    return ParsedEvent(
        timestamp=timestamp,
        level="access",
        raw_line=line.rstrip("\n"),
        website=website,
        website_raw=website_raw,
        client_ip=m.group("client"),
        request=request,
        source_type="nginx_access",
        method=method,
        path=path,
        status=int(m.group("status")),
    )


def parse_syslog_line(line: str, default_year: int) -> ParsedEvent | None:
    m = SYSLOG_RE.match(line)
    if not m:
        return None

    timestamp = parse_syslog_time(m.group("mon"), m.group("day"), m.group("time"), default_year)
    if not timestamp:
        return None

    website, website_raw = canonicalize_website(m.group("host"))
    return ParsedEvent(
        timestamp=timestamp,
        level="syslog",
        raw_line=line.rstrip("\n"),
        website=website,
        website_raw=website_raw,
        client_ip=None,
        request=None,
        source_type="syslog",
    )


def parse_modsec_audit_events(path: Path) -> list[ParsedEvent]:
    events: list[ParsedEvent] = []
    tx_id: str | None = None
    current_section: str | None = None
    a_line: str | None = None
    b_lines: list[str] = []

    def flush_transaction() -> None:
        nonlocal a_line, b_lines
        if not tx_id or not a_line:
            a_line = None
            b_lines = []
            return

        am = MODSEC_A_LINE_RE.match(a_line)
        if not am:
            a_line = None
            b_lines = []
            return

        try:
            dt = datetime.strptime(am.group("dt"), "%d/%b/%Y:%H:%M:%S %z").replace(tzinfo=None)
        except ValueError:
            a_line = None
            b_lines = []
            return

        src_ip = am.group("src_ip")
        dst_ip = am.group("dst_ip")
        website, website_raw = canonicalize_website(dst_ip)

        request_line = None
        for bl in b_lines:
            if bl and not bl.startswith(
                ("Host:", "User-Agent:", "Accept:", "Content-", "Connection:", "Pragma:")
            ):
                request_line = bl.strip()
                break

        method, req_path = parse_request_line(request_line)

        text_parts = [f"---{tx_id}---A--", a_line]
        if b_lines:
            text_parts.append(f"---{tx_id}---B--")
            text_parts.extend(b_lines)

        events.append(
            ParsedEvent(
                timestamp=dt,
                level="audit",
                raw_line="\n".join(text_parts),
                website=website,
                website_raw=website_raw,
                client_ip=src_ip,
                request=request_line,
                source_type="modsec_audit",
                method=method,
                path=req_path,
                src_ip=src_ip,
                dst_ip=dst_ip,
            )
        )

        a_line = None
        b_lines = []

    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            marker = MODSEC_MARKER_RE.match(line)
            if marker:
                marker_tx = marker.group(1)
                marker_section = marker.group(2)

                if marker_section == "A":
                    flush_transaction()
                    tx_id = marker_tx
                    current_section = "A"
                    continue

                if tx_id == marker_tx:
                    if marker_section in {"Z", "K"}:
                        flush_transaction()
                        tx_id = None
                        current_section = None
                        continue
                    current_section = marker_section
                    continue

            if tx_id and current_section == "A" and not a_line and line:
                a_line = line
            elif tx_id and current_section == "B":
                b_lines.append(line)

    flush_transaction()
    return events


def parse_events_from_file(
    log_file: Path,
    include_levels: set[str],
    default_year: int,
) -> list[ParsedEvent]:
    if log_file.name.startswith("modsec_audit.log"):
        return parse_modsec_audit_events(log_file)

    events: list[ParsedEvent] = []
    with log_file.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line.strip():
                continue

            event = parse_nginx_error_line(line, include_levels)
            if event:
                events.append(event)
                continue

            event = parse_access_line(line)
            if event:
                events.append(event)
                continue

            event = parse_syslog_line(line, default_year)
            if event:
                events.append(event)
                continue

    return events


def bucket_start(ts: datetime, interval_minutes: int) -> datetime:
    seconds = interval_minutes * 60
    epoch = int(ts.timestamp())
    floor_epoch = epoch - (epoch % seconds)
    return datetime.fromtimestamp(floor_epoch)


def iter_log_files(input_dir: Path, pattern: str) -> Iterable[Path]:
    return sorted(p for p in input_dir.glob(pattern) if p.is_file())


def summarize_events(events: list[ParsedEvent]) -> tuple[str, ChunkStats, list[dict], int]:
    stats = ChunkStats()
    counts: dict[str, int] = {}
    order: list[str] = []

    for event in events:
        if event.raw_line not in counts:
            counts[event.raw_line] = 0
            order.append(event.raw_line)
        counts[event.raw_line] += 1

        if event.method:
            stats.methods.add(event.method)
        if event.path:
            stats.paths.add(event.path)
        if event.status is not None:
            stats.statuses.add(event.status)
        if event.rule_id:
            stats.rule_ids.add(event.rule_id)
        if event.signature:
            stats.signatures.add(event.signature)
        if event.src_ip:
            stats.src_ips.add(event.src_ip)
        if event.dst_ip:
            stats.dst_ips.add(event.dst_ip)

    dedup_lines: list[dict] = []
    text_lines: list[str] = []
    repeat_total = 0
    for line in order:
        c = counts[line]
        dedup_lines.append({"count": c, "line": line})
        if c > 1:
            repeat_total += c - 1
            text_lines.append(f"[x{c}] {line}")
        else:
            text_lines.append(line)

    summary_parts = [
        f"events={len(events)}",
        f"unique_lines={len(order)}",
        f"repeated_events={repeat_total}",
    ]
    if stats.methods:
        summary_parts.append(f"methods={','.join(sorted(stats.methods)[:8])}")
    if stats.statuses:
        summary_parts.append(
            "statuses=" + ",".join(str(s) for s in sorted(stats.statuses)[:10])
        )
    if stats.rule_ids:
        summary_parts.append(f"rule_ids={','.join(sorted(stats.rule_ids)[:10])}")

    return " | ".join(summary_parts), stats, dedup_lines, repeat_total


def split_events_with_caps(
    events: list[ParsedEvent],
    max_events_per_chunk: int,
    max_text_chars: int,
) -> list[list[ParsedEvent]]:
    chunks: list[list[ParsedEvent]] = []
    current: list[ParsedEvent] = []
    current_chars = 0

    for event in events:
        event_chars = len(event.raw_line) + 1
        must_split = False
        if current and len(current) >= max_events_per_chunk:
            must_split = True
        if current and current_chars + event_chars > max_text_chars:
            must_split = True

        if must_split:
            chunks.append(current)
            current = []
            current_chars = 0

        current.append(event)
        current_chars += event_chars

    if current:
        chunks.append(current)

    return chunks


def build_chunks(
    log_file: Path,
    interval_minutes: int,
    include_levels: set[str],
    default_year: int,
    max_events_per_chunk: int,
    max_text_chars: int,
) -> list[dict]:
    grouped: dict[tuple[str, datetime], list[ParsedEvent]] = defaultdict(list)
    events = parse_events_from_file(log_file, include_levels, default_year)

    for event in events:
        grouped[(event.website, bucket_start(event.timestamp, interval_minutes))].append(event)

    chunks: list[dict] = []
    for (website, start), grouped_events in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1])):
        grouped_events.sort(key=lambda e: e.timestamp)
        partitions = split_events_with_caps(grouped_events, max_events_per_chunk, max_text_chars)

        for part_idx, part_events in enumerate(partitions, start=1):
            end = start + timedelta(minutes=interval_minutes) - timedelta(seconds=1)
            client_ips = sorted({e.client_ip for e in part_events if e.client_ip})
            levels = sorted({e.level for e in part_events})
            source_types = sorted({e.source_type for e in part_events})
            website_raw_values = sorted({e.website_raw for e in part_events if e.website_raw})

            summary, stats, dedup_lines, repeated_events = summarize_events(part_events)
            text = "\n".join(item["line"] if item["count"] == 1 else f"[x{item['count']}] {item['line']}" for item in dedup_lines)
            sample_requests = [e.request for e in part_events if e.request][:10]

            chunk_id = (
                f"{log_file.name}:{slugify(website)}:{start.strftime('%Y%m%dT%H%M%S')}:p{part_idx}"
            )

            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "source_file": log_file.name,
                    "website": website,
                    "website_raw_values": website_raw_values[:20],
                    "time_start": start.isoformat(sep=" "),
                    "time_end": end.isoformat(sep=" "),
                    "time_interval": f"{start.isoformat(sep=' ')}  to  {end.isoformat(sep=' ')}",
                    "interval_minutes": interval_minutes,
                    "partition_index": part_idx,
                    "partition_count": len(partitions),
                    "event_count": len(part_events),
                    "unique_line_count": len(dedup_lines),
                    "repeated_event_count": repeated_events,
                    "levels": levels,
                    "source_types": source_types,
                    "client_ips": client_ips,
                    "sample_requests": sample_requests,
                    "methods": sorted(stats.methods),
                    "paths": sorted(stats.paths)[:50],
                    "statuses": sorted(stats.statuses),
                    "rule_ids": sorted(stats.rule_ids),
                    "signatures": sorted(stats.signatures)[:20],
                    "src_ips": sorted(stats.src_ips),
                    "dst_ips": sorted(stats.dst_ips),
                    "summary": summary,
                    "dedup_lines": dedup_lines,
                    "text": text,
                }
            )

    return chunks


def write_chunks_for_file(chunks: list[dict], log_file: Path, output_root: Path) -> None:
    file_output = output_root / log_file.name
    if file_output.exists():
        shutil.rmtree(file_output)
    file_output.mkdir(parents=True, exist_ok=True)

    def write_jsonl_with_interval_spacers(path: Path, items: list[dict]) -> None:
        prev_interval = None
        with path.open("w", encoding="utf-8") as fh:
            for item in items:
                current_interval = item.get("time_interval")
                if prev_interval is not None and current_interval != prev_interval:
                    fh.write("\n")
                fh.write(json.dumps(item, ensure_ascii=True) + "\n")
                prev_interval = current_interval

    all_chunks_path = file_output / "all_chunks.jsonl"
    write_jsonl_with_interval_spacers(all_chunks_path, chunks)

    per_website: dict[str, list[dict]] = defaultdict(list)
    for chunk in chunks:
        per_website[chunk["website"]].append(chunk)

    websites_dir = file_output / "by_website"
    websites_dir.mkdir(parents=True, exist_ok=True)

    for website, website_chunks in sorted(per_website.items(), key=lambda x: x[0]):
        website_file = websites_dir / f"{slugify(website)}.jsonl"
        write_jsonl_with_interval_spacers(website_file, website_chunks)

    summary = {
        "source_file": log_file.name,
        "chunk_count": len(chunks),
        "website_count": len(per_website),
        "websites": sorted(per_website.keys()),
        "outputs": {
            "all_chunks": str(all_chunks_path),
            "by_website_dir": str(websites_dir),
        },
    }
    with (file_output / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chunk SIEM logs by website and fixed time windows for RAG."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing log files.",
    )
    parser.add_argument(
        "--pattern",
        default=DEFAULT_INPUT_GLOB,
        help=f"Glob pattern for input files (default: {DEFAULT_INPUT_GLOB}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / DEFAULT_OUTPUT_DIRNAME,
        help="Root directory for chunked output.",
    )
    parser.add_argument(
        "--interval-minutes",
        type=int,
        default=DEFAULT_INTERVAL_MINUTES,
        help=f"Time bucket size in minutes (default: {DEFAULT_INTERVAL_MINUTES}).",
    )
    parser.add_argument(
        "--intervals",
        default=DEFAULT_INTERVALS,
        help=(
            "Comma-separated interval minutes to generate in one run. "
            f"Default: {DEFAULT_INTERVALS}. Example: 10,5,2"
        ),
    )
    parser.add_argument(
        "--max-events-per-chunk",
        type=int,
        default=DEFAULT_MAX_EVENTS_PER_CHUNK,
        help=(
            "Hard cap on number of events in a single output chunk "
            f"(default: {DEFAULT_MAX_EVENTS_PER_CHUNK})."
        ),
    )
    parser.add_argument(
        "--max-text-chars",
        type=int,
        default=DEFAULT_MAX_TEXT_CHARS,
        help=(
            "Hard cap on approximated raw text characters in a single output chunk "
            f"(default: {DEFAULT_MAX_TEXT_CHARS})."
        ),
    )
    parser.add_argument(
        "--levels",
        default=DEFAULT_LEVELS,
        help=(
            "Comma-separated nginx error levels to include. "
            f"Used for nginx error logs only (default: {DEFAULT_LEVELS})."
        ),
    )
    parser.add_argument(
        "--default-year",
        type=int,
        default=DEFAULT_YEAR,
        help=(
            "Year used for syslog-like lines that do not include a year "
            f"(default: {DEFAULT_YEAR})."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    interval_values = []
    for raw in args.intervals.split(","):
        raw = raw.strip()
        if not raw:
            continue
        value = int(raw)
        if value <= 0:
            raise ValueError("--intervals values must be > 0")
        interval_values.append(value)
    if not interval_values:
        raise ValueError("--intervals must include at least one positive integer")

    # Keep backward compatibility: if user explicitly overrides --interval-minutes
    # to a non-default value, generate only that interval.
    if args.interval_minutes != DEFAULT_INTERVAL_MINUTES:
        if args.interval_minutes <= 0:
            raise ValueError("--interval-minutes must be > 0")
        interval_values = [args.interval_minutes]
    if args.max_events_per_chunk <= 0:
        raise ValueError("--max-events-per-chunk must be > 0")
    if args.max_text_chars <= 0:
        raise ValueError("--max-text-chars must be > 0")

    include_levels = {lvl.strip().lower() for lvl in args.levels.split(",") if lvl.strip()}

    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    files = list(iter_log_files(input_dir, args.pattern))
    if not files:
        print(f"No files found in {input_dir} matching pattern {args.pattern!r}")
        return

    print(
        f"Found {len(files)} file(s). Building chunks for intervals: "
        + ", ".join(f"{v}m" for v in interval_values)
    )
    for interval_minutes in interval_values:
        interval_output_dir = output_dir / f"{interval_minutes}min"
        interval_output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Interval {interval_minutes}m -> {interval_output_dir}")

        for log_file in files:
            chunks = build_chunks(
                log_file,
                interval_minutes,
                include_levels,
                args.default_year,
                args.max_events_per_chunk,
                args.max_text_chars,
            )
            write_chunks_for_file(chunks, log_file, interval_output_dir)
            print(
                f"- {log_file.name}: wrote {len(chunks)} chunk(s) to "
                f"{interval_output_dir / log_file.name}"
            )


if __name__ == "__main__":
    main()
