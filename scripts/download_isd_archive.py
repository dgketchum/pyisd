"""
Bulk download ISD raw archive from s3://noaa-isd-pds/data/ to local storage.

Downloads .gz station-year files with threaded I/O, manifest-based resume,
and time-window scheduling so downloads can run unattended during off-hours.

Layout
------
  {dest}/{YYYY}/{USAF}-{WBAN}-{YYYY}.gz
  {dest}/manifest.parquet

Usage
-----
  uv run python scripts/download_isd_archive.py \
      --dest /nas/climate/isd/raw --workers 16 \
      --schedule "00:00-08:00" --weekend-free
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

import boto3
import pandas as pd
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import ClientError

# ── constants ────────────────────────────────────────────────────────────────

S3_BUCKET = "noaa-isd-pds"
DATA_PREFIX = "data/"
DEFAULT_DEST = "/nas/climate/isd/raw"
_MAX_RETRIES = 3


# ── S3 helpers ───────────────────────────────────────────────────────────────


def _make_s3_client():
    return boto3.client("s3", config=Config(signature_version=UNSIGNED))


def _list_year_keys(s3, year: int) -> list[str]:
    """List all object keys under data/{year}/ in the ISD bucket."""
    prefix = f"{DATA_PREFIX}{year}/"
    keys: list[str] = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith(".gz"):
                keys.append(key)
    return keys


def _local_path(dest: str, key: str) -> str:
    """Map S3 key data/YYYY/filename.gz → {dest}/YYYY/filename.gz."""
    # key looks like "data/2024/010010-99999-2024.gz"
    relative = key[len(DATA_PREFIX) :]  # "2024/010010-99999-2024.gz"
    return os.path.join(dest, relative)


# ── scheduling ───────────────────────────────────────────────────────────────


def _parse_schedule(schedule: str) -> tuple[int, int]:
    """Parse 'HH:MM-HH:MM' into (start_minutes, end_minutes) from midnight."""
    start_s, end_s = schedule.split("-")
    sh, sm = start_s.strip().split(":")
    eh, em = end_s.strip().split(":")
    return int(sh) * 60 + int(sm), int(eh) * 60 + int(em)


def _in_window(schedule: str | None, weekend_free: bool) -> bool:
    if schedule is None:
        return True
    now = datetime.now()
    if weekend_free and now.weekday() >= 5:
        return True
    start_m, end_m = _parse_schedule(schedule)
    cur_m = now.hour * 60 + now.minute
    if start_m <= end_m:
        return start_m <= cur_m < end_m
    # crosses midnight
    return cur_m >= start_m or cur_m < end_m


def _seconds_until_window(schedule: str, weekend_free: bool) -> float:
    """Seconds to sleep before the next download window opens."""
    now = datetime.now()

    start_m, _ = _parse_schedule(schedule)
    start_time = datetime.combine(now.date(), datetime.min.time()) + timedelta(
        minutes=start_m
    )
    if start_time <= now:
        start_time += timedelta(days=1)
    wait_weeknight = (start_time - now).total_seconds()

    if not weekend_free:
        return wait_weeknight

    for day_offset in range(1, 7):
        future = now + timedelta(days=day_offset)
        if future.weekday() >= 5:
            target = datetime.combine(future.date(), datetime.min.time())
            wait_weekend = (target - now).total_seconds()
            return min(wait_weeknight, wait_weekend)

    return wait_weeknight


def _wait_for_window(schedule: str | None, weekend_free: bool) -> None:
    if schedule is None or _in_window(schedule, weekend_free):
        return
    wait = _seconds_until_window(schedule, weekend_free)
    resume_at = datetime.now() + timedelta(seconds=wait)
    print(f"  Outside schedule — sleeping until {resume_at:%Y-%m-%d %H:%M}")
    sys.stdout.flush()
    time.sleep(wait)


# ── download ─────────────────────────────────────────────────────────────────


def _download_one(s3, key: str, dest_path: str) -> dict:
    """Download a single .gz file from S3 with retries.  Returns manifest row."""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    for attempt in range(_MAX_RETRIES):
        try:
            s3.download_file(S3_BUCKET, key, dest_path)
            size = os.path.getsize(dest_path)
            return {"local_path": dest_path, "size_bytes": size, "status": "done"}
        except ClientError as e:
            code = e.response["Error"]["Code"]
            if code in ("404", "NoSuchKey"):
                return {"local_path": dest_path, "size_bytes": 0, "status": "missing"}
            if attempt == _MAX_RETRIES - 1:
                return {
                    "local_path": dest_path,
                    "size_bytes": 0,
                    "status": f"error_{code}",
                }
            time.sleep(2**attempt)
        except Exception:
            if attempt == _MAX_RETRIES - 1:
                return {"local_path": dest_path, "size_bytes": 0, "status": "error"}
            time.sleep(2**attempt)
    return {"local_path": dest_path, "size_bytes": 0, "status": "error"}


# ── manifest ─────────────────────────────────────────────────────────────────

_MANIFEST_COLS = [
    "year",
    "s3_key",
    "local_path",
    "size_bytes",
    "status",
    "downloaded_at",
]


def _load_manifest(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        return pd.read_parquet(path)
    return pd.DataFrame(columns=_MANIFEST_COLS)


def _save_manifest(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_parquet(path, index=False)


def _done_keys(manifest: pd.DataFrame) -> set[str]:
    """Set of S3 keys already completed or confirmed missing."""
    if manifest.empty:
        return set()
    done = manifest[manifest["status"].isin(("done", "missing"))]
    return set(done["s3_key"])


# ── signal handling ──────────────────────────────────────────────────────────

_shutdown_requested = False


def _signal_handler(signum, frame):
    global _shutdown_requested
    _shutdown_requested = True
    print("\n  Shutdown requested — finishing current batch ...")
    sys.stdout.flush()


# ── main ─────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Bulk download NOAA ISD raw archive from S3."
    )
    p.add_argument(
        "--dest",
        default=DEFAULT_DEST,
        help=f"Root destination directory (default: {DEFAULT_DEST}).",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Number of concurrent download threads.",
    )
    p.add_argument(
        "--start-year",
        type=int,
        default=1901,
        help="First year to download (default: 1901).",
    )
    p.add_argument(
        "--end-year",
        type=int,
        default=datetime.now().year,
        help="Last year to download (default: current year).",
    )
    p.add_argument(
        "--schedule",
        default=None,
        help='Download time window, e.g. "00:00-08:00".  Unrestricted if omitted.',
    )
    p.add_argument(
        "--weekend-free",
        action="store_true",
        help="Allow unrestricted downloads on Saturday and Sunday.",
    )
    p.add_argument(
        "--oldest-first",
        action="store_true",
        help="Download oldest years first (default: most recent first).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    s3 = _make_s3_client()

    manifest_path = os.path.join(args.dest, "manifest.parquet")
    manifest = _load_manifest(manifest_path)
    done = _done_keys(manifest)
    new_rows: list[dict] = []

    years = list(range(args.start_year, args.end_year + 1))
    if not args.oldest_first:
        years.reverse()

    total_downloaded = 0
    total_bytes = 0
    t_start = time.time()

    print(f"\n{'=' * 60}")
    print(
        f"  ISD archive download: {args.start_year} → {args.end_year}  ({len(years)} years)"
    )
    print(f"  Dest: {args.dest}")
    print(f"  Workers: {args.workers}  Schedule: {args.schedule or 'unrestricted'}")
    print(f"  Already done: {len(done)} files")
    print(f"{'=' * 60}\n")
    sys.stdout.flush()

    for year_i, year in enumerate(years):
        if _shutdown_requested:
            break

        _wait_for_window(args.schedule, args.weekend_free)
        if _shutdown_requested:
            break

        # Discover files for this year
        print(f"  Listing s3://{S3_BUCKET}/{DATA_PREFIX}{year}/ ...")
        sys.stdout.flush()
        all_keys = _list_year_keys(s3, year)

        # Filter out already-done keys and files that exist locally with correct size
        tasks: list[tuple[str, str]] = []
        for key in all_keys:
            if key in done:
                continue
            lp = _local_path(args.dest, key)
            if os.path.exists(lp) and os.path.getsize(lp) > 0:
                done.add(key)
                continue
            tasks.append((key, lp))

        if not tasks:
            print(f"  {year}: all {len(all_keys)} files already done")
            sys.stdout.flush()
            continue

        print(f"  {year}: {len(tasks)} to download ({len(all_keys)} total on S3)")
        sys.stdout.flush()

        # Download this year's files
        year_done = 0
        year_bytes = 0
        year_t0 = time.time()

        # Each thread gets its own S3 client (boto3 clients aren't thread-safe)
        def _download_task(key_path: tuple[str, str]) -> tuple[str, dict]:
            key, lp = key_path
            thread_s3 = _make_s3_client()
            result = _download_one(thread_s3, key, lp)
            return key, result

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_download_task, (key, lp)): key for key, lp in tasks}
            for fut in as_completed(futures):
                if _shutdown_requested:
                    break
                key, result = fut.result()
                row = {
                    "year": year,
                    "s3_key": key,
                    **result,
                    "downloaded_at": datetime.now().isoformat(),
                }
                new_rows.append(row)
                done.add(key)
                if result["status"] == "done":
                    year_done += 1
                    year_bytes += result["size_bytes"]

        total_downloaded += year_done
        total_bytes += year_bytes
        elapsed = time.time() - year_t0
        rate = year_bytes / elapsed / 1e6 if elapsed > 0 else 0
        total_elapsed = time.time() - t_start

        print(
            f"  {year}  {year_done}/{len(tasks)} files  "
            f"{year_bytes / 1e6:.0f} MB  {rate:.1f} MB/s  "
            f"[total: {total_downloaded} files, {total_bytes / 1e9:.1f} GB, "
            f"{total_elapsed / 3600:.1f}h]"
        )
        sys.stdout.flush()

        # Flush manifest every 10 years
        if (year_i + 1) % 10 == 0 and new_rows:
            manifest = pd.concat([manifest, pd.DataFrame(new_rows)], ignore_index=True)
            new_rows.clear()
            _save_manifest(manifest, manifest_path)

    # Final manifest save
    if new_rows:
        manifest = pd.concat([manifest, pd.DataFrame(new_rows)], ignore_index=True)
    _save_manifest(manifest, manifest_path)

    elapsed_h = (time.time() - t_start) / 3600
    print(
        f"\n  Done: {total_downloaded} files, {total_bytes / 1e9:.1f} GB in {elapsed_h:.1f}h"
    )
    if _shutdown_requested:
        print("  (interrupted — resume by re-running the same command)")


if __name__ == "__main__":
    main()
