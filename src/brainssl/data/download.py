"""Fetch the two public datasets used in this repository.

MSD Task01 (brain tumour MRI) is distributed as a single 7.6 GB tar file. The
host serves each connection slowly, so the archive is read as many byte ranges
in parallel and unpacked while it streams in. The tar itself never touches the
disk, which keeps peak usage at the size of the extracted data. An interrupted
run picks up again at the first member that was not fully written.

ABIDE I (resting-state fMRI) comes from the Preprocessed Connectomes Project as
one small text file of ROI time series per subject and atlas, plus the atlas
images that are needed later for ROI coordinates.

    python -m brainssl.data.download msd   --dest data/raw
    python -m brainssl.data.download abide --dest data/raw
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import csv
import hashlib
import http.client
import io
import json
import logging
import os
import shutil
import tarfile
import threading
import time
import urllib.parse
from pathlib import Path, PurePosixPath

log = logging.getLogger(__name__)

MSD_URL = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar"
MSD_SIZE = 7_608_266_240
# Checksum published with MONAI's DecathlonDataset for the same file.
MSD_MD5 = "240a19d752f0d9e9101544901065d872"

ABIDE_ROOT = "https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative"
ABIDE_PHENOTYPES = "Phenotypic_V1_0b_preprocessed1.csv"
ABIDE_ATLASES = ("cc200", "cc400", "aal", "ho", "dosenbach160")

# The derivative folders and the files under Resources/ do not share names.
ATLAS_RESOURCES = {
    "cc200": ("cc200_roi_atlas.nii.gz", "CC200_ROI_labels.csv"),
    "cc400": ("cc400_roi_atlas.nii.gz", "CC400_ROI_labels.csv"),
    "aal": ("aal_roi_atlas.nii.gz", "aal_labels.csv"),
    "ho": ("ho_roi_atlas.nii.gz", "ho_labels.csv"),
    "dosenbach160": ("dos160_roi_atlas.nii.gz", "dos160_labels.csv"),
    "ez": ("ez_roi_atlas.nii.gz", "ez_labels.csv"),
    "tt": ("tt_roi_atlas.nii.gz", "tt_labels.csv"),
}

_USER_AGENT = "brainssl/0.1 (+https://github.com/mahdisabetkish/brainssl)"
_BLOCK = 512


class HTTPStatusError(OSError):
    def __init__(self, url: str, code: int) -> None:
        super().__init__(f"HTTP {code} for {url}")
        self.code = code


_local = threading.local()


def _connection(scheme: str, host: str, timeout: float) -> http.client.HTTPConnection:
    """One keep-alive connection per thread and host.

    The ABIDE files are small and the servers are far away, so a fresh TLS
    handshake per file costs more time than the transfer itself.
    """
    pool = getattr(_local, "pool", None)
    if pool is None:
        pool = _local.pool = {}
    key = (scheme, host)
    if key not in pool:
        cls = http.client.HTTPSConnection if scheme == "https" else http.client.HTTPConnection
        pool[key] = cls(host, timeout=timeout)
    return pool[key]


def _drop_connection(scheme: str, host: str) -> None:
    conn = getattr(_local, "pool", {}).pop((scheme, host), None)
    if conn is not None:
        conn.close()


def _read_body(response: http.client.HTTPResponse, deadline: float | None) -> bytes:
    if deadline is None:
        return response.read()
    out = bytearray()
    while True:
        block = response.read(1 << 16)
        if not block:
            return bytes(out)
        out += block
        if time.monotonic() > deadline:
            # A connection can stay alive while delivering a trickle, which the
            # socket timeout never catches. Give up on it and let the retry start fresh.
            raise TimeoutError("transfer too slow")


def http_get(url: str, start: int | None = None, end: int | None = None,
             timeout: float = 60.0, retries: int = 8,
             max_seconds: float | None = None) -> bytes:
    """GET a URL, optionally a closed byte range, retrying transient failures.

    ``max_seconds`` bounds a single attempt from request to last byte.
    """
    parts = urllib.parse.urlsplit(url)
    path = parts.path + (f"?{parts.query}" if parts.query else "")
    headers = {"User-Agent": _USER_AGENT}
    if start is not None:
        headers["Range"] = f"bytes={start}-{end}"
    delay = 2.0
    for attempt in range(retries):
        try:
            deadline = None if max_seconds is None else time.monotonic() + max_seconds
            conn = _connection(parts.scheme, parts.netloc, timeout)
            conn.request("GET", path, headers=headers)
            response = conn.getresponse()
            data = _read_body(response, deadline)
            if response.status in (403, 404):
                raise HTTPStatusError(url, response.status)
            if response.status not in (200, 206):
                raise OSError(f"HTTP {response.status}")
            if start is not None and len(data) != end - start + 1:
                raise OSError(f"short read, got {len(data)} of {end - start + 1} bytes")
            return data
        except HTTPStatusError:
            raise
        except (http.client.HTTPException, OSError) as err:
            _drop_connection(parts.scheme, parts.netloc)
            if attempt == retries - 1:
                raise
            log.debug("%s for %s, retrying", err, url)
        time.sleep(delay)
        delay = min(2 * delay, 30.0)
    raise RuntimeError("unreachable")


def download_file(url: str, target: Path) -> Path:
    """Small-file download with an atomic rename, skipped if the file exists."""
    if target.exists() and target.stat().st_size > 0:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".part")
    tmp.write_bytes(http_get(url))
    os.replace(tmp, target)
    return target


class _Chunk:
    __slots__ = ("start", "end", "futures", "started")

    def __init__(self, start: int, end: int) -> None:
        self.start, self.end = start, end
        self.futures: list[cf.Future[bytes]] = []
        self.started: float | None = None  # when the first request actually went out


class RangeReader(io.RawIOBase):
    """Read a remote file front to back while keeping several ranges in flight.

    Chunks are requested ahead of the read position by a thread pool and handed
    out strictly in order, so the consumer sees an ordinary sequential stream.
    Ordering has a cost: one slow connection at the head of the queue stalls the
    whole stream while finished chunks wait behind it. When the head chunk is
    late, a second request for the same range is sent and whichever copy lands
    first is used.
    """

    def __init__(self, url: str, size: int, start: int = 0,
                 chunk_size: int = 4 << 20, workers: int = 16,
                 digest: hashlib._Hash | None = None,
                 hedge_after: float = 45.0, max_seconds: float = 180.0) -> None:
        super().__init__()
        self.url = url
        self.size = size
        self.chunk_size = chunk_size
        self.digest = digest
        self.position = start
        self.hedge_after = hedge_after
        self.max_seconds = max_seconds
        self._next = start
        self._depth = 3 * workers
        self._pending: collections.deque[_Chunk] = collections.deque()
        self._buffer = memoryview(b"")
        self._pool = cf.ThreadPoolExecutor(max_workers=workers, thread_name_prefix="range")
        # Hedged requests get their own threads so they do not queue behind
        # the regular read-ahead.
        self._hedge_pool = cf.ThreadPoolExecutor(max_workers=4, thread_name_prefix="hedge")
        self._schedule()

    def _fetch(self, chunk: _Chunk) -> bytes:
        if chunk.started is None:
            chunk.started = time.monotonic()
        return http_get(self.url, chunk.start, chunk.end, max_seconds=self.max_seconds)

    def _submit(self, chunk: _Chunk, hedge: bool = False) -> None:
        pool = self._hedge_pool if hedge else self._pool
        chunk.futures.append(pool.submit(self._fetch, chunk))

    def _schedule(self) -> None:
        while len(self._pending) < self._depth and self._next < self.size:
            end = min(self._next + self.chunk_size, self.size) - 1
            chunk = _Chunk(self._next, end)
            self._submit(chunk)
            self._pending.append(chunk)
            self._next = end + 1

    def _take_head(self) -> bytes:
        head = self._pending[0]
        while True:
            done, _ = cf.wait(head.futures, timeout=5.0, return_when=cf.FIRST_COMPLETED)
            for future in done:
                if future.exception() is None:
                    self._pending.popleft()
                    return future.result()
            if done and all(f.done() for f in head.futures):
                if len(head.futures) >= 4:
                    raise next(iter(done)).exception()  # type: ignore[misc]
                self._submit(head, hedge=True)  # every copy failed; try once more
            elif (head.started is not None and len(head.futures) < 3
                  and time.monotonic() - head.started > self.hedge_after * len(head.futures)):
                log.debug("hedging range %d-%d", head.start, head.end)
                self._submit(head, hedge=True)

    def readable(self) -> bool:
        return True

    def readinto(self, buffer) -> int:  # type: ignore[override]
        if not self._buffer:
            if not self._pending:
                return 0
            chunk = self._take_head()
            if self.digest is not None:
                self.digest.update(chunk)
            self._buffer = memoryview(chunk)
            self._schedule()
        n = min(len(buffer), len(self._buffer))
        buffer[:n] = self._buffer[:n]
        self._buffer = self._buffer[n:]
        self.position += n
        return n

    def close(self) -> None:
        for chunk in self._pending:
            for future in chunk.futures:
                future.cancel()
        self._pool.shutdown(wait=False, cancel_futures=True)
        self._hedge_pool.shutdown(wait=False, cancel_futures=True)
        super().close()


def _member_path(root: Path, name: str) -> Path | None:
    """Destination for a tar member, or None for entries we do not keep."""
    parts = PurePosixPath(name).parts
    if not parts or any(p in ("..", "") for p in parts) or PurePosixPath(name).is_absolute():
        raise ValueError(f"refusing to extract suspicious path {name!r}")
    base = parts[-1]
    if base.startswith("._") or base == ".DS_Store" or "__MACOSX" in parts:
        return None
    return root.joinpath(*parts)


def fetch_msd(dest: Path, workers: int = 16) -> Path:
    """Stream MSD Task01_BrainTumour into ``dest/Task01_BrainTumour``."""
    dest.mkdir(parents=True, exist_ok=True)
    task_dir = dest / "Task01_BrainTumour"
    state_path = dest / ".Task01_BrainTumour.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"offset": 0, "files": 0}
    if state.get("complete"):
        log.info("MSD Task01 already extracted in %s", task_dir)
        return task_dir

    base = state["offset"]
    # The checksum only means something when the whole file passes through us.
    digest = hashlib.md5() if base == 0 else None
    if base:
        log.info("resuming MSD Task01 at byte %d (%.1f%%)", base, 100 * base / MSD_SIZE)

    reader = RangeReader(MSD_URL, MSD_SIZE, start=base, workers=workers, digest=digest)
    stream = io.BufferedReader(reader, buffer_size=1 << 20)
    started, last_report = time.time(), 0.0
    try:
        with tarfile.open(fileobj=stream, mode="r|") as tar:
            for member in tar:
                target = _member_path(dest, member.name)
                if member.isfile() and target is not None:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    tmp = target.with_name(target.name + ".part")
                    source = tar.extractfile(member)
                    assert source is not None
                    with source, open(tmp, "wb") as out:
                        shutil.copyfileobj(source, out, 1 << 20)
                    os.replace(tmp, target)
                    state["files"] += 1
                padded = -(-member.size // _BLOCK) * _BLOCK
                state["offset"] = base + member.offset_data + padded
                state_path.write_text(json.dumps(state))

                now = time.time()
                if now - last_report > 30:
                    done = reader.position - base
                    rate = done / max(now - started, 1e-6) / 2**20
                    eta = (MSD_SIZE - reader.position) / max(rate * 2**20, 1.0) / 60
                    log.info("MSD %5.1f%%  %d files  %.2f MB/s  ~%.0f min left",
                             100 * reader.position / MSD_SIZE, state["files"], rate, eta)
                    last_report = now
            # Tar pads its end with zero records; read them so the checksum covers the file.
            while stream.read(1 << 20):
                pass
    finally:
        stream.close()

    if digest is not None:
        md5 = digest.hexdigest()
        if md5 != MSD_MD5:
            raise RuntimeError(f"MSD Task01 checksum mismatch: {md5} != {MSD_MD5}")
        log.info("MSD Task01 checksum verified")
    else:
        log.info("resumed download, checksum skipped; run the data check to read every file")
    state["complete"] = True
    state_path.write_text(json.dumps(state))
    log.info("MSD Task01: %d files in %s", state["files"], task_dir)
    return task_dir


def fetch_abide(dest: Path, atlases: tuple[str, ...] = ABIDE_ATLASES,
                pipeline: str = "cpac", strategy: str = "filt_noglobal",
                workers: int = 16) -> Path:
    """Download ABIDE I ROI time series, phenotypes and atlas images."""
    root = dest / "abide"
    phenotypes = download_file(f"{ABIDE_ROOT}/{ABIDE_PHENOTYPES}", root / ABIDE_PHENOTYPES)
    with open(phenotypes, newline="", encoding="utf-8") as fh:
        file_ids = [row["FILE_ID"] for row in csv.DictReader(fh) if row["FILE_ID"] != "no_filename"]

    jobs: list[tuple[str, Path]] = []
    for atlas in atlases:
        image, labels = ATLAS_RESOURCES[atlas]
        for name in (image, labels):
            jobs.append((f"{ABIDE_ROOT}/Resources/{name}", root / "atlases" / name))
        folder = f"rois_{atlas}"
        for file_id in file_ids:
            name = f"{file_id}_{folder}.1D"
            url = f"{ABIDE_ROOT}/Outputs/{pipeline}/{strategy}/{folder}/{name}"
            jobs.append((url, root / pipeline / strategy / folder / name))

    todo = [(u, t) for u, t in jobs if not (t.exists() and t.stat().st_size > 0)]
    log.info("ABIDE: %d subjects x %d atlases, %d files to fetch", len(file_ids), len(atlases), len(todo))
    failed: list[str] = []
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(download_file, url, target): url for url, target in todo}
        for i, future in enumerate(cf.as_completed(futures), 1):
            try:
                future.result()
            except HTTPStatusError as err:
                failed.append(f"{futures[future]} ({err.code})")
            if i % 250 == 0 or i == len(todo):
                log.info("ABIDE %d/%d", i, len(todo))
    if failed:
        # A handful of subjects lack a derivative for some atlas; record them rather than abort.
        (root / "missing.txt").write_text("\n".join(sorted(failed)) + "\n")
        log.warning("ABIDE: %d files unavailable, listed in %s", len(failed), root / "missing.txt")
    return root


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="dataset", required=True)
    for name in ("msd", "abide"):
        p = sub.add_parser(name)
        p.add_argument("--dest", type=Path, default=Path("data/raw"))
        p.add_argument("--workers", type=int, default=16)
    sub.choices["abide"].add_argument("--atlases", nargs="+", default=list(ABIDE_ATLASES))
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        datefmt="%H:%M:%S")
    if args.dataset == "msd":
        fetch_msd(args.dest, workers=args.workers)
    else:
        fetch_abide(args.dest, atlases=tuple(args.atlases), workers=args.workers)


if __name__ == "__main__":
    main()
