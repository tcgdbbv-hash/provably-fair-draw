"""
draw.py - provably-fair selection of k distinct items from a list.

Two-phase commit-reveal scheme: the operator commits to a secret seed before
the draw, then reveals it after combining with an external (public) entropy
source.  Anyone can re-run the selection algorithm and confirm the result.

Originally written for wallet raffles, but the input is just lines of text -
use it to pick winners from any list (raffle wallets, contest entrants,
giveaway tickets, randomised speaker order, etc.).

WHY COMMIT-REVEAL

A pure RNG draw run by the operator is unverifiable - participants have to
trust that the operator didn't reroll until they liked the result.  The
fix is a two-step scheme:

  Phase 1 (BEFORE the draw window closes):
    Operator generates a 32-byte server_seed and publishes
    sha256(server_seed).  At this point the seed is committed - any later
    attempt to swap it will produce a different hash and the swap is
    detectable.

  Phase 2 (AFTER the draw window closes):
    Operator picks a public entropy source that was *not knowable* during
    phase 1 - e.g. a Solana block hash from a slot scheduled after the
    commitment, a future Bitcoin block hash, a drand round at a future
    epoch.  Combining the committed server_seed with the post-commit
    public_seed means neither party alone can predict the result.

  Verification:
    Operator publishes server_seed, public_seed, the wallet list used,
    and the winners.  Anyone reruns this script in `verify` mode and
    re-derives the same winners.  They also confirm
    sha256(server_seed) == the originally announced commitment.

USAGE

  Phase 1 - commit a seed (writes to .draw-state/server_seed.txt):
    python draw.py commit

  Phase 2 - execute the draw:
    python draw.py draw \\
      --wallets wallets.txt \\
      --public-seed <hex string from external entropy>

  Verification (anyone can run, with the revealed server_seed):
    python draw.py verify \\
      --server-seed <revealed hex> \\
      --public-seed <hex> \\
      --wallets wallets.txt

ALGORITHM

  1. Canonicalise the wallet list - sort + dedupe so input order can't
     bias the outcome and the same set always produces the same draw.
  2. Build a CSPRNG stream: HMAC-SHA256(server_seed, counter || public_seed)
     where counter is an 8-byte big-endian integer that increments per draw.
  3. For each of the k picks, sample a uniform integer in [0, |remaining|)
     via rejection sampling on the HMAC stream (rejects values >= the
     largest multiple of `remaining` that fits in the sample width, so
     there's zero modulo bias), then remove that wallet from the pool.
  4. Result is a list of k distinct wallets in pick order.

NO EXTERNAL DEPENDENCIES

  hashlib, hmac, secrets are all stdlib.  This is intentional so the
  script can be audited by anyone with a fresh Python install and so
  the verification path stays trivially reproducible.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import secrets
import sys
from pathlib import Path
from typing import List

# State file lives alongside wherever the operator runs the script from.
# Kept out of the repo via .gitignore - the seed is secret until reveal.
STATE_DIR = Path(".draw-state")
SEED_FILE = STATE_DIR / "server_seed.txt"


def _read_wallet_list(path: str) -> List[str]:
    """Load a wallet list from disk.

    Format: one wallet per line.  Blank lines and lines starting with '#'
    are ignored, leading/trailing whitespace is stripped.  No format
    validation - if the input has bad addresses the draw still works
    deterministically against whatever strings are there.
    """
    raw = Path(path).read_text().splitlines()
    return [
        line.strip()
        for line in raw
        if line.strip() and not line.strip().startswith("#")
    ]


def deterministic_select(
    server_seed_hex: str,
    public_seed: str,
    wallets: List[str],
    k: int,
) -> List[str]:
    """Pick k distinct wallets deterministically from `server_seed + public_seed`.

    The pool is canonicalised (sorted + deduplicated) before sampling so
    the result depends only on the SET of wallets, not the order they
    arrived in.  Sampling is rejection-based on an HMAC-SHA256 stream
    keyed by the server seed, with the per-draw counter and the public
    seed in the message - no modulo bias, deterministic given the same
    three inputs (server_seed, public_seed, wallet set).
    """
    if k <= 0:
        raise ValueError("k must be > 0")
    deck = sorted(set(wallets))
    n = len(deck)
    if k > n:
        raise ValueError(f"Cannot select {k} from {n} distinct wallets")

    key = bytes.fromhex(server_seed_hex)
    public_bytes = public_seed.encode("utf-8")
    counter = 0

    def next_int(upper: int) -> int:
        nonlocal counter
        # Width: smallest number of bytes that can represent `upper`.
        width = max(1, (upper.bit_length() + 7) // 8)
        max_unbiased = (256**width) // upper * upper
        while True:
            counter += 1
            msg = counter.to_bytes(8, "big") + public_bytes
            digest = hmac.new(key, msg, hashlib.sha256).digest()
            value = int.from_bytes(digest[:width], "big")
            if value < max_unbiased:
                return value % upper

    remaining = list(range(n))
    picks: List[str] = []
    for _ in range(k):
        idx = next_int(len(remaining))
        picks.append(deck[remaining.pop(idx)])
    return picks


def _print_block(title: str, rows: List[tuple[str, str]]) -> None:
    print("=" * 72)
    print(title)
    print("=" * 72)
    width = max(len(label) for label, _ in rows)
    for label, value in rows:
        print(f"  {label.ljust(width)}  {value}")
    print()


def cmd_commit() -> int:
    """Generate a fresh server_seed and print its commitment hash."""
    if SEED_FILE.exists():
        print(
            f"refusing to overwrite existing seed at {SEED_FILE}",
            file=sys.stderr,
        )
        print(
            "move or delete it manually before generating a new commitment.",
            file=sys.stderr,
        )
        return 1
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    server_seed = secrets.token_hex(32)  # 256 bits
    SEED_FILE.write_text(server_seed + "\n")
    try:
        SEED_FILE.chmod(0o600)
    except PermissionError:
        # Best-effort on systems that don't honour chmod (e.g. some
        # network mounts).  Seed will still be readable only by the
        # filesystem's default ACL.
        pass

    commitment = hashlib.sha256(server_seed.encode("utf-8")).hexdigest()
    _print_block(
        "RAFFLE COMMITMENT (PUBLISH THE COMMITMENT HASH BEFORE THE DRAW)",
        [
            ("seed_file", str(SEED_FILE)),
            ("server_seed (KEEP SECRET)", server_seed),
            ("commitment_hash (publish)", commitment),
            ("hash function", "sha256(server_seed_hex_utf8)"),
        ],
    )
    print("publish only the commitment_hash now.")
    print(
        "reveal server_seed AFTER the draw window closes, alongside the "
        "public_seed and winners."
    )
    return 0


def cmd_draw(wallets_path: str, public_seed: str, k: int) -> int:
    """Execute the draw using the committed server_seed + the public_seed."""
    if not SEED_FILE.exists():
        print(
            f"no commitment found at {SEED_FILE} - run `commit` first.",
            file=sys.stderr,
        )
        return 1

    server_seed = SEED_FILE.read_text().strip()
    commitment = hashlib.sha256(server_seed.encode("utf-8")).hexdigest()

    wallets = _read_wallet_list(wallets_path)
    if len(set(wallets)) < k:
        print(
            f"wallet list has {len(set(wallets))} distinct entries, "
            f"need at least {k}.",
            file=sys.stderr,
        )
        return 1

    winners = deterministic_select(server_seed, public_seed, wallets, k)

    _print_block(
        "PROVABLY-FAIR DRAW RESULT",
        [
            ("commitment_hash", commitment),
            ("server_seed (revealed)", server_seed),
            ("public_seed", public_seed),
            ("wallets_input", wallets_path),
            ("wallets_total", str(len(wallets))),
            ("wallets_distinct", str(len(set(wallets)))),
            ("k (winners drawn)", str(k)),
        ],
    )
    print("WINNERS (in pick order):")
    for i, w in enumerate(winners, 1):
        print(f"  {i}. {w}")
    print()
    print("VERIFICATION")
    print("  step 1 - commitment integrity:")
    print(f"    sha256({server_seed}) == {commitment}  # should match")
    print("  step 2 - reproduce the draw:")
    print(
        "    python draw.py verify \\\n"
        f"      --server-seed {server_seed} \\\n"
        f"      --public-seed {public_seed} \\\n"
        f"      --wallets {wallets_path} \\\n"
        f"      --k {k}"
    )
    return 0


def cmd_verify(
    server_seed: str, public_seed: str, wallets_path: str, k: int
) -> int:
    """Recompute the draw to confirm a published result."""
    commitment = hashlib.sha256(server_seed.encode("utf-8")).hexdigest()
    wallets = _read_wallet_list(wallets_path)
    winners = deterministic_select(server_seed, public_seed, wallets, k)

    _print_block(
        "VERIFICATION RESULT",
        [
            ("commitment_hash", commitment),
            ("(compare against the operator's announced commitment)", ""),
            ("server_seed (input)", server_seed),
            ("public_seed (input)", public_seed),
            ("wallets_input", wallets_path),
            ("wallets_distinct", str(len(set(wallets)))),
            ("k (winners drawn)", str(k)),
        ],
    )
    print("REPRODUCED WINNERS (in pick order):")
    for i, w in enumerate(winners, 1):
        print(f"  {i}. {w}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="draw",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser(
        "commit",
        help="generate server_seed and print commitment hash",
    )

    p_draw = sub.add_parser(
        "draw", help="execute the draw using the committed seed"
    )
    p_draw.add_argument(
        "--wallets",
        required=True,
        help="path to wallet list, one address per line",
    )
    p_draw.add_argument(
        "--public-seed",
        required=True,
        help=(
            "external public entropy chosen AFTER the commit was published "
            "(e.g. a Solana block hash, drand round, Bitcoin block hash)"
        ),
    )
    p_draw.add_argument(
        "--k",
        type=int,
        default=3,
        help="number of winners to draw (default: 3)",
    )

    p_verify = sub.add_parser(
        "verify", help="re-run the draw to verify a result"
    )
    p_verify.add_argument("--server-seed", required=True)
    p_verify.add_argument("--public-seed", required=True)
    p_verify.add_argument("--wallets", required=True)
    p_verify.add_argument("--k", type=int, default=3)

    args = parser.parse_args()

    if args.cmd == "commit":
        return cmd_commit()
    if args.cmd == "draw":
        return cmd_draw(args.wallets, args.public_seed, args.k)
    if args.cmd == "verify":
        return cmd_verify(
            args.server_seed, args.public_seed, args.wallets, args.k
        )
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
