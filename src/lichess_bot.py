"""Play on Lichess with the trained model.

    export LICHESS_TOKEN="..."          # never committed, never passed as an argument
    python -m src.lichess_bot --check   # verify the token and the account
    python -m src.lichess_bot --ckpt checkpoints/chess_best.pt --depth 2

Listens on the event stream, accepts challenges that fit the time control, and
plays each game with alpha-beta on top of the network. The token is read from the
environment only: it is never logged, never written to disk, and never included in
an error message.

Search depth adapts to the clock. Depth 2 costs 1.5-3s per move on CPU, which is
fine at 3+2 and suicide in bullet, so the bot declines what it cannot play and
drops to the bare network when short on time.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import threading
import time
from pathlib import Path

import chess
import requests

from .puzzles_eval import best_move, load_model
from .search import Searcher

API = "https://lichess.org"
MIN_INITIAL_SECONDS = 180          # below this, depth-2 search loses on time
SEEK_CLOCK = (600, 3)              # 10+3: comfortable for a 1.5-3s per move search
SEEK_PERF = "rapid"                # ...which Lichess rates as rapid, so compare that


def is_incoming(challenge: dict, me: str) -> bool:
    """True if someone else challenged us.

    The event stream reports our OWN outgoing challenges as `challenge` events too.
    Trying to accept one gets refused by Lichess and no game ever starts -- which is
    exactly how three games were "accepted" while the bot sat idle.
    """
    return (challenge.get("challenger") or {}).get("id") != me


def should_accept(challenge: dict) -> tuple[bool, str]:
    """Accept standard games slow enough for the search. Returns (ok, decline reason).

    Lichess wants one of its own decline reasons, so the strings are its vocabulary.
    """
    if challenge.get("variant", {}).get("key", "standard") != "standard":
        return False, "variant"
    clock = challenge.get("timeControl", {})
    if clock.get("type") != "clock" or clock.get("limit") is None:
        return False, "timeControl"                 # correspondence or unlimited
    if clock["limit"] < MIN_INITIAL_SECONDS:
        return False, "tooFast"
    return True, ""


TOKEN_FILE = Path.home() / ".config" / "lichess" / "token"


def _token() -> str:
    """The token, from the environment or from a file -- never from an argument.

    Exporting it in the right shell is easy to get wrong (an empty read leaves the
    variable set to nothing and the bot exits), so a file is the durable option:
    written once, readable only by its owner, never in the repo or in shell history.
    """
    token = (os.environ.get("LICHESS_TOKEN") or "").strip()
    if not token and TOKEN_FILE.exists():
        mode = TOKEN_FILE.stat().st_mode & 0o077
        if mode:
            sys.exit(f"{TOKEN_FILE} is readable by others: chmod 600 it first")
        token = TOKEN_FILE.read_text().strip()
        if token:
            print(f"[bot] token read from {TOKEN_FILE}", flush=True)
    if not token:
        sys.exit(f"No token. Either export LICHESS_TOKEN, or put it in {TOKEN_FILE} "
                 f"(chmod 600). Note that an empty export counts as missing.")
    return token


class LichessBot:
    def __init__(self, ckpt: str, device: str = "cpu", depth: int = 2) -> None:
        self.session = requests.Session()
        self.session.headers["Authorization"] = f"Bearer {_token()}"
        self.model, self.centres, _ = load_model(ckpt, device)
        self.device, self.depth = device, depth
        self.busy = threading.Event()          # set while a game is being played

    # --- helpers ---------------------------------------------------------
    def _get(self, path: str, **kw):
        return self.session.get(f"{API}{path}", **kw)

    def _post(self, path: str, **kw):
        return self.session.post(f"{API}{path}", **kw)

    def account(self) -> dict:
        r = self._get("/api/account", timeout=10)
        r.raise_for_status()
        return r.json()

    def pick(self, board: chess.Board, time_left_ms: int | None) -> str:
        """Depth 2 normally, the bare network when the clock gets short."""
        depth = self.depth
        if time_left_ms is not None and time_left_ms < 30_000:
            depth = 1
        if depth <= 1:
            return best_move(self.model, self.centres, board, self.device)
        return Searcher(self.model, self.centres, self.device).best_move(board, depth)

    # --- challenges ------------------------------------------------------
    def handle_challenge(self, challenge: dict) -> None:
        cid = challenge["id"]
        if not is_incoming(challenge, self.me):
            return                                   # our own challenge coming back
        clock = challenge.get("timeControl", {})
        initial = clock.get("limit")
        ok, reason = should_accept(challenge)

        # The reply is checked: printing "accepted" on a refusal hides the failure and
        # leaves the bot waiting for a game that will never start.
        if ok:
            r = self._post(f"/api/challenge/{cid}/accept", timeout=10)
            print(f"[bot] accepted {cid} ({initial}s + {clock.get('increment', 0)}s)"
                  if r.ok else f"[bot] accept of {cid} refused: {r.status_code} {r.text[:90]}",
                  flush=True)
        else:
            r = self._post(f"/api/challenge/{cid}/decline", data={"reason": reason}, timeout=10)
            print(f"[bot] declined {cid}: {reason}"
                  if r.ok else f"[bot] decline of {cid} failed: {r.status_code}", flush=True)

    # --- seeking opponents ------------------------------------------------
    def online_bots(self, limit: int = 50) -> list[dict]:
        """Bots currently online, as ndjson."""
        import json

        r = self._get("/api/bot/online", params={"nb": limit}, stream=True, timeout=20)
        bots = []
        for line in r.iter_lines():
            if line:
                bots.append(json.loads(line))
        return bots

    def challenge_someone(self, rating_window: int = 400, ceiling: int = 1800) -> None:
        """Challenge a random online bot of comparable strength.

        Waiting to be challenged can mean waiting a long time, and a rating needs
        games. Opponents far stronger are skipped: losing 50 games to a 2700 bot
        says nothing that one game would not.

        Two things this has to get right, both learned by getting them wrong:
        the rating compared must be the one for the time control actually being
        played (10+3 is rapid, not blitz -- a bot can be 1700 blitz and 2793
        rapid), and a provisional rating cannot be trusted as the centre of the
        window, so `ceiling` caps opponents outright until ours settles.
        """
        try:
            bots = self.online_bots()
        except requests.RequestException as exc:
            print(f"[bot] could not list bots: {type(exc).__name__}", flush=True)
            return

        mine, provisional = self.rating or 1500, self.provisional
        candidates = []
        for bot in bots:
            if bot.get("id") == self.me:
                continue
            rating = bot.get("perfs", {}).get(SEEK_PERF, {}).get("rating")
            if not rating:
                continue
            if provisional:
                # A provisional rating is not a measurement and must not steer the
                # search. One loss to a 2793 opponent put this account at 2513 after
                # a single game; centring a window on that finds no opponents at all
                # under the cap, and the bot stops playing exactly when it most
                # needs games. While provisional: anyone at or below the ceiling.
                if rating <= ceiling:
                    candidates.append((bot["id"], rating))
            elif abs(rating - mine) <= rating_window:
                candidates.append((bot["id"], rating))
        if not candidates:
            print(f"[bot] no opponent online: "
                  + (f"none at or below {ceiling} {SEEK_PERF}" if provisional
                     else f"none within {rating_window} of {mine} {SEEK_PERF}"), flush=True)
            return

        # Lichess caps bot-vs-bot games at 100 per bot per day, and by evening most of
        # the pool has hit it. One challenge per cycle mostly buys a refusal, so try
        # candidates until one is actually sent.
        random.shuffle(candidates)
        refused = 0
        for target, rating in candidates[:8]:
            r = self._post(f"/api/challenge/{target}", timeout=10, data={
                "rated": "true", "clock.limit": SEEK_CLOCK[0], "clock.increment": SEEK_CLOCK[1],
                "color": "random", "variant": "standard",
            })
            if r.ok:
                print(f"[bot] challenged {target} ({SEEK_PERF} {rating}, mine {mine}"
                      f"{' provisional' if provisional else ''}): sent"
                      f"{f' (after {refused} unavailable)' if refused else ''}", flush=True)
                return
            refused += 1
        print(f"[bot] no opponent took a challenge ({refused} unavailable, "
              f"{len(candidates)} in range)", flush=True)

    def _refresh_rating(self) -> None:
        """Re-read our own rating before picking opponents.

        It is read once at startup and then moves fast while provisional. Thirteen
        hours into a run the log still said "mine 1270 provisional" while the account
        was at 1461: the window was being centred on a number two hours out of date.
        """
        try:
            perf = self.account().get("perfs", {}).get(SEEK_PERF, {})
        except requests.RequestException:
            return                                   # keep the previous value
        if perf.get("rating"):
            self.rating = perf["rating"]
            self.provisional = bool(perf.get("prov", False))

    def _seek_loop(self, every: int = 90) -> None:
        # Nothing restarts this thread, so it must never raise. One RemoteDisconnected
        # from Lichess killed it mid-run and the bot then sat for four hours challenging
        # no one -- with the process still up and the log still open, which looks exactly
        # like a bot that simply cannot find opponents.
        while True:
            try:
                if not self.busy.is_set():
                    self._refresh_rating()
                    self.challenge_someone()
            except Exception as exc:                 # noqa: BLE001 - staying alive wins
                print(f"[bot] seek failed: {type(exc).__name__}: {exc}", flush=True)
            time.sleep(every)

    # --- playing ---------------------------------------------------------
    def play_game(self, game_id: str) -> None:
        import json

        self.busy.set()
        print(f"[bot] game {game_id} started", flush=True)
        stream = self._get(f"/api/bot/game/stream/{game_id}", stream=True, timeout=None)
        my_colour = None
        for line in stream.iter_lines():
            if not line:
                continue                                  # keep-alive newline
            event = json.loads(line)

            if event["type"] == "gameFull":
                my_colour = chess.WHITE if event["white"].get("id") == self.me else chess.BLACK
                state = event["state"]
            elif event["type"] == "gameState":
                state = event
            else:
                continue

            if state.get("status") != "started":
                print(f"[bot] game {game_id} finished: {state.get('status')}", flush=True)
                self.busy.clear()
                return

            board = chess.Board()
            for uci in state["moves"].split():
                board.push(chess.Move.from_uci(uci))
            if board.turn != my_colour or board.is_game_over():
                continue

            clock_ms = state.get("wtime" if my_colour == chess.WHITE else "btime")
            t = time.time()
            move = self.pick(board, clock_ms)
            reply = self._post(f"/api/bot/game/{game_id}/move/{move}", timeout=10)
            print(f"[bot] {game_id} move {board.fullmove_number}: {move} "
                  f"({time.time()-t:.1f}s, clock {clock_ms/1000:.0f}s)"
                  f"{'' if reply.ok else ' REJECTED: ' + reply.text[:80]}", flush=True)

    # --- main loop -------------------------------------------------------
    def run(self, seek: bool = False) -> None:
        import json

        account = self.account()
        self.me = account["id"]
        perf = account.get("perfs", {}).get(SEEK_PERF, {})
        self.rating = perf.get("rating")
        self.provisional = bool(perf.get("prov", True)) or not perf.get("games")
        title = account.get("title", "")
        print(f"[bot] logged in as {account['username']} (title: {title or 'none'})", flush=True)
        if title != "BOT":
            sys.exit("This account is not a BOT account. Upgrade it first:\n"
                     "  curl -d '' https://lichess.org/api/bot/account/upgrade "
                     '-H "Authorization: Bearer $LICHESS_TOKEN"')

        print(f"[bot] waiting for challenges at {API}/@/{account['username']}", flush=True)
        if seek:
            print(f"[bot] also challenging online bots every 90s "
                  f"({SEEK_CLOCK[0]//60}+{SEEK_CLOCK[1]}, rated)", flush=True)
            threading.Thread(target=self._seek_loop, daemon=True).start()
        # The event stream is a long-lived HTTP connection and Lichess (or the local
        # network, or a sleeping laptop) will drop it eventually. Exiting on that would
        # end an overnight run at the first hiccup, so it reconnects instead.
        while True:
            try:
                stream = self._get("/api/stream/event", stream=True, timeout=None)
                for line in stream.iter_lines():
                    if not line:
                        continue
                    event = json.loads(line)
                    kind = event["type"]
                    if kind == "challenge":
                        self.handle_challenge(event["challenge"])
                    elif kind == "gameStart":
                        try:
                            self.play_game(event["game"]["id"])
                        except requests.RequestException as exc:
                            print(f"[bot] game {event['game']['id']} dropped: "
                                  f"{type(exc).__name__}", flush=True)
                        finally:
                            self.busy.clear()        # or the bot never seeks again
                    elif kind in ("challengeDeclined", "challengeCanceled"):
                        # Without this the log shows a challenge going out and then
                        # nothing, which looks identical to the bot being broken.
                        c = event.get("challenge", {})
                        who = (c.get("destUser") or {}).get("id", "?")
                        why = (c.get("declineReason")
                               or kind.replace("challenge", "").lower())
                        print(f"[bot] {who} did not play: {why}", flush=True)
                print("[bot] event stream closed by Lichess; reconnecting", flush=True)
            except requests.RequestException as exc:
                print(f"[bot] event stream dropped ({type(exc).__name__}); "
                      f"reconnecting in 5s", flush=True)
            time.sleep(5)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/chess_best.pt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--seek", action="store_true",
                    help="also challenge online bots, instead of only waiting")
    ap.add_argument("--check", action="store_true",
                    help="verify the token and account, then exit without playing")
    args = ap.parse_args()

    if args.check:                                   # no checkpoint needed for this
        session = requests.Session()
        session.headers["Authorization"] = f"Bearer {_token()}"
        r = session.get(f"{API}/api/account", timeout=10)
        if not r.ok:
            sys.exit(f"Lichess rejected the token: HTTP {r.status_code} {r.text[:120]}")
        me = r.json()
        print(f"username : {me['username']}")
        print(f"title    : {me.get('title') or 'none (not a BOT account yet)'}")
        print(f"games     : {sum(v.get('games', 0) for v in me.get('perfs', {}).values())}")
        scopes = r.headers.get("X-OAuth-Scopes", "unknown")
        print(f"scopes   : {scopes}")
        return

    LichessBot(args.ckpt, args.device, args.depth).run(seek=args.seek)


if __name__ == "__main__":
    main()
