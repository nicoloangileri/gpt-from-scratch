"""Challenge filtering, tested without touching the network or a token."""

import pytest

from src.lichess_bot import MIN_INITIAL_SECONDS, should_accept


def challenge(variant="standard", kind="clock", limit=300, increment=2):
    c = {"id": "abc123", "variant": {"key": variant}}
    c["timeControl"] = {"type": kind} if limit is None else {
        "type": kind, "limit": limit, "increment": increment}
    return c


def test_accepts_a_sensible_rapid_game():
    assert should_accept(challenge(limit=300)) == (True, "")


def test_declines_bullet_because_search_cannot_move_that_fast():
    ok, reason = should_accept(challenge(limit=60))
    assert (ok, reason) == (False, "tooFast")


def test_accepts_exactly_at_the_boundary():
    assert should_accept(challenge(limit=MIN_INITIAL_SECONDS))[0] is True
    assert should_accept(challenge(limit=MIN_INITIAL_SECONDS - 1))[0] is False


@pytest.mark.parametrize("variant", ["chess960", "atomic", "horde", "crazyhouse"])
def test_declines_variants_the_model_never_saw(variant):
    assert should_accept(challenge(variant=variant)) == (False, "variant")


def test_declines_correspondence_and_unlimited():
    assert should_accept(challenge(kind="correspondence", limit=None))[1] == "timeControl"
    assert should_accept(challenge(kind="unlimited", limit=None))[1] == "timeControl"


def test_variant_is_checked_before_the_clock():
    """A fast Chess960 challenge is declined as a variant, not as too fast."""
    assert should_accept(challenge(variant="chess960", limit=30)) == (False, "variant")


def test_token_comes_from_the_environment(monkeypatch):
    from src import lichess_bot
    monkeypatch.setenv("LICHESS_TOKEN", "  abc123  ")
    assert lichess_bot._token() == "abc123"


def test_empty_export_is_treated_as_missing(monkeypatch, tmp_path):
    from src import lichess_bot
    monkeypatch.setenv("LICHESS_TOKEN", "   ")
    monkeypatch.setattr(lichess_bot, "TOKEN_FILE", tmp_path / "absent")
    with pytest.raises(SystemExit) as e:
        lichess_bot._token()
    assert "No token" in str(e.value)


def test_token_file_is_used_and_must_be_private(monkeypatch, tmp_path):
    from src import lichess_bot
    monkeypatch.delenv("LICHESS_TOKEN", raising=False)
    f = tmp_path / "token"
    f.write_text("lip_secret\n")
    monkeypatch.setattr(lichess_bot, "TOKEN_FILE", f)

    f.chmod(0o644)
    with pytest.raises(SystemExit) as e:
        lichess_bot._token()
    assert "readable by others" in str(e.value)

    f.chmod(0o600)
    assert lichess_bot._token() == "lip_secret"


def test_our_own_outgoing_challenge_is_ignored():
    from src.lichess_bot import is_incoming
    mine = {"id": "abc", "challenger": {"id": "imbotterthanyou"}}
    theirs = {"id": "abc", "challenger": {"id": "maia5"}}
    assert not is_incoming(mine, "imbotterthanyou")
    assert is_incoming(theirs, "imbotterthanyou")
    assert is_incoming({"id": "abc"}, "imbotterthanyou")      # no challenger field: treat as incoming

def _bare_bot():
    """A bot with no network, no model and no token: only the fields the loops use."""
    import threading
    from src.lichess_bot import LichessBot
    bot = LichessBot.__new__(LichessBot)
    bot.busy = threading.Event()
    bot.wake = threading.Event()
    bot.me = "imbotterthanyou"
    bot.ratings = {"rapid": (1461, True)}
    bot.cooldown = {}
    return bot


def test_seek_loop_survives_a_dropped_connection(monkeypatch):
    """One RemoteDisconnected used to kill the thread and stop the bot for hours.

    The process stayed up and the log stayed open, so nothing looked wrong: the bot
    simply never challenged anyone again. The loop must swallow anything Lichess
    throws at it and come back on the next cycle.
    """
    import requests
    from src import lichess_bot

    bot = _bare_bot()
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) == 1:
            raise requests.ConnectionError("Remote end closed connection")
        if len(calls) == 3:
            raise KeyboardInterrupt              # ends the otherwise infinite loop

    bot.seek_any_clock = flaky
    bot._refresh_ratings = lambda: None
    monkeypatch.setattr(lichess_bot.time, "sleep", lambda _: None)

    with pytest.raises(KeyboardInterrupt):       # not an Exception, so it gets through
        bot._seek_loop(every=0)
    assert len(calls) == 3                       # it kept going after the failure


def test_ratings_are_refreshed_before_choosing_an_opponent():
    bot = _bare_bot()
    bot.account = lambda: {"perfs": {
        "rapid": {"rating": 1520, "games": 30},           # no "prov" key: settled
        "blitz": {"rating": 1400, "prov": True, "games": 2},
        "puzzle": {"rating": 1600},
    }}
    bot._refresh_ratings()
    assert bot.ratings["rapid"] == (1520, False)
    assert bot.ratings["blitz"] == (1400, True)


def test_a_failed_refresh_keeps_the_last_known_ratings():
    import requests
    bot = _bare_bot()
    bot.ratings = {"rapid": (1461, False)}

    def down():
        raise requests.ConnectionError("down")

    bot.account = down
    bot._refresh_ratings()
    assert bot.ratings == {"rapid": (1461, False)}


@pytest.mark.parametrize("clock,expected", [
    ((600, 3), "rapid"),        # 10+3, what the bot played its first games at
    ((300, 3), "blitz"),        # 5+3  -- 300 + 120 = 420 < 480
    ((180, 2), "blitz"),
    ((60, 0), "bullet"),
    ((1800, 20), "classical"),
])
def test_perf_of_matches_lichess_speed_categories(clock, expected):
    """The rating compared must belong to the clock being offered.

    Comparing a blitz rating while offering a rapid game is how a 1270 provisional
    account challenged an opponent rated 2793 at the speed it was actually offering.
    """
    from src.lichess_bot import perf_of
    assert perf_of(*clock) == expected


def _bots(*pairs, perf="rapid"):
    return [{"id": i, "perfs": {perf: {"rating": r}}} for i, r in pairs]


def test_a_bot_that_refuses_is_not_asked_again_this_cycle():
    """Eleven hours of asking the same few bots produced 88 refusals and no game."""
    from src.lichess_bot import COOLDOWN_S
    bot = _bare_bot()
    asked = []

    class Reply:
        ok, status_code = False, 400

        def __init__(self, text):
            self.text = text

    def post(path, **kw):
        who = path.rsplit("/", 1)[-1]
        asked.append(who)
        # Lichess says why: one has spent its 100 bot games today, the other is simply
        # not accepting. Only the first deserves a cooldown measured in hours.
        return Reply(f'{{"error":"{who} played 100 games today"}}' if who == "pawnrobot"
                     else '{"error":"challenge declined"}')

    bot._post = post
    bot.online_bots = lambda: _bots(("pawnrobot", 1300), ("anti-bot", 1500))
    assert bot.challenge_someone((600, 3)) is False
    assert sorted(asked) == ["anti-bot", "pawnrobot"]

    # Both are now on cooldown, the capped one for much longer than the other.
    assert bot.challenge_someone((600, 3)) is False
    assert len(asked) == 2                            # no second round of refusals
    assert bot.cooldown["pawnrobot"] > bot.cooldown["anti-bot"] + COOLDOWN_S


def test_seek_falls_back_to_a_faster_clock():
    """Nobody takes 10+3 at 3am; more of the pool will take 5+3."""
    bot = _bare_bot()
    bot.ratings = {"rapid": (1461, True), "blitz": (1461, True)}
    offered = []

    class Reply:
        ok, status_code, text = True, 200, "{}"

    def post(path, **kw):
        offered.append(int(kw["data"]["clock.limit"]))
        return Reply()

    # Only blitz-rated bots are online, so the rapid pass finds nobody to ask.
    bot._post = post
    bot.online_bots = lambda: _bots(("worstfish", 1500), perf="blitz")
    assert bot.seek_any_clock() is True
    assert offered == [300]                            # rapid skipped, 5+3 sent


def test_the_seek_loop_stops_waiting_when_a_challenge_is_refused(monkeypatch):
    """A decline arrives seconds after the challenge, not 90 seconds later.

    The POST that sent the challenge succeeded, so the loop thinks it found a game
    and settles in for the full cycle; meanwhile the opponent has already said no.
    """
    bot = _bare_bot()
    waited = []

    def wait(timeout=None):
        waited.append(timeout)
        return True

    bot.wake.wait = wait
    bot.seek_any_clock = lambda: True
    bot._refresh_ratings = lambda: None
    monkeypatch.setattr(bot.wake, "clear", lambda: (_ for _ in ()).throw(KeyboardInterrupt))

    with pytest.raises(KeyboardInterrupt):
        bot._seek_loop(every=90)
    assert waited == [90]          # the wait is interruptible, not a bare time.sleep


def test_no_challenge_goes_out_while_a_game_is_being_played(monkeypatch):
    """play_game blocks the event loop, so a second game would sit idle and flag."""
    from src import lichess_bot
    bot = _bare_bot()
    bot.busy.set()
    bot._refresh_ratings = lambda: None
    bot.seek_any_clock = lambda: pytest.fail("challenged somebody mid-game")
    monkeypatch.setattr(bot.wake, "clear", lambda: (_ for _ in ()).throw(KeyboardInterrupt))
    monkeypatch.setattr(lichess_bot.time, "sleep", lambda _: None)

    with pytest.raises(KeyboardInterrupt):
        bot._seek_loop(every=0)


def test_a_dropped_game_is_reconnected_not_abandoned(monkeypatch):
    """Two rated games were lost on time this way, each with 630+ seconds left.

    Returning on a network error leaves the game running at Lichess with our clock
    ticking. The only correct move is to reconnect to the stream.
    """
    import requests
    from src import lichess_bot

    bot = _bare_bot()
    tries = []

    def stream(game_id):
        tries.append(game_id)
        if len(tries) < 3:
            raise requests.ConnectTimeout("connection timed out")
        # third attempt: the stream comes back and the game plays out

    bot._play_stream = stream
    monkeypatch.setattr(lichess_bot.time, "sleep", lambda _: None)
    bot.play_game("uVc5lAFA")
    assert tries == ["uVc5lAFA"] * 3


def test_reconnecting_gives_up_eventually(monkeypatch):
    """Not forever, though: a game that cannot be reached is lost anyway."""
    import requests
    from src import lichess_bot

    bot = _bare_bot()
    tries = []

    def always_down(game_id):
        tries.append(game_id)
        raise requests.ConnectionError("down")

    bot._play_stream = always_down
    monkeypatch.setattr(lichess_bot.time, "sleep", lambda _: None)
    bot.play_game("uVc5lAFA", attempts=4)
    assert len(tries) == 4
