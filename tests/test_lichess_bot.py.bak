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
    """A bot with no network, no model and no token: only the fields the loop uses."""
    import threading
    from src.lichess_bot import LichessBot
    bot = LichessBot.__new__(LichessBot)
    bot.busy = threading.Event()
    bot.rating, bot.provisional = 1270, True
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

    bot.challenge_someone = flaky
    bot._refresh_rating = lambda: None
    monkeypatch.setattr(lichess_bot.time, "sleep", lambda _: None)

    with pytest.raises(KeyboardInterrupt):       # not an Exception, so it gets through
        bot._seek_loop(every=0)
    assert len(calls) == 3                       # it kept going after the failure


def test_rating_is_refreshed_before_choosing_an_opponent():
    bot = _bare_bot()
    bot.account = lambda: {"perfs": {"rapid": {"rating": 1461, "prov": True, "games": 17}}}
    bot._refresh_rating()
    assert (bot.rating, bot.provisional) == (1461, True)

    bot.account = lambda: {"perfs": {"rapid": {"rating": 1520, "games": 30}}}
    bot._refresh_rating()
    assert (bot.rating, bot.provisional) == (1520, False)     # no "prov" key: settled


def test_a_failed_refresh_keeps_the_last_known_rating():
    import requests
    bot = _bare_bot()
    bot.rating, bot.provisional = 1461, False

    def down():
        raise requests.ConnectionError("down")

    bot.account = down
    bot._refresh_rating()
    assert (bot.rating, bot.provisional) == (1461, False)
