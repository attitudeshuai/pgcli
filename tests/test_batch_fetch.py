"""Tests for fixed-batch, server-side result fetching.

Covers the streaming contract:

* rows are pulled from the server in fixed batches and rendered as they
  arrive, with memory bounded by the batch size;
* closing the output early (pager exit, write failure, row limit) stops
  fetching immediately and closes the server-side portal;
* a read failure keeps the rows already displayed, leaves the connection
  usable and the server without a hanging cursor;
* each query gets a fresh batch buffer/counter/stop flag, so reruns return
  the complete result;
* two sessions, and the completion refresh connection, never share state.
"""

import re
import threading
from unittest import mock

import psycopg
import pytest

import pgcli.pgexecute as pgexecute_module
from pgcli.main import PGCli, _LazyOutput
from pgcli.pgexecute import BatchedCursor, PGExecute
from utils import dbtest


@pytest.fixture
def populated_executor(executor):
    list(executor.run("create table batch_table(n int, t text)"))
    list(executor.run("insert into batch_table select g, case when g % 3 = 0 then null else 'v' || g end from generate_series(1, 2500) g"))
    return executor


def _portal_names(executor):
    with executor.conn.cursor() as cur:
        cur.execute("select name from pg_cursors")
        return {row[0] for row in cur.fetchall()}


def _render(cli, sql="select * from batch_table order by n"):
    output, _ = cli._evaluate_command(sql)
    return list(output)


def _make_cli(executor, table_format="csv", row_limit=0, **kwargs):
    cli = PGCli(pgexecute=executor, row_limit=row_limit, **kwargs)
    cli.table_format = table_format
    return cli


@dbtest
def test_select_returns_batched_server_cursor(executor):
    result = list(executor.run("select 1 as n"))
    cur = result[0][1]
    assert isinstance(cur, BatchedCursor)
    assert cur.description[0].name == "n"
    assert cur.statusmessage == "SELECT"
    cur.close()
    assert _portal_names(executor) == set()


@dbtest
def test_non_select_statements_keep_client_cursor(executor):
    list(executor.run("create table keep_client(a int)"))
    ddl = list(executor.run("insert into keep_client values (1) returning a"))
    assert not isinstance(ddl[0][1], BatchedCursor)
    assert ddl[0][3] == "INSERT 0 1"


@dbtest
def test_fetchmany_returns_fixed_batches_and_counts(populated_executor, monkeypatch):
    monkeypatch.setattr(pgexecute_module, "FETCH_BATCH_SIZE", 100)
    result = list(populated_executor.run("select * from batch_table order by n"))
    cur = result[0][1]
    # The first batch was fetched eagerly when the cursor was opened.
    assert cur.rowcount == 100
    sizes = []
    while True:
        batch = cur.fetchmany()
        if not batch:
            break
        sizes.append(len(batch))
    assert all(size == 100 for size in sizes)
    assert cur.rowcount == 2500
    cur.close()
    assert _portal_names(populated_executor) == set()


@dbtest
def test_close_is_idempotent_and_releases_portal(populated_executor, monkeypatch):
    monkeypatch.setattr(pgexecute_module, "FETCH_BATCH_SIZE", 100)
    result = list(populated_executor.run("select * from batch_table order by n"))
    cur = result[0][1]
    cur.fetchmany()
    cur.close()
    cur.close()
    assert cur.closed
    assert cur.fetchmany() == []
    assert _portal_names(populated_executor) == set()
    # The connection is immediately usable for the next statement.
    with populated_executor.conn.cursor() as check:
        check.execute("select 42")
        assert check.fetchone() == (42,)


@dbtest
def test_stop_prevents_further_server_roundtrips(populated_executor, monkeypatch):
    monkeypatch.setattr(pgexecute_module, "FETCH_BATCH_SIZE", 100)
    result = list(populated_executor.run("select * from batch_table order by n"))
    cur = result[0][1]
    # Drain the eagerly prefetched first batch (no server round trip).
    assert len(cur.fetchmany()) == 100
    fetch_sizes = []
    original_fetch = psycopg.ServerCursor.fetchmany

    def spy(self, size=0):
        rows = original_fetch(self, size)
        fetch_sizes.append(len(rows))
        return rows

    with mock.patch.object(psycopg.ServerCursor, "fetchmany", spy):
        cur.fetchmany()
        cur.stop()
        cur.close()
        assert cur.fetchmany() == []

    # Exactly one batch was requested from the server before the stop.
    assert fetch_sizes == [100]
    assert _portal_names(populated_executor) == set()


@dbtest
def test_large_result_is_lazy_and_bounded_by_batch_size(populated_executor, monkeypatch):
    monkeypatch.setattr(pgexecute_module, "FETCH_BATCH_SIZE", 100)
    cli = _make_cli(populated_executor)
    output, _ = cli._evaluate_command("select * from batch_table order by n")
    assert isinstance(output, _LazyOutput)

    it = iter(output)
    next(it)  # header
    batched = output.state.cursors[0]
    assert isinstance(batched, BatchedCursor)

    # The eager first batch plus the peek that proves the result is larger
    # than one batch are the only rows in memory at this point.
    assert batched.rowcount == 200
    list(it)
    assert batched.closed
    assert _portal_names(populated_executor) == set()


@dbtest
def test_small_result_renders_through_single_batch_path(populated_executor, monkeypatch):
    from pgcli.main import format_output

    monkeypatch.setattr(pgexecute_module, "FETCH_BATCH_SIZE", 10000)
    with populated_executor.conn.cursor() as check:
        check.execute("select * from batch_table order by n limit 5")
        expected_rows = check.fetchall()

    cli = _make_cli(populated_executor)
    settings = cli._build_output_settings()
    output, _ = cli._evaluate_command("select * from batch_table order by n limit 5")
    rendered = "\n".join(str(line) for line in output)

    # Exactly what the historical buffered path would produce with the same
    # output settings (styles, missing value placeholder, preprocessors).
    expected = "\n".join(format_output(None, expected_rows, ["n", "t"], "SELECT 5", settings))
    assert rendered == expected


@dbtest
@pytest.mark.parametrize("table_format", ["csv", "tsv", "vertical"])
def test_streaming_output_independent_of_batch_size(populated_executor, table_format, monkeypatch):
    monkeypatch.setattr(pgexecute_module, "FETCH_BATCH_SIZE", 10000)
    reference = "\n".join(_render(_make_cli(populated_executor, table_format)))

    for batch_size in (7, 100, 999):
        monkeypatch.setattr(pgexecute_module, "FETCH_BATCH_SIZE", batch_size)
        rendered = "\n".join(_render(_make_cli(populated_executor, table_format)))
        assert rendered == reference


@dbtest
def test_batch_size_keeps_types_nulls_and_order(populated_executor, monkeypatch):
    monkeypatch.setattr(pgexecute_module, "FETCH_BATCH_SIZE", 10000)
    reference = "\n".join(_render(_make_cli(populated_executor, "vertical")))

    monkeypatch.setattr(pgexecute_module, "FETCH_BATCH_SIZE", 37)
    rendered = "\n".join(_render(_make_cli(populated_executor, "vertical")))
    assert rendered == reference
    # Null placeholders and ordering survive the batch boundaries.
    assert rendered.count("<null>") == reference.count("<null>") == 833
    record_numbers = [int(part) for part in re.findall(r"RECORD (\d+)", rendered)]
    assert record_numbers == list(range(1, 2501))


@dbtest
def test_row_limit_matches_message_footer_and_displayed_rows(populated_executor, monkeypatch):
    monkeypatch.setattr(pgexecute_module, "FETCH_BATCH_SIZE", 100)
    cli = _make_cli(populated_executor, row_limit=1000)
    with mock.patch("pgcli.main.click.secho") as secho:
        lines = _render(cli)
    limited_messages = [call.args[0] for call in secho.call_args_list if "limited" in str(call.args)]
    assert limited_messages == ["The result was limited to 1000 rows"]
    # One header, one thousand data rows, one status footer.
    data_lines = [line for line in lines if line and not line.startswith("SELECT")]
    assert len(data_lines) == 1001
    assert lines[-1] == "SELECT 1000"


@dbtest
def test_row_limit_exactly_at_boundary_is_not_truncated(populated_executor, monkeypatch):
    monkeypatch.setattr(pgexecute_module, "FETCH_BATCH_SIZE", 100)
    cli = _make_cli(populated_executor, row_limit=2500)
    with mock.patch("pgcli.main.click.secho") as secho:
        lines = _render(cli)
    assert not any("limited" in str(call.args) for call in secho.call_args_list)
    assert lines[-1] == "SELECT 2500"


@dbtest
def test_row_limit_at_batch_edge_in_streaming_path(populated_executor, monkeypatch):
    # The limit lands exactly on a batch boundary while more rows exist.
    monkeypatch.setattr(pgexecute_module, "FETCH_BATCH_SIZE", 100)
    cli = _make_cli(populated_executor, row_limit=500)
    with mock.patch("pgcli.main.click.secho") as secho:
        lines = _render(cli)
    assert any("limited to 500 rows" in str(call.args) for call in secho.call_args_list)
    data_lines = [line for line in lines if line and not line.startswith("SELECT")]
    assert len(data_lines) == 501  # one header + 500 rows
    assert lines[-1] == "SELECT 500"
    assert _portal_names(populated_executor) == set()


@dbtest
def test_zero_row_select_renders_header_and_status(populated_executor, monkeypatch):
    monkeypatch.setattr(pgexecute_module, "FETCH_BATCH_SIZE", 100)
    cli = _make_cli(populated_executor)
    lines = _render(cli, "select * from batch_table where n < 0")
    assert lines[-1] == "SELECT 0"
    assert _portal_names(populated_executor) == set()


@dbtest
def test_early_stop_closes_cursor_and_rerun_is_complete(populated_executor, monkeypatch):
    monkeypatch.setattr(pgexecute_module, "FETCH_BATCH_SIZE", 100)
    cli = _make_cli(populated_executor)

    output, _ = cli._evaluate_command("select * from batch_table order by n")
    for index, _ in enumerate(iter(output)):
        if index >= 50:
            break
    output.close()
    assert _portal_names(populated_executor) == set()

    rerun, _ = cli._evaluate_command("select * from batch_table order by n")
    lines = list(rerun)
    # Header + 2500 data rows + status footer.
    assert len([line for line in lines if line and not line.startswith("SELECT")]) == 2501
    assert lines[-1] == "SELECT 2500"
    assert _portal_names(populated_executor) == set()


@dbtest
def test_two_full_runs_are_byte_identical(populated_executor, monkeypatch):
    monkeypatch.setattr(pgexecute_module, "FETCH_BATCH_SIZE", 100)
    cli = _make_cli(populated_executor, "vertical")
    first = "\n".join(_render(cli))
    second = "\n".join(_render(cli))
    assert first == second


@dbtest
def test_mid_fetch_error_keeps_rows_and_connection(populated_executor, monkeypatch):
    monkeypatch.setattr(pgexecute_module, "FETCH_BATCH_SIZE", 100)
    cli = _make_cli(populated_executor)
    output, _ = cli._evaluate_command("select * from batch_table order by n")

    original_fetch = psycopg.ServerCursor.fetchmany
    call_count = {"n": 0}

    def failing_fetch(self, size=0):
        call_count["n"] += 1
        if call_count["n"] == 3:
            raise psycopg.OperationalError("simulated read failure")
        return original_fetch(self, size)

    with mock.patch.object(psycopg.ServerCursor, "fetchmany", failing_fetch):
        iterator = iter(output)
        kept_lines = []
        with pytest.raises(psycopg.OperationalError, match="simulated read failure"):
            for line in iterator:
                kept_lines.append(line)

    # Eager first batch (100) plus two successful server fetches (200) were
    # displayed before the failure; the header is one extra line.
    batched = output.state.cursors[0]
    assert call_count["n"] == 3
    assert len(kept_lines) == 301
    assert output.state.failed
    assert batched.closed
    assert _portal_names(populated_executor) == set()
    # Every displayed line is preserved in the replay cache.
    assert output._cache == kept_lines
    # The connection still serves statements.
    with populated_executor.conn.cursor() as check:
        check.execute("select 1")
        assert check.fetchone() == (1,)


@dbtest
def test_two_sessions_have_isolated_cursors(executor, monkeypatch):
    monkeypatch.setattr(pgexecute_module, "FETCH_BATCH_SIZE", 100)
    list(executor.run("create table iso_one(n int)"))
    list(executor.run("insert into iso_one select g from generate_series(1, 2000) g"))

    second = PGExecute(
        database="_test_db",
        user=executor.user,
        host=executor.host,
        password=executor.password,
        port=executor.port,
        dsn=None,
    )
    list(second.run("create table iso_two(n int)"))
    list(second.run("insert into iso_two select g from generate_series(1, 1500) g"))
    try:
        cli_one = _make_cli(executor)
        cli_two = _make_cli(second)
        out_one, _ = cli_one._evaluate_command("select * from iso_one order by n")
        out_two, _ = cli_two._evaluate_command("select * from iso_two order by n")

        iter_one, iter_two = iter(out_one), iter(out_two)
        next(iter_one)
        next(iter_two)
        cursor_one, cursor_two = out_one.state.cursors[0], out_two.state.cursors[0]
        assert cursor_one is not cursor_two
        assert cursor_one.connection is not cursor_two.connection

        # Interrupt the first session; the second must be untouched.
        for _ in range(150):
            next(iter_one)
        out_one.close()
        assert cursor_one.closed
        assert not cursor_two.closed

        remaining = list(iter_two)
        assert cursor_two.closed
        assert remaining[-1] == "SELECT 1500"
        assert _portal_names(executor) == set()
        assert _portal_names(second) == set()
    finally:
        second.conn.close()


@dbtest
def test_completion_refresh_uses_independent_connection(populated_executor, monkeypatch):
    monkeypatch.setattr(pgexecute_module, "FETCH_BATCH_SIZE", 200)
    cli = _make_cli(populated_executor)
    main_pid = populated_executor.conn.info.backend_pid

    refresh_pids = set()
    original_copy = PGExecute.copy

    def spying_copy(self):
        clone = original_copy(self)
        refresh_pids.add(clone.conn.info.backend_pid)
        return clone

    output, _ = cli._evaluate_command("select * from batch_table order by n")
    iterator = iter(output)
    next(iterator)
    try:
        with mock.patch.object(PGExecute, "copy", spying_copy):
            thread = threading.Thread(target=cli.refresh_completions)
            thread.start()
            thread.join(timeout=30)
        assert not thread.is_alive()

        lines = list(iterator)
        # The fetch was not truncated by the refresh.
        assert lines[-1] == "SELECT 2500"
        assert refresh_pids and main_pid not in refresh_pids
        assert _portal_names(populated_executor) == set()
    finally:
        output.close()


@dbtest
def test_server_cursor_failure_falls_back_to_client_cursor(executor, monkeypatch):
    list(executor.run("create table fallback_table(a int)"))
    list(executor.run("insert into fallback_table values (1), (2)"))

    original_cursor = psycopg.Connection.cursor

    def failing_named_cursor(self, name="", **kwargs):
        if name:
            raise psycopg.errors.FeatureNotSupported("portals not supported")
        return original_cursor(self, **kwargs)

    monkeypatch.setattr(psycopg.Connection, "cursor", failing_named_cursor)
    results = list(executor.run("select * from fallback_table order by a"))
    cur = results[0][1]
    assert not isinstance(cur, BatchedCursor)
    assert list(cur) == [(1,), (2,)]


@dbtest
def test_no_open_transaction_or_cursor_left_after_full_run(populated_executor, monkeypatch):
    monkeypatch.setattr(pgexecute_module, "FETCH_BATCH_SIZE", 100)
    cli = _make_cli(populated_executor)
    _render(cli)
    assert populated_executor.conn.info.transaction_status == psycopg.pq.TransactionStatus.IDLE
    assert _portal_names(populated_executor) == set()


@dbtest
def test_tuples_only_output_is_rows_only(populated_executor, monkeypatch):
    monkeypatch.setattr(pgexecute_module, "FETCH_BATCH_SIZE", 100)
    cli = _make_cli(populated_executor, tuples_only=True)
    lines = _render(cli)
    assert not any(line.startswith("SELECT") for line in lines)
    assert not any("|" in line for line in lines)
    assert len(lines) == 2500
