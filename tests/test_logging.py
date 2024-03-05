"""The server log keeps real warnings and drops the per-frame MFCC notice."""

from __future__ import annotations

import logging

from config.settings import _not_nfft_truncation


def record(message: str, *args) -> logging.LogRecord:
    return logging.LogRecord("root", logging.WARNING, __file__, 1, message, args, None)


def test_the_nfft_truncation_notice_is_dropped():
    notice = record(
        "frame length (%d) is greater than FFT size (%d), frame will be truncated. "
        "Increase NFFT to avoid.",
        551,
        512,
    )
    assert _not_nfft_truncation(notice) is False


def test_other_warnings_still_reach_the_log():
    assert _not_nfft_truncation(record("model warm-up skipped: %s", "no model")) is True
