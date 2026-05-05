# worker/tests/test_dispatch_auth.py
import os
import pytest


def test_mark_done_sends_email_when_last(mocker):
    """_mark_done triggers email exactly when done == total."""
    mock_redis = mocker.MagicMock()
    mock_redis.incr.return_value = 3   # done counter reaches total
    mock_redis.get.side_effect = lambda key: {
        "batch:b1:total": "3",
        "batch:b1:email": "a@b.com",
        "batch:b1:completed": "2",
    }.get(key)
    mock_email = mocker.MagicMock()

    from worker.modal_app import _mark_done
    _mark_done(mock_redis, "b1", success=True, send_email=mock_email)

    mock_email.assert_called_once_with("a@b.com", "b1", succeeded=2, failed=1)


def test_mark_done_does_not_send_when_not_last(mocker):
    """_mark_done does not send email if done < total."""
    mock_redis = mocker.MagicMock()
    mock_redis.incr.return_value = 2   # only 2 of 3 done
    mock_redis.get.side_effect = lambda key: {"batch:b1:total": "3"}.get(key)
    mock_email = mocker.MagicMock()

    from worker.modal_app import _mark_done
    _mark_done(mock_redis, "b1", success=False, send_email=mock_email)

    mock_email.assert_not_called()
