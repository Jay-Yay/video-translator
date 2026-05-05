import os
import pytest


def test_send_completion_email_calls_sendgrid(mocker):
    mocker.patch.dict(os.environ, {
        "SENDGRID_API_KEY": "SG.test",
        "SENDGRID_FROM_EMAIL": "translator@example.com",
    })
    mock_sg_class = mocker.patch("worker.notify.SendGridAPIClient")
    mock_sg = mock_sg_class.return_value

    from worker.notify import send_completion_email
    send_completion_email("user@example.com", "batch-001", succeeded=10, failed=2)

    mock_sg_class.assert_called_once_with("SG.test")
    mock_sg.send.assert_called_once()
    mail = mock_sg.send.call_args[0][0]
    assert mail.personalizations[0].tos[0]['email'] == "user@example.com"
    assert mail.from_email.email == "translator@example.com"
    # Get the actual content value from the Mail object
    msg_dict = mail.get()
    assert "10" in msg_dict['content'][0]['value']
    assert "2" in msg_dict['content'][0]['value']
    assert "batch-001" in msg_dict['content'][0]['value']


def test_send_completion_email_subject(mocker):
    mocker.patch.dict(os.environ, {
        "SENDGRID_API_KEY": "SG.test",
        "SENDGRID_FROM_EMAIL": "translator@example.com",
    })
    mock_sg_class = mocker.patch("worker.notify.SendGridAPIClient")

    from worker.notify import send_completion_email
    send_completion_email("user@example.com", "batch-001", succeeded=5, failed=0)

    mail = mock_sg_class.return_value.send.call_args[0][0]
    assert "[Video Translator]" in mail.subject.subject
