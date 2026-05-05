import os
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail


def send_completion_email(
    to_email: str,
    batch_id: str,
    succeeded: int,
    failed: int,
) -> None:
    total = succeeded + failed
    body = (
        f"{total} video(s) processed. Results are in Google Drive:\n"
        f"KR→JP Translations/{batch_id}/\n\n"
        f"✓ Succeeded: {succeeded}\n"
        f"✗ Failed: {failed}\n"
    )
    message = Mail(
        from_email=os.environ["SENDGRID_FROM_EMAIL"],
        to_emails=to_email,
        subject="[Video Translator] Your batch is ready",
        plain_text_content=body,
    )
    SendGridAPIClient(os.environ["SENDGRID_API_KEY"]).send(message)
