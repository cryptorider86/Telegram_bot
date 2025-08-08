#!/usr/bin/env python3
import argparse
import os
import smtplib
from email.message import EmailMessage
from pathlib import Path


def send_email(
    smtp_host: str,
    smtp_port: int,
    smtp_user: str,
    smtp_pass: str,
    sender: str,
    recipient: str,
    subject: str,
    body: str,
    filepath: str,
    use_ssl: bool = False,
):
    msg = EmailMessage()
    msg["From"] = sender
    msg["To"] = recipient
    msg["Subject"] = subject
    msg.set_content(body)

    file_path = Path(filepath)
    if not file_path.exists():
        raise FileNotFoundError(f"Attachment not found: {filepath}")

    with open(file_path, "rb") as f:
        data = f.read()
    msg.add_attachment(
        data,
        maintype="application",
        subtype="octet-stream",
        filename=file_path.name,
    )

    if use_ssl:
        with smtplib.SMTP_SSL(smtp_host, smtp_port) as server:
            if smtp_user and smtp_pass:
                server.login(smtp_user, smtp_pass)
            server.send_message(msg)
    else:
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.ehlo()
            try:
                server.starttls()
                server.ehlo()
            except smtplib.SMTPException:
                pass
            if smtp_user and smtp_pass:
                server.login(smtp_user, smtp_pass)
            server.send_message(msg)


def main():
    parser = argparse.ArgumentParser(description="Send a file via SMTP as email attachment")
    parser.add_argument("--smtp-host", required=True)
    parser.add_argument("--smtp-port", type=int, default=587)
    parser.add_argument("--smtp-user", default=os.getenv("SMTP_USER", ""))
    parser.add_argument("--smtp-pass", default=os.getenv("SMTP_PASS", ""))
    parser.add_argument("--from", dest="sender", required=True)
    parser.add_argument("--to", dest="recipient", required=True)
    parser.add_argument("--subject", default="VWAP strategy file")
    parser.add_argument("--body", default="Attached is vwap_scalp_refactored.py")
    parser.add_argument(
        "--file",
        default=str(Path("/workspace/vwap_scalp_refactored.py")),
        help="Path to file to attach",
    )
    parser.add_argument("--ssl", action="store_true", help="Use SMTP SSL (e.g., port 465)")

    args = parser.parse_args()

    send_email(
        smtp_host=args.smtp_host,
        smtp_port=args.smtp_port,
        smtp_user=args.smtp_user,
        smtp_pass=args.smtp_pass,
        sender=args.sender,
        recipient=args.recipient,
        subject=args.subject,
        body=args.body,
        filepath=args.file,
        use_ssl=args.ssl,
    )


if __name__ == "__main__":
    main()