"""CLI.  python3 main.py "cricket"            -> chat answer
         python3 main.py "research cricket"   -> cited, critiqued report
         python3 main.py                      -> interactive chat loop"""

import sys
import uuid

from dotenv import load_dotenv

# Must run before importing anything that reads env vars.
load_dotenv()

from agents import route  # noqa: E402
from pipeline import chat_reply, run_research  # noqa: E402


def handle(prompt: str, session_id: str) -> None:
    mode, text = route(prompt)
    if mode == "research":
        if not text:
            print('Tell me what to research, e.g. "research cricket".')
            return
        print("\n" + run_research(text, session_id) + "\n")
    else:
        for chunk in chat_reply(text, session_id):
            print(chunk, end="", flush=True)
        print("\n")


if __name__ == "__main__":
    session = f"cli-{uuid.uuid4()}"
    prompt = " ".join(sys.argv[1:]).strip()
    try:
        if prompt:
            handle(prompt, session)
            sys.exit(0)
        print('Chat with me, or start a message with "research" for a report. Ctrl+C to quit.\n')
        while True:
            prompt = input("you> ").strip()
            if prompt:
                handle(prompt, session)
    except (KeyboardInterrupt, EOFError):
        print()
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)
