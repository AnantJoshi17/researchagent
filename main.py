import sys

from dotenv import load_dotenv

# Must run before importing anything that reads env vars.
load_dotenv()

from pipeline import run_research  # noqa: E402

if __name__ == "__main__":
    topic = " ".join(sys.argv[1:]).strip()
    if not topic:
        topic = input("Research topic: ").strip()
    if not topic:
        print("No topic given. Usage: python3 main.py \"your topic\"")
        sys.exit(1)

    try:
        report = run_research(topic)
        print("\n" + "=" * 60 + "\n")
        print(report)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)
