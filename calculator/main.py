import sys

from dotenv import load_dotenv

load_dotenv()

from orchestrator import run


def main():
    if len(sys.argv) < 2:
        print('Usage: python main.py "2*(3+5)"')
        sys.exit(1)

    expression = sys.argv[1]
    print(f"Expression: {expression}\n", file=sys.stderr)
    result = run(expression)
    print(result)


if __name__ == "__main__":
    main()
