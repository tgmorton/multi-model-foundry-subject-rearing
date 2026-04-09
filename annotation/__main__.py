"""Allow running as `python -m annotation [--port 8643]`."""

import argparse
import json
import os

from . import db


def main():
    parser = argparse.ArgumentParser(
        description="Launch the Spanish null subject annotation server",
        usage="python -m annotation [--port PORT] [--host HOST]",
    )
    parser.add_argument(
        "--port", "-p", type=int, default=8643,
        help="Port to serve on (default: 8643)",
    )
    parser.add_argument(
        "--host", type=str, default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--seed-users", type=str, default=None,
        help='JSON list of users to seed, e.g. \'[{"username":"a","display_name":"Alice","role":"annotator"}]\'',
    )
    args = parser.parse_args()

    # Initialize database
    db.init_db()

    # Seed users from CLI arg or env var
    seed_json = args.seed_users or os.environ.get("ANNOTATION_SEED_USERS")
    if seed_json:
        users = json.loads(seed_json)
        db.seed_users(users)
        # Print tokens for seeded users
        for u in users:
            user = db.get_user_by_username(u["username"])
            if user:
                print(f"  {user['display_name']} ({user['username']}): token={user['token']}")

    import uvicorn
    from .server import app

    print(f"Annotation server on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
