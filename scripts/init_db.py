"""Initialize the database schema from init.sql (for local PostgreSQL, not Docker)."""

import psycopg2

from src.config.settings import settings


def main():
    conn = psycopg2.connect(
        host=settings.POSTGRES_HOST,
        port=settings.POSTGRES_PORT,
        user=settings.POSTGRES_USER,
        password=settings.POSTGRES_PASSWORD,
        dbname=settings.POSTGRES_DB,
    )
    conn.autocommit = True
    cur = conn.cursor()

    sql = open("docker/postgres/init.sql").read()  # noqa: SIM115

    # Filter out CREATE DATABASE and GRANT statements (need superuser / no tx)
    filtered_lines = []
    for line in sql.split("\n"):
        stripped = line.strip().upper()
        if stripped.startswith("CREATE DATABASE") or stripped.startswith("GRANT ALL"):
            continue
        filtered_lines.append(line)

    schema_sql = "\n".join(filtered_lines)
    cur.execute(schema_sql)
    print("Schema created successfully!")

    # Verify
    cur.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema='public' ORDER BY table_name"
    )
    tables = [r[0] for r in cur.fetchall()]
    print(f"Tables: {tables}")

    conn.close()


if __name__ == "__main__":
    main()
