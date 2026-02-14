"""Quick script to inspect PostgreSQL metadata."""

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
    cur = conn.cursor()

    # List tables
    cur.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema='public' "
        "ORDER BY table_name"
    )

    tables = [r[0] for r in cur.fetchall()]
    print(f"\n📋 Tables in '{settings.POSTGRES_DB}': {tables}\n")

    # Check papers table
    if "papers" in tables:
        cur.execute("SELECT COUNT(*) FROM papers")
        count = cur.fetchone()[0]
        print(f"📄 Papers count: {count}")

        if count > 0:
            cur.execute(
                "SELECT arxiv_id, title, parsing_status, published_date, categories "
                "FROM papers ORDER BY published_date DESC LIMIT 10"
            )
            columns = [desc[0] for desc in cur.description]
            print(f"\n{'─'*100}")
            print(f"{'arxiv_id':<20} {'title':<50} {'status':<10} {'date':<12} {'categories'}")
            print(f"{'─'*100}")
            for row in cur.fetchall():
                r = dict(zip(columns, row, strict=False))
                title = (r["title"][:47] + "...") if len(r["title"]) > 50 else r["title"]
                date = str(r["published_date"])[:10] if r["published_date"] else "N/A"
                cats = str(r["categories"])[:30] if r["categories"] else "N/A"
                print(
                    f"{r['arxiv_id']:<20} {title:<50} {r['parsing_status']:<10} {date:<12} {cats}"
                )
            print()

    # Check other tables
    for table in tables:
        if table != "papers":
            cur.execute(f"SELECT COUNT(*) FROM {table}")  # noqa: S608
            print(f"  {table}: {cur.fetchone()[0]} rows")

    conn.close()


if __name__ == "__main__":
    main()
