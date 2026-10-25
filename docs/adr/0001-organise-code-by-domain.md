# Organise code by domain, with one shared schema

Code lives in one package per domain under `src/` (`dataset`, `shelf`, `prediction`, `scoring`, `vision`, `labelling`, `runs`, `tasks`, `store`), and each domain owns its rules, its queries and its API router. `config/db/` keeps only the shared infrastructure: the engine, the SQLAlchemy schema and the Alembic migrations. This replaces the earlier rule that everything database-related lives in `config/db/`, which spread one concept (an area's status, the latest completed run) across `config/db/queries.py`, `config/db/save_run.py` and `src/api/`.

`config/db/schema.py` imports its column types from the domains that own them: `StockLevel` and `STOCK_VALUES` from `src/shelf/status.py`, `TaskKind` from `src/tasks/kind.py`. Those modules import nothing from `config/db/`, so the schema never imports itself back.

## Considered Options

- **All database code in `config/db/`, domains hold only pure logic.** Rejected: a domain's rules and the queries that apply them change together, and splitting them is what made the review find the same rule in SQL and in Python.
- **Each domain also owns its table models.** Rejected: the tables share composite foreign keys across domains (predictions reference runs, frames and areas), and Alembic autogenerate needs one metadata. One schema module is simpler than registering models from every domain.
