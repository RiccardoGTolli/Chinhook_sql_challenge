#!/bin/bash
set -e

# Call the original entrypoint script in the background
docker-entrypoint.sh postgres &

# Wait for PostgreSQL to start
until pg_isready --quiet; do
  echo "Waiting for PostgreSQL to start..."
  sleep 1
done

# Execute your SQL script
psql -U $POSTGRES_USER -d $POSTGRES_DB -f /chinook_db/Chinook_PostgreSql.sql

# Keep the container running
wait $!
