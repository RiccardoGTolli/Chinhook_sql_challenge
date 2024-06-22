# Use the official PostgreSQL version as base
FROM postgres:15.1

# Copy SQL script into the container
COPY ./chinook_db/chinook_PostgreSql.sql chinook_db/chinook_PostgreSql.sql

# Add a custom script to execute SQL
COPY ./chinook_db/custom_entrypoint.sh /usr/local/bin/custom_entrypoint.sh
RUN chmod +x /usr/local/bin/custom_entrypoint.sh

# Set the custom entrypoint script
ENTRYPOINT ["/usr/local/bin/custom_entrypoint.sh"]
CMD ["postgres"]
