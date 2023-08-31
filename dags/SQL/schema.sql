-- create dataframe table
DROP TABLE IF EXISTS dataframe;
CREATE TABLE IF NOT EXISTS dataframe(
    useriD INTEGER, 
    StreamId BIGINT,
    StreamerName VARCHAR,
    StartTime REAL,
    EndTime REAL
);


-- load into dataframe table 

COPY dataframe FROM '/var/lib/postgresql/14/main/100k_a.csv' WITH CSV HEADER;



