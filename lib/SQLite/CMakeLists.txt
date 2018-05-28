add_definitions(-DSQLITE_THREADSAFE=1)

COLMAP_ADD_LIBRARY(sqlite3
    sqlite3.h
    sqlite3ext.h
    sqlite3.c
)
