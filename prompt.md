Create focused tests before any implementation:

tests/test_data_io.py

Loads tiny fixtures from HSX+HNX → returns typed MultiIndex [date,ticker], sorted.

Duplicate (date,ticker) raises.

Missing required column raises.

Date filter [start,end] returns exact count.

Known split fixture keeps adjusted-close continuity.
