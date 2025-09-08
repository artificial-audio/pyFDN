def db2mag(db):
    """Convert decibels to magnitude."""
    return 10 ** (db / 20.0)