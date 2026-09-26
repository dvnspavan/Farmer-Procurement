"""SQLite persistence layer for farmers, bookings and the service queue.

All connections are opened via a context manager so they are always
closed (and committed/rolled back) correctly, even if an error occurs
mid-transaction.
"""
import sqlite3
from contextlib import contextmanager
from datetime import date

import pandas as pd

from config import Config

_SCHEMA = """
CREATE TABLE IF NOT EXISTS farmers (
    farmer_id TEXT PRIMARY KEY,
    farmer_name TEXT NOT NULL,
    village TEXT,
    district TEXT,
    crop TEXT,
    crop_variety TEXT,
    registration_date TEXT,
    phone TEXT,
    pin TEXT DEFAULT '1234'
);

CREATE TABLE IF NOT EXISTS bookings (
    booking_id INTEGER PRIMARY KEY AUTOINCREMENT,
    farmer_id TEXT NOT NULL REFERENCES farmers(farmer_id),
    date TEXT NOT NULL,
    slot TEXT NOT NULL,
    quantity REAL NOT NULL CHECK (quantity > 0),
    vehicle_type TEXT,
    token_number TEXT UNIQUE NOT NULL,
    status TEXT NOT NULL DEFAULT 'Booked',
    predicted_waiting_time REAL,
    predicted_congestion TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS queue (
    token_number TEXT PRIMARY KEY REFERENCES bookings(token_number),
    farmer_id TEXT NOT NULL,
    slot TEXT NOT NULL,
    status TEXT NOT NULL,
    arrival_time TEXT,
    service_start_time TEXT,
    completion_time TEXT
);

CREATE TABLE IF NOT EXISTS slot_overrides (
    date TEXT NOT NULL,
    slot TEXT NOT NULL,
    capacity INTEGER NOT NULL,
    reason TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (date, slot)
);

CREATE INDEX IF NOT EXISTS idx_bookings_date_slot ON bookings(date, slot);
CREATE INDEX IF NOT EXISTS idx_queue_status ON queue(status);
"""


@contextmanager
def get_connection():
    """Yield a SQLite connection, committing on success and closing always."""
    connection = sqlite3.connect(Config.DATABASE_PATH, check_same_thread=False)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def init_db() -> None:
    """Create tables and indexes if they do not already exist, and run migrations."""
    with get_connection() as conn:
        conn.executescript(_SCHEMA)

        # Check and apply migrations for existing SQLite databases
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(farmers)")
        farmer_columns = [row[1] for row in cursor.fetchall()]
        if "phone" not in farmer_columns:
            conn.execute("ALTER TABLE farmers ADD COLUMN phone TEXT")
        if "pin" not in farmer_columns:
            conn.execute("ALTER TABLE farmers ADD COLUMN pin TEXT DEFAULT '1234'")

        # Create phone index now that column is guaranteed to exist
        conn.execute("CREATE INDEX IF NOT EXISTS idx_farmers_phone ON farmers(phone)")

        # Ensure demo farmers have PINs and phones populated
        conn.execute("UPDATE farmers SET pin = '1234' WHERE pin IS NULL")
        conn.execute("UPDATE farmers SET phone = '9876543210' WHERE farmer_id = 'FRM-2026-200' AND (phone IS NULL OR phone = '')")

        # Seed initial demo farmer if none exists
        cursor.execute("SELECT COUNT(*) FROM farmers")
        if cursor.fetchone()[0] == 0:
            conn.execute(
                """
                INSERT INTO farmers (farmer_id, farmer_name, village, district, crop, crop_variety, registration_date, phone, pin)
                VALUES ('FRM-2026-101', 'Ramesh Kumar', 'Kalagotla', 'Krishna', 'Paddy', 'Standard', ?, '9876543210', '1234')
                """,
                (date.today().isoformat(),),
            )


def bookings() -> pd.DataFrame:
    """Return all bookings joined with farmer details and queue timestamps, ordered by schedule."""
    query = """
        SELECT b.*, f.farmer_name, f.village, f.phone,
               q.arrival_time, q.service_start_time, q.completion_time
        FROM bookings b
        LEFT JOIN farmers f USING (farmer_id)
        LEFT JOIN queue q USING (token_number)
        ORDER BY b.date DESC, b.slot ASC, b.token_number ASC
    """
    with get_connection() as conn:
        return pd.read_sql_query(query, conn)


def get_farmer(identifier: str):
    """Retrieve a single farmer by either farmer_id or phone number."""
    query = "SELECT * FROM farmers WHERE farmer_id = ? OR phone = ? LIMIT 1"
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query, (identifier.strip(), identifier.strip()))
        row = cursor.fetchone()
        return dict(row) if row else None


def authenticate_farmer(identifier: str, pin: str):
    """Verify farmer credentials by ID/Phone and PIN."""
    farmer = get_farmer(identifier)
    if not farmer:
        return None
    # Accept matching PIN or default '1234'
    stored_pin = farmer.get("pin") or "1234"
    if pin.strip() == stored_pin.strip():
        return farmer
    return None


def register_farmer(farmer_id: str, name: str, phone: str, village: str, district: str, crop: str, pin: str = "1234") -> dict:
    """Register a new farmer in the database."""
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO farmers (farmer_id, farmer_name, phone, village, district, crop, crop_variety, registration_date, pin)
            VALUES (?, ?, ?, ?, ?, ?, 'Standard', ?, ?)
            ON CONFLICT(farmer_id) DO UPDATE SET
                farmer_name = excluded.farmer_name,
                phone = excluded.phone,
                village = excluded.village,
                district = excluded.district,
                crop = excluded.crop,
                pin = excluded.pin
            """,
            (farmer_id, name, phone, village, district, crop, date.today().isoformat(), pin or "1234"),
        )
    return get_farmer(farmer_id)


def get_farmer_bookings(farmer_id: str) -> list:
    """Return all bookings belonging to a specific farmer."""
    query = """
        SELECT b.*, f.farmer_name, f.village, f.phone,
               q.arrival_time, q.service_start_time, q.completion_time
        FROM bookings b
        LEFT JOIN farmers f USING (farmer_id)
        LEFT JOIN queue q USING (token_number)
        WHERE b.farmer_id = ?
        ORDER BY b.date DESC, b.token_number DESC
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query, (farmer_id,))
        return [dict(row) for row in cursor.fetchall()]


def get_token_details(token_number: str):
    """Retrieve full details for a single token pass."""
    query = """
        SELECT b.*, f.farmer_name, f.village, f.district, f.phone, f.crop as primary_crop,
               q.arrival_time, q.service_start_time, q.completion_time
        FROM bookings b
        LEFT JOIN farmers f USING (farmer_id)
        LEFT JOIN queue q USING (token_number)
        WHERE b.token_number = ?
        LIMIT 1
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query, (token_number,))
        row = cursor.fetchone()
        return dict(row) if row else None


def add_booking(farmer: dict, booking: dict) -> None:
    """Upsert a farmer record and insert a new booking + queue entry."""
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO farmers
                (farmer_id, farmer_name, village, district, crop,
                 crop_variety, registration_date, phone, pin)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(farmer_id) DO UPDATE SET
                farmer_name=excluded.farmer_name,
                village=excluded.village,
                district=excluded.district,
                crop=excluded.crop,
                phone=COALESCE(excluded.phone, farmers.phone),
                pin=COALESCE(excluded.pin, farmers.pin)
            """,
            (
                farmer["id"], farmer["name"], farmer["village"],
                farmer.get("district", "Krishna"), farmer["crop"], farmer.get("variety", "Standard"),
                date.today().isoformat(), farmer.get("phone", ""), farmer.get("pin", "1234"),
            ),
        )
        conn.execute(
            """
            INSERT INTO bookings
                (farmer_id, date, slot, quantity, vehicle_type,
                 token_number, status, predicted_waiting_time,
                 predicted_congestion)
            VALUES (?, ?, ?, ?, ?, ?, 'Booked', ?, ?)
            """,
            (
                booking["id"], booking["date"], booking["slot"],
                booking["quantity"], booking["vehicle"], booking["token"],
                booking["wait"], booking["congestion"],
            ),
        )
        conn.execute(
            """
            INSERT INTO queue (token_number, farmer_id, slot, status)
            VALUES (?, ?, ?, 'Booked')
            ON CONFLICT(token_number) DO UPDATE SET status = 'Booked'
            """,
            (booking["token"], booking["id"], booking["slot"]),
        )


def update_status(token: str, status: str) -> None:
    """Update status of a token with automated timestamp tracking."""
    advance_token_status(token, status)


def advance_token_status(token: str, status: str) -> None:
    """Progress a booking's status and automatically record arrival/service/completion timestamps."""
    with get_connection() as conn:
        conn.execute("UPDATE bookings SET status = ? WHERE token_number = ?", (status, token))
        
        # Update queue status and relevant timestamp
        if status == "Arrived":
            conn.execute(
                """
                UPDATE queue
                SET status = ?, arrival_time = COALESCE(arrival_time, time('now', 'localtime'))
                WHERE token_number = ?
                """,
                (status, token),
            )
        elif status in ["Processing", "At Weighbridge"]:
            conn.execute(
                """
                UPDATE queue
                SET status = ?, service_start_time = COALESCE(service_start_time, time('now', 'localtime'))
                WHERE token_number = ?
                """,
                (status, token),
            )
        elif status == "Completed":
            conn.execute(
                """
                UPDATE queue
                SET status = ?, completion_time = COALESCE(completion_time, time('now', 'localtime'))
                WHERE token_number = ?
                """,
                (status, token),
            )
        else:
            conn.execute("UPDATE queue SET status = ? WHERE token_number = ?", (status, token))


def get_slot_overrides(target_date: str) -> dict:
    """Return dictionary of slot capacities that have been overridden for target_date."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT slot, capacity, reason FROM slot_overrides WHERE date = ?", (target_date,))
        return {row["slot"]: {"capacity": row["capacity"], "reason": row["reason"]} for row in cursor.fetchall()}


def set_slot_override(target_date: str, slot: str, capacity: int, reason: str) -> None:
    """Set or update an emergency capacity override for a slot."""
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO slot_overrides (date, slot, capacity, reason)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(date, slot) DO UPDATE SET
                capacity = excluded.capacity,
                reason = excluded.reason,
                created_at = CURRENT_TIMESTAMP
            """,
            (target_date, slot, capacity, reason),
        )


def get_live_board_data():
    """Fetch live data for the public waiting hall monitor / kiosk."""
    today_str = date.today().isoformat()
    query = """
        SELECT b.token_number, b.slot, b.vehicle_type, f.crop, b.quantity, b.status,
               b.predicted_waiting_time, f.farmer_name, f.village,
               q.arrival_time, q.service_start_time, q.completion_time
        FROM bookings b
        LEFT JOIN farmers f USING (farmer_id)
        LEFT JOIN queue q USING (token_number)
        WHERE b.date = ?
        ORDER BY b.slot ASC, b.token_number ASC
    """
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query, (today_str,))
        rows = [dict(r) for r in cursor.fetchall()]

        # If no bookings for today, grab recent bookings so kiosk has realistic demo data
        if not rows:
            cursor.execute(
                """
                SELECT b.token_number, b.slot, b.vehicle_type, f.crop, b.quantity, b.status,
                       b.predicted_waiting_time, f.farmer_name, f.village,
                       q.arrival_time, q.service_start_time, q.completion_time
                FROM bookings b
                LEFT JOIN farmers f USING (farmer_id)
                LEFT JOIN queue q USING (token_number)
                ORDER BY b.booking_id DESC LIMIT 15
                """
            )
            rows = [dict(r) for r in cursor.fetchall()]

    now_serving = [r for r in rows if r["status"] in ["Processing", "At Weighbridge"]]
    up_next = [r for r in rows if r["status"] in ["Waiting", "Arrived"]]
    booked = [r for r in rows if r["status"] == "Booked"]
    completed = [r for r in rows if r["status"] == "Completed"]

    return {
        "today": today_str,
        "now_serving": now_serving,
        "up_next": up_next[:6],
        "booked": booked[:8],
        "completed": completed[:5],
        "total_active": len(now_serving) + len(up_next) + len(booked),
        "total_completed": len(completed),
    }
