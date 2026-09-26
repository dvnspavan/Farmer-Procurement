"""Farmer Procurement Slot & Queue Management System — Flask web application.

Dual-role architecture:
- Dedicated Farmer Portal: Self-service registration, 1-click booking, live token tracker, digital pass with QR code.
- Dedicated Mandi Officer / Admin Portal: Real-time floor queue manager, ML forecasting, emergency slot overrides, CSV export.
- Mandi Waiting Hall Public Kiosk (/live-board): Real-time large screen queue display.
- Bilingual support (English + Telugu).

Version: 3.0 — Dual-Role Enterprise Edition.
"""
import csv
import datetime as dt
import functools
import io
import logging
import random

import joblib
import numpy as np
import pandas as pd
from flask import (
    Flask,
    Response,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)

from config import Config, SLOTS, SLOT_CAPACITY
from database.db import (
    add_booking,
    advance_token_status,
    authenticate_farmer,
    bookings,
    get_farmer,
    get_farmer_bookings,
    get_live_board_data,
    get_slot_overrides,
    get_token_details,
    init_db,
    register_farmer,
    set_slot_override,
    update_status,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config.from_object(Config)
init_db()

# Custom Jinja2 filters
@app.template_filter("enumerate")
def jinja_enumerate(iterable, start=0):
    """Make Python's built-in enumerate available in Jinja2 templates."""
    return enumerate(iterable, start=start)


# Language translations dictionary for English and Telugu
TRANSLATIONS = {
    "en": {
        "portal_name": "AgroProcure",
        "mandi_title": "Mandal Procurement Centre",
        "farmer_portal": "Farmer Portal",
        "officer_portal": "Mandi Officer Portal",
        "dashboard": "Dashboard",
        "book_slot": "Book a Slot",
        "queue_board": "Queue Board",
        "live_board": "Live Kiosk TV",
        "my_dashboard": "My Farmer Dashboard",
        "active_token": "Active Token",
        "no_active_token": "No active booking for today. Ready to schedule your crop delivery?",
        "booked": "Booked",
        "arrived": "Arrived at Mandi",
        "waiting": "Waiting in Bay",
        "processing": "At Weighbridge",
        "completed": "Completed",
        "cancelled": "Cancelled",
        "view_pass": "View Digital Pass",
        "simulate_sms": "Simulate SMS Alert",
        "book_new_slot": "Book Consignment Slot",
        "past_deliveries": "My Delivery History",
        "date": "Date",
        "slot": "Slot",
        "crop": "Crop",
        "quantity": "Quantity",
        "status": "Status",
        "action": "Action",
        "welcome": "Welcome",
        "logout": "Sign Out",
    },
    "te": {
        "portal_name": "ఆగ్రో ప్రొక్యూర్",
        "mandi_title": "మండల వ్యవసాయ కొనుగోలు కేంద్రం",
        "farmer_portal": "రైతు పోర్టల్",
        "officer_portal": "అధికారుల పోర్టల్",
        "dashboard": "డాష్‌బోర్డ్",
        "book_slot": "స్లాట్ బుక్ చేయండి",
        "queue_board": "క్యూ బోర్డు",
        "live_board": "లైవ్ టీవీ స్క్రీన్",
        "my_dashboard": "నా రైతు డాష్‌బోర్డ్",
        "active_token": "ప్రస్తుత టోకెన్",
        "no_active_token": "ఈరోజుకి ఎటువంటి క్రియాశీల టోకెన్ లేదు. కొత్త స్లాట్ బుక్ చేయండి.",
        "booked": "బుక్ చేయబడింది",
        "arrived": "కేంద్రానికి చేరుకున్నారు",
        "waiting": "వేచి ఉండే ప్రదేశంలో ఉన్నారు",
        "processing": "తూకం వేస్తున్నారు",
        "completed": "పూర్తయింది",
        "cancelled": "రద్దు చేయబడింది",
        "view_pass": "డిజిటల్ పాస్ చూడండి",
        "simulate_sms": "SMS నోటిఫికేషన్ చూడండి",
        "book_new_slot": "కొత్త స్లాట్ బుక్ చేసుకోండి",
        "past_deliveries": "గత డెలివరీల చరిత్ర",
        "date": "తేదీ",
        "slot": "సమయం",
        "crop": "పంట",
        "quantity": "పరిమాణం",
        "status": "స్థితి",
        "action": "చర్య",
        "welcome": "స్వాగతం",
        "logout": "లాగ్ అవుట్",
    },
}


@app.context_processor
def inject_global_context():
    """Inject language dictionary and current role into all templates."""
    lang = session.get("lang", "en")
    return {
        "lang": lang,
        "t": TRANSLATIONS.get(lang, TRANSLATIONS["en"]),
        "current_role": session.get("role"),
        "logged_in": session.get("logged_in", False),
        "now_date": dt.date.today().isoformat(),
        "now_time": dt.datetime.now().strftime("%H:%M"),
    }


@app.route("/set-lang/<lang>")
def set_lang(lang):
    """Switch language between English ('en') and Telugu ('te')."""
    if lang in ["en", "te"]:
        session["lang"] = lang
    return redirect(request.referrer or url_for("home"))


_MODELS_DIR = Config.MODELS_DIR
MODELS = {
    "arrival": joblib.load(_MODELS_DIR / "arrival_model.pkl"),
    "wait": joblib.load(_MODELS_DIR / "waiting_time_model.pkl"),
    "congestion": joblib.load(_MODELS_DIR / "congestion_model.pkl"),
}
HISTORICAL_DATA = pd.read_csv(Config.DATASET_PATH)


# ══ AUTHENTICATION DECORATORS ══

def login_required(view):
    """Ensure the user is authenticated (either admin or farmer)."""
    @functools.wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for("login", next=request.path))
        return view(*args, **kwargs)
    return wrapped


def admin_required(view):
    """Ensure the user is authenticated as an Admin/Officer."""
    @functools.wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("logged_in") or session.get("role") != "admin":
            flash("Officer login required to access administrative queue operations.", "warning")
            return redirect(url_for("login_admin", next=request.path))
        return view(*args, **kwargs)
    return wrapped


def farmer_required(view):
    """Ensure the user is authenticated as a registered Farmer."""
    @functools.wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("logged_in") or session.get("role") != "farmer":
            flash("Please sign in with your Farmer ID or Mobile Number.", "info")
            return redirect(url_for("login_farmer", next=request.path))
        return view(*args, **kwargs)
    return wrapped


# ══ AUTHENTICATION ROUTES ══

@app.route("/login")
def login():
    """Unified landing page with role selector tabs."""
    if session.get("logged_in"):
        if session.get("role") == "farmer":
            return redirect(url_for("farmer_dashboard"))
        return redirect(url_for("home"))
    
    active_tab = request.args.get("tab", "farmer")
    return render_template("login.html", active_tab=active_tab)


@app.route("/login/admin", methods=["GET", "POST"])
def login_admin():
    """Officer / Administrator authentication gate."""
    if session.get("logged_in") and session.get("role") == "admin":
        return redirect(url_for("home"))

    username = ""
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        if username == Config.LOGIN_USERNAME and password == Config.LOGIN_PASSWORD:
            session["logged_in"] = True
            session["role"] = "admin"
            session["user"] = username
            session["user_name"] = "Mandal Procurement Officer"
            flash("Officer authentication successful. Welcome back!", "success")
            return redirect(request.args.get("next") or url_for("home"))
        flash("Invalid officer username or password.", "error")

    return render_template("login.html", active_tab="admin", username=username)


@app.route("/login/farmer", methods=["GET", "POST"])
def login_farmer():
    """Farmer authentication via Farmer ID / Mobile Number and PIN."""
    if session.get("logged_in") and session.get("role") == "farmer":
        return redirect(url_for("farmer_dashboard"))

    identifier = ""
    if request.method == "POST":
        identifier = request.form.get("identifier", "").strip()
        pin = request.form.get("pin", "1234").strip()
        farmer = authenticate_farmer(identifier, pin)
        if farmer:
            session["logged_in"] = True
            session["role"] = "farmer"
            session["farmer_id"] = farmer["farmer_id"]
            session["farmer_name"] = farmer["farmer_name"]
            session["user"] = farmer["farmer_name"]
            session["phone"] = farmer.get("phone", "")
            session["village"] = farmer.get("village", "")
            session["crop"] = farmer.get("crop", "Paddy")
            flash(f"Welcome, {farmer['farmer_name']}! You are signed in to your Farmer Portal.", "success")
            return redirect(request.args.get("next") or url_for("farmer_dashboard"))
        flash("Farmer ID / Phone or PIN not recognized. Please check your credentials or register below.", "error")

    return render_template("login.html", active_tab="farmer", identifier=identifier)


@app.route("/register", methods=["GET", "POST"])
def register():
    """Farmer self-registration page."""
    if request.method == "POST":
        name = request.form.get("farmer_name", "").strip()
        phone = request.form.get("phone", "").strip()
        village = request.form.get("village", "").strip()
        district = request.form.get("district", "Krishna").strip()
        crop = request.form.get("crop", "Paddy").strip()
        pin = request.form.get("pin", "1234").strip()
        
        # Auto-generate Farmer ID if not provided
        farmer_id = request.form.get("farmer_id", "").strip()
        if not farmer_id:
            farmer_id = f"FRM-2026-{random.randint(100, 999)}"

        if not name or not phone:
            flash("Name and Mobile Number are required.", "error")
            return render_template("register.html", form=request.form)

        try:
            farmer = register_farmer(farmer_id, name, phone, village, district, crop, pin)
            session["logged_in"] = True
            session["role"] = "farmer"
            session["farmer_id"] = farmer["farmer_id"]
            session["farmer_name"] = farmer["farmer_name"]
            session["user"] = farmer["farmer_name"]
            session["phone"] = farmer.get("phone", "")
            session["village"] = farmer.get("village", "")
            session["crop"] = farmer.get("crop", "Paddy")
            flash(f"Registration successful! Your Farmer ID is {farmer_id}. Welcome to AgroProcure!", "success")
            return redirect(url_for("farmer_dashboard"))
        except Exception as exc:
            logger.error("Registration failed: %s", exc)
            flash("Registration could not be completed. Please try again.", "error")

    suggested_id = f"FRM-2026-{random.randint(100, 999)}"
    return render_template("register.html", suggested_id=suggested_id)


@app.post("/logout")
def logout():
    """Sign out user and clear session."""
    session.clear()
    flash("You have been signed out successfully.", "success")
    return redirect(url_for("login"))


# ══ ML FORECASTING CORE ══

def predict(date, slot, crop, quantity, vehicle="Tractor", already_booked=0):
    """Run ML pipelines for arrival, wait time and congestion forecasting."""
    features = pd.DataFrame([{
        "Procurement_Center": "Mandal Procurement Centre",
        "Crop_Type": crop,
        "Weather_Condition": "Clear",
        "Time_Slot": slot,
        "Month": date.month,
        "Historical_Farmer_Arrivals": 25 + already_booked,
        "Festival_or_Holiday": int(date.weekday() >= 5),
        "Quantity_Quintals": quantity,
        "Number_of_Vehicles": 1 if vehicle == "Auto" else 2,
        "Available_Machines": 3,
        "Number_of_Staff": 8,
        "Average_Processing_Time": 8 + quantity * 1.2,
        "Slot_Capacity": SLOT_CAPACITY[slot],
    }])

    arrivals = max(1, round(MODELS["arrival"].predict(features)[0]))
    wait_minutes = max(1, round(MODELS["wait"].predict(features)[0]))
    probabilities = MODELS["congestion"].predict_proba(features)[0]
    labels = MODELS["congestion"].classes_
    top_label = labels[np.argmax(probabilities)]
    probability_map = dict(zip(labels, (probabilities * 100).round(1)))

    return arrivals, wait_minutes, top_label, probability_map


def compare_slots(form):
    """Score every slot for the requested booking with dynamic capacity override support."""
    existing = bookings()
    booking_date = dt.date.fromisoformat(form["date"])
    quantity = float(form["quantity"])
    results = []
    overrides = get_slot_overrides(form["date"])

    for slot in SLOTS:
        already_booked = 0
        if len(existing):
            already_booked = len(
                existing[(existing.date.astype(str) == form["date"]) & (existing.slot == slot) & (~existing.status.isin(["Cancelled"]))]
            )

        arrivals, wait, congestion, _ = predict(
            booking_date, slot, form["crop"], quantity, form["vehicle"], already_booked
        )
        
        # Check if capacity has an emergency override
        if slot in overrides:
            capacity = overrides[slot]["capacity"]
            is_overridden = True
            override_reason = overrides[slot]["reason"]
        else:
            capacity = SLOT_CAPACITY[slot]
            is_overridden = False
            override_reason = ""

        overflow_penalty = max(0, arrivals - capacity) * 3
        score = wait + already_booked * 2 + overflow_penalty

        results.append({
            "slot": slot,
            "capacity": capacity,
            "booked": already_booked,
            "remaining": capacity - already_booked,
            "arrivals": arrivals,
            "wait": wait,
            "congestion": congestion,
            "score": score,
            "is_overridden": is_overridden,
            "override_reason": override_reason,
        })

    available = [row for row in results if row["remaining"] > 0]
    best = min(available, key=lambda row: row["score"]) if available else None
    return results, best


# ══ FARMER DASHBOARD & TOKEN PASS ══

@app.route("/farmer/dashboard")
@farmer_required
def farmer_dashboard():
    """Personalized Farmer Portal dashboard with active token tracker and booking history."""
    farmer_id = session.get("farmer_id")
    farmer_records = get_farmer_bookings(farmer_id)
    
    # Identify active token (most recent non-completed booking)
    active_booking = None
    for b in farmer_records:
        if b["status"] not in ["Completed", "Cancelled"]:
            active_booking = b
            break

    # Calculate queue position ahead of farmer
    queue_ahead = 0
    if active_booking:
        all_b = bookings()
        if len(all_b):
            same_day_active = all_b[
                (all_b.date.astype(str) == active_booking["date"]) &
                (all_b.slot == active_booking["slot"]) &
                (~all_b.status.isin(["Completed", "Cancelled"]))
            ]
            # Tokens ahead in the same slot
            my_token = active_booking["token_number"]
            ahead_df = same_day_active[same_day_active.token_number < my_token]
            queue_ahead = len(ahead_df)

    return render_template(
        "farmer_dashboard.html",
        active_booking=active_booking,
        queue_ahead=queue_ahead,
        bookings=farmer_records,
        farmer_id=farmer_id,
        farmer_name=session.get("farmer_name", "Farmer"),
        phone=session.get("phone", ""),
        village=session.get("village", ""),
        crop=session.get("crop", "Paddy"),
    )


@app.route("/token/<token_number>")
@login_required
def token_pass(token_number):
    """Digital intake pass with check-in QR code, ML wait predictions and document checklist."""
    token_data = get_token_details(token_number)
    if not token_data:
        flash("Token pass not found in system records.", "error")
        if session.get("role") == "farmer":
            return redirect(url_for("farmer_dashboard"))
        return redirect(url_for("queue"))

    # Compute queue vehicles ahead for this token
    all_b = bookings()
    queue_ahead = 0
    if len(all_b):
        same_slot = all_b[
            (all_b.date.astype(str) == str(token_data["date"])) &
            (all_b.slot == token_data["slot"]) &
            (~all_b.status.isin(["Completed", "Cancelled"]))
        ]
        queue_ahead = len(same_slot[same_slot.token_number < token_number])

    return render_template(
        "token_pass.html",
        token=token_data,
        queue_ahead=queue_ahead,
    )


@app.route("/api/sms-preview/<token_number>")
@login_required
def api_sms_preview(token_number):
    """Return simulated SMS notification for farmer communication."""
    token_data = get_token_details(token_number)
    if not token_data:
        return jsonify({"error": "Token not found"}), 404

    status = token_data.get("status", "Booked")
    if status == "Booked":
        action_msg = f"Your procurement slot is CONFIRMED for {token_data['date']} ({token_data['slot']}). Please bring your Aadhaar and Bank Passbook."
    elif status == "Arrived":
        action_msg = f"Your arrival is RECORDED. Please proceed to Waiting Bay #2. Estimated wait: ~{int(token_data.get('predicted_waiting_time') or 15)} mins."
    elif status in ["Processing", "At Weighbridge"]:
        action_msg = "Your token is NOW CALLED to Weighbridge Gate 1! Please position your vehicle immediately for weighing and moisture inspection."
    elif status == "Completed":
        action_msg = f"Consignment of {token_data['quantity']} Qtl {token_data.get('primary_crop','crop')} successfully ACCEPTED. MSP payment voucher is generated."
    else:
        action_msg = f"Status updated to {status}."

    sms_text = (
        f"🌾 [AGRO-PROCURE] Mandi Alert:\n"
        f"Dear {token_data['farmer_name']},\n"
        f"Token: #{token_data['token_number']}\n"
        f"{action_msg}\n"
        f"— Mandal Procurement Centre"
    )

    return jsonify({
        "success": True,
        "token_number": token_number,
        "farmer_name": token_data["farmer_name"],
        "phone": token_data.get("phone") or "9876543210",
        "status": status,
        "sms_text": sms_text,
    })


# ══ ADMIN / MANDI OFFICER ROUTES ══

@app.route("/")
@login_required
def home():
    """Main dashboard: redirects farmers to their portal, or renders officer analytics."""
    if session.get("role") == "farmer":
        return redirect(url_for("farmer_dashboard"))

    all_bookings = bookings()
    open_bookings = 0
    if len(all_bookings):
        open_bookings = len(all_bookings[~all_bookings.status.isin(["Completed", "Cancelled"])])

    return render_template(
        "home.html",
        record_count=len(HISTORICAL_DATA),
        booking_count=len(all_bookings),
        open_count=open_bookings,
        avg_wait=round(HISTORICAL_DATA.Waiting_Time_Minutes.mean()),
        monthly=HISTORICAL_DATA.groupby("Month").Actual_Farmer_Arrivals.mean().round(1).to_dict(),
    )


@app.route("/queue")
@admin_required
def queue():
    """Live floor token queue with one-click progression, timestamps and capacity override."""
    all_b = bookings().to_dict("records")
    today_str = dt.date.today().isoformat()
    overrides = get_slot_overrides(today_str)
    
    return render_template(
        "queue.html",
        rows=all_b,
        slots=SLOTS,
        slot_capacities=SLOT_CAPACITY,
        overrides=overrides,
        today=today_str,
    )


@app.post("/queue/advance")
@admin_required
def queue_advance():
    """One-click progression: Mark Arrived -> Call to Weighbridge -> Complete Inspection."""
    token = request.form.get("token")
    status = request.form.get("status")
    if token and status:
        advance_token_status(token, status)
        flash(f"Token {token} updated to '{status}'. Stage timestamp recorded.", "success")
    return redirect(url_for("queue"))


@app.post("/queue/status")
@admin_required
def queue_status():
    """Update a booking's status via dropdown."""
    token = request.form.get("token")
    status = request.form.get("status")
    if token and status:
        advance_token_status(token, status)
        flash(f"Token {token} status updated to {status}.", "success")
    return redirect(url_for("queue"))


@app.post("/admin/slot-override")
@admin_required
def slot_override():
    """Emergency capacity override for adverse weather, machine outage or staff shortage."""
    target_date = request.form.get("date", dt.date.today().isoformat())
    slot = request.form.get("slot")
    capacity = int(request.form.get("capacity", 25))
    reason = request.form.get("reason", "Operational adjustment")

    if slot in SLOTS:
        set_slot_override(target_date, slot, capacity, reason)
        flash(f"Capacity for slot {slot} on {target_date} set to {capacity} ({reason}).", "success")
    else:
        flash("Invalid slot specified.", "error")

    return redirect(request.referrer or url_for("queue"))


@app.route("/export/bookings")
@admin_required
def export_bookings():
    """One-click export of complete procurement queue to CSV for government record-keeping."""
    df = bookings()
    output = io.StringIO()
    writer = csv.writer(output)
    
    headers = [
        "Token Number", "Date", "Slot", "Farmer ID", "Farmer Name", "Mobile",
        "Village", "Quantity (Qtl)", "Vehicle Type", "Status",
        "Predicted Wait (Min)", "Predicted Congestion", "Arrival Time",
        "Weighbridge Start Time", "Completion Time", "Created At"
    ]
    writer.writerow(headers)

    for _, row in df.iterrows():
        writer.writerow([
            row.get("token_number", ""),
            row.get("date", ""),
            row.get("slot", ""),
            row.get("farmer_id", ""),
            row.get("farmer_name", ""),
            row.get("phone", ""),
            row.get("village", ""),
            row.get("quantity", ""),
            row.get("vehicle_type", ""),
            row.get("status", ""),
            row.get("predicted_waiting_time", ""),
            row.get("predicted_congestion", ""),
            row.get("arrival_time", ""),
            row.get("service_start_time", ""),
            row.get("completion_time", ""),
            row.get("created_at", ""),
        ])

    csv_data = output.getvalue()
    today_str = dt.date.today().isoformat()
    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename=agroprocure_queue_{today_str}.csv"}
    )


# ══ MANDI PUBLIC DISPLAY / WAITING HALL KIOSK ══

@app.route("/live-board")
def live_board():
    """Public TV display / Kiosk for the farmer waiting shed (No login required)."""
    data = get_live_board_data()
    return render_template("live_board.html", data=data)


# ══ SLOT BOOKING & RECOMMENDATION ENGINE ══

@app.route("/recommend", methods=["GET", "POST"])
@login_required
def recommend():
    """Compare all slots for consignment delivery and recommend optimal time window."""
    # Pre-populate farmer details if logged in as a farmer
    default_id = session.get("farmer_id", "") if session.get("role") == "farmer" else ""
    default_name = session.get("farmer_name", "") if session.get("role") == "farmer" else ""
    default_village = session.get("village", "Kalagotla") if session.get("role") == "farmer" else "Kalagotla"
    default_crop = session.get("crop", "Paddy") if session.get("role") == "farmer" else "Paddy"

    form = {
        "farmer_id": default_id,
        "farmer_name": default_name,
        "village": default_village,
        "crop": default_crop,
        "quantity": "10",
        "date": str(dt.date.today()),
        "vehicle": "Tractor",
    }
    rows, best = None, None

    if request.method == "POST":
        form.update(request.form)
        try:
            if not form["farmer_id"].strip():
                raise ValueError("Farmer ID is required.")
            if float(form["quantity"]) <= 0:
                raise ValueError("Quantity must be positive.")
            rows, best = compare_slots(form)
        except ValueError as exc:
            flash(str(exc), "error")

    return render_template("recommend.html", form=form, rows=rows, best=best, slots=SLOTS)


@app.post("/book")
@login_required
def book():
    """Confirm slot booking, issue token, and route to token pass or queue."""
    form = request.form.to_dict()
    token = None
    try:
        rows, _ = compare_slots(form)
        choice = next(row for row in rows if row["slot"] == form["slot"])
        if choice["remaining"] <= 0:
            raise ValueError("That slot is full. Please recheck available slots.")

        token = f"A-{len(bookings()) + 1:03d}"
        farmer = {
            "id": form["farmer_id"],
            "name": form.get("farmer_name") or "Farmer",
            "village": form.get("village", ""),
            "district": "Krishna",
            "crop": form["crop"],
            "variety": "Standard",
            "phone": form.get("phone", session.get("phone", "")),
            "pin": session.get("pin", "1234"),
        }
        booking_details = {
            "id": form["farmer_id"],
            "date": form["date"],
            "slot": form["slot"],
            "quantity": float(form["quantity"]),
            "vehicle": form["vehicle"],
            "token": token,
            "wait": choice["wait"],
            "congestion": choice["congestion"],
        }
        add_booking(farmer, booking_details)
        flash(f"Booking confirmed successfully! Your queue token is {token}.", "success")
        
        # Farmers go straight to their digital token pass
        if session.get("role") == "farmer":
            return redirect(url_for("token_pass", token_number=token))
        return redirect(url_for("token_pass", token_number=token))

    except (ValueError, KeyError, StopIteration) as exc:
        logger.warning("Booking failed: %s", exc)
        flash(str(exc), "error")
        return redirect(url_for("recommend"))


# ══ ML PREDICTIONS & ANALYTICS ══

@app.route("/predictions", methods=["GET", "POST"])
@admin_required
def predictions():
    """Standalone ML prediction laboratory for what-if operational analysis."""
    form = {
        "date": str(dt.date.today()), "slot": SLOTS[0],
        "crop": "Paddy", "quantity": "10",
    }
    output = None

    if request.method == "POST":
        form.update(request.form)
        try:
            arrivals, wait, congestion, probabilities = predict(
                dt.date.fromisoformat(form["date"]), form["slot"],
                form["crop"], float(form["quantity"]),
            )
            output = {
                "arrivals": arrivals,
                "wait": wait,
                "congestion": congestion,
                "probabilities": probabilities,
                "operational": round(max(0, arrivals - 1) * (8 + float(form["quantity"]) * 1.2) / 3),
            }
        except ValueError:
            flash("Enter valid values.", "error")

    return render_template("predictions.html", form=form, output=output, slots=SLOTS)


@app.route("/analytics")
@admin_required
def analytics():
    """Historical aggregate statistics and K-Means farmer clustering."""
    crop_qty = HISTORICAL_DATA.groupby("Crop_Type").Quantity_Quintals.sum().round(0).to_dict()
    slot_waits = HISTORICAL_DATA.groupby("Time_Slot").Waiting_Time_Minutes.mean().round(1).to_dict()
    cluster_df = (
        HISTORICAL_DATA
        .groupby("Cluster")[["Quantity_Quintals", "Waiting_Time_Minutes", "Actual_Farmer_Arrivals"]]
        .mean()
        .round(1)
        .reset_index()
    )
    return render_template(
        "analytics.html",
        crop=crop_qty,
        waits=slot_waits,
        clusters=cluster_df.to_dict("records"),
    )


@app.errorhandler(404)
def not_found(_error):
    return render_template("base.html"), 404


if __name__ == "__main__":
    app.run(debug=Config.DEBUG)
