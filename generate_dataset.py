"""Create a synthetic historical dataset for academic demonstration."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
SLOTS = ["08:00-09:00", "09:00-10:00", "10:00-11:00", "11:00-12:00", "12:00-13:00", "14:00-15:00", "15:00-16:00", "16:00-17:00"]
CENTERS = ["Mandal Procurement Centre", "Rural Market Yard", "Cooperative Collection Centre"]
CROPS = {"Paddy": ["BPT-5204", "Sona Masuri"], "Maize": ["DHM-117", "Pioneer-3396"], "Cotton": ["Bt Cotton", "NCS-145"]}

def main(n=4000):
    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-01-01", "2025-12-31", freq="D")
    rows = []
    for i in range(n):
        date = rng.choice(dates); slot_i = int(rng.choice(len(SLOTS), p=[.15,.18,.16,.13,.08,.11,.11,.08]))
        crop = rng.choice(list(CROPS)); weather = rng.choice(["Clear", "Cloudy", "Rainy"], p=[.55,.27,.18])
        holiday = int(rng.random() < .08); dow = pd.Timestamp(date).dayofweek
        seasonal = 13 if pd.Timestamp(date).month in [10,11,12,1] else 0
        historical = max(5, int(22 + 13*(slot_i < 3) + seasonal + 5*(dow >= 5) - 9*(weather == "Rainy") + rng.normal(0,5)))
        actual = max(3, int(historical + rng.normal(0,4)))
        quantity = max(2, round(rng.gamma(4, 2.4) * (1.12 if crop == "Paddy" else 1), 2))
        machines = int(rng.integers(2,6)); staff = int(rng.integers(4,13)); vehicles = int(rng.integers(1,4))
        avg_process = round(8 + quantity*1.2 + rng.normal(0,2), 1)
        capacity = 20 if slot_i in [0,1,4,7] else 25
        load = actual/capacity + quantity/18 + vehicles/6 - machines*.12 - staff*.025
        wait = max(5, round(18 + load*42 + avg_process*0.8 + rng.normal(0,8), 1))
        congestion = "High" if wait > 90 or actual > capacity*1.35 else "Medium" if wait > 45 else "Low"
        rows.append([f"F{1000+i%900:04d}", pd.Timestamp(date).date(), rng.choice(CENTERS), rng.choice(["Lakshmi Nagar","Greenfield","Kisanpur","Rythu Colony"]), "Krishna", crop, rng.choice(CROPS[crop]), quantity, int(np.ceil(quantity*2)), rng.choice(["Tractor","Mini Truck","Auto"], p=[.45,.35,.2]), vehicles, machines, staff, SLOTS[slot_i], pd.Timestamp(date).day_name(), pd.Timestamp(date).month, historical, actual, avg_process, wait, round(avg_process+wait/4,1), congestion, weather, holiday, capacity, rng.choice(["Completed","Completed","Completed","Cancelled"])])
    columns = ["Farmer_ID","Date","Procurement_Center","Village","District","Crop_Type","Crop_Variety","Quantity_Quintals","Number_of_Bags","Vehicle_Type","Number_of_Vehicles","Available_Machines","Number_of_Staff","Time_Slot","Day_of_Week","Month","Historical_Farmer_Arrivals","Actual_Farmer_Arrivals","Average_Processing_Time","Waiting_Time_Minutes","Procurement_Duration_Minutes","Congestion_Level","Weather_Condition","Festival_or_Holiday","Slot_Capacity","Completed_Status"]
    pd.DataFrame(rows, columns=columns).to_csv(ROOT / "procurement_data.csv", index=False)
    print(f"Created {n} records: Synthetic Historical Procurement Dataset created for academic/project demonstration.")
if __name__ == "__main__": main()
