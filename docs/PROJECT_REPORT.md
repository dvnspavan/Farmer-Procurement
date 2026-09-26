# Project Report: Farmer Procurement Slot & Queue Management Using Machine Learning

## Abstract
This project develops a local decision-support application for procurement centres. It forecasts arrivals, predicts expected waiting time and congestion, recommends available time slots, and records token queue operations. It uses a relationship-driven synthetic historical dataset for academic demonstration.

## Objectives
Reduce crowding, provide transparent slot recommendations, estimate waits, and provide administrators with an operational dashboard.

## Methodology
The data generator models higher waits from greater arrivals, quantities and vehicle load, while staff and machines reduce operational burden. Scikit-learn `ColumnTransformer` pipelines impute numerical data and encode categorical fields. Random Forest regression models predict arrivals and waiting time; a Random Forest classifier predicts congestion; K-Means identifies four operating patterns. SQLite stores farmer, booking and queue records.

## Evaluation
Run `python train_models.py` to create a held-out evaluation. The generated run reported arrival R² 0.869 and waiting-time R² 0.843; congestion accuracy 0.879. These values are calculated, not hard-coded, and can change when the data generator is altered.

## Conclusion and limitation
The prototype demonstrates an end-to-end ML-enabled workflow. Synthetic data is appropriate for classroom demonstration but should be replaced with centre-specific, consented operational data before real deployment.
