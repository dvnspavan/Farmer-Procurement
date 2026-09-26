# Presentation Content

1. Title and student details
2. Procurement crowding problem
3. Objective: predict, recommend, queue and analyse
4. Architecture and dataset design
5. ML pipelines and leakage prevention
6. Evaluation metrics from training
7. Streamlit booking and token workflow
8. Recommendation logic versus ML prediction
9. Limitations and future collection of real data

# Viva Questions

**What makes this more than CRUD?** Predictions feed a capacity-aware recommendation before booking, and the queue dashboard captures operational states.

**Why not call the recommendation ML?** The models predict operational variables; the final selection is a transparent optimisation rule using those variables and capacity.

**How are waits operationally estimated?** Farmers ahead multiplied by average processing time, divided by active machines; this is compared with ML output.

**How is overfitting checked?** Metrics are calculated on a held-out test split, never the training rows.
