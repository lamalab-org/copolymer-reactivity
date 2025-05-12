import pandas as pd
import plotly.graph_objects as go

# Load the CSV file
df = pd.read_csv("../output/xgboost_predictions_for_error_analysis.csv")  # Replace with your actual file path

# Define the features for coloring
features = [
    'temperature',
    'solvent_logp', 'delta_HOMO_LUMO_AA', 'delta_HOMO_LUMO_AB',
    'delta_HOMO_LUMO_BB', 'delta_HOMO_LUMO_BA', 'polytype_emb_1',
    'polytype_emb_2', 'method_emb_1', 'method_emb_2'
]

# Define axes
x_vals = df["true_r1r2"]
y_vals = df["pred_r1r2"]

# Create a frame for each feature
frames = []
for feature in features:
    scatter = go.Scatter(
        x=x_vals,
        y=y_vals,
        mode="markers",
        marker=dict(
            color=df[feature],
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title=feature)
        ),
        name=feature,
        text=[f"{feature} = {v:.3f}" for v in df[feature]]
    )
    frames.append(go.Frame(data=[scatter], name=feature))

# Create the initial figure
fig = go.Figure(
    data=[frames[0].data[0]],
    layout=go.Layout(
        title="Parity Plot: true_r1r2 vs. pred_r1r2 (colored by feature)",
        xaxis=dict(title="true_r1r2", range=[0, 10]),
        yaxis=dict(title="pred_r1r2", range=[0, 10]),
        sliders=[{
            "active": 0,
            "pad": {"t": 50},
            "steps": [{
                "args": [[feature], {"frame": {"duration": 0, "redraw": True},
                                     "mode": "immediate"}],
                "label": feature,
                "method": "animate"
            } for feature in features]
        }]
    ),
    frames=frames
)


fig.show()
