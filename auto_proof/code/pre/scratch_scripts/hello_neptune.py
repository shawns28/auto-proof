import neptune
import torch
from neptune.types import File
import numpy as np

# Create a Neptune run
run = neptune.init_run(
    project="shawns28/AutoProof", 
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJlOTA3ZDNjNS0wNGI5LTQ5OWEtYjRkYi05NmFlMzNjNzBkMGIifQ==", 
    name="sincere-oxpecker", 
    tags=["quickstart", "script"], 
    dependencies="infer", 
    monitoring_namespace="monitoring", 
)

# Log a single value
# Specify a field name ("seed") inside the run and assign a value to it
run["seed"] = 0.42

# Define a simple linear model
class LinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

# Create the model
model = LinearModel()

# Define a loss function and optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Create some sample data
x_data = torch.tensor([[1.0], [2.0], [3.0]])
y_data = torch.tensor([[2.0], [4.0], [6.0]])

# Training loop
for epoch in range(100):
    # Forward pass
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    run["loss"].append(loss)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch: {epoch}, Loss: {loss.item()}')

# Upload an image
run["single_image"].upload("sample.png")

# Upload a series of images

small_image_array = np.array([
    [0, 1, 0, 0],
    [1, 1, 1, 0],
    [0, 1, 0, 1],
    [0, 0, 1, 1]
])

run["image_series"].append(
        File.as_image(
            small_image_array
        ),  # You can upload arrays as images using Neptune's File.as_image() method
        name="small_image_array",
    )

# Stop the connection and synchronize the data with the Neptune servers
run.stop()