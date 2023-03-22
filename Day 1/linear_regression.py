   # Import libraries
import torch

# Define input and output data
X = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.float32)
y = torch.tensor([[2.0], [4.0], [6.0], [8.0]], dtype=torch.float32)

# Define model architecture
model = torch.nn.Linear(1, 1)

# Define loss function and optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Train the model
for epoch in range(1000):
    # Forward pass
    y_pred = model(X)
    loss = criterion(y_pred, y)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Print loss every 100 epochs
    if epoch % 100 == 0:
        print('Epoch: {}, Loss: {:.4f}'.format(epoch, loss.item()))

# Make a prediction
new_data = torch.tensor([[5.0]], dtype=torch.float32)
prediction = model(new_data)
print('Prediction: {:.4f}'.format(prediction.item()))
