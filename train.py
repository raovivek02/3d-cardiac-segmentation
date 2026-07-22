import torch
from monai.networks.nets import UNet
from monai.losses import DiceCELoss
from dataset import get_acdc_data_dicts, get_transforms
from monai.data import Dataset, DataLoader

def train():
    # 1. Device Setup (Utilize the HPC cluster GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Data Loading Pipeline
    data_dir = "./data"
    data_dicts = get_acdc_data_dicts(data_dir)
    
    # Split into 80% train / 20% validation
    train_size = int(0.8 * len(data_dicts))
    train_dicts = data_dicts[:train_size]
    val_dicts = data_dicts[train_size:]

    transforms = get_transforms()
    train_ds = Dataset(data=train_dicts, transform=transforms)
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=2)

    # 3. Model Definition (3D U-Net configuration)
    model = UNet(
        spatial_dims=3,          # 3D spatial convolutions
        in_channels=1,           # Raw grayscale MRI channel
        out_channels=4,          # 4 output classes: background, RV, MYO, LV
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)

    # 4. Specialized Loss & Optimizer
    # Combination of Dice Loss (for spatial overlap) and Cross Entropy (for pixel-level stability)
    loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 5. Simple Training Loop
    print("Starting training process...")
    model.train()
    epochs = 40  # Scaled up for GPU training
    
    for epoch in range(epochs):
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data["image"].to(device), batch_data["label"].to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            if step % 5 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{step}/{len(train_loader)}], Loss: {loss.item():.4f}")
        
        print(f"Epoch {epoch+1} Complete. Average Loss: {epoch_loss / step:.4f}\n")

    # 6. Save the trained weights
    torch.save(model.state_dict(), "best_metric_model.pth")
    print("Model weights saved successfully to best_metric_model.pth")

if __name__ == "__main__":
    train()