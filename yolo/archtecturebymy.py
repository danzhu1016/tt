import torch
import hiddenlayer as hl
from models.yolo import Model

# Load YOLOv5x model
model_path = 'models/yolov5x.yaml'  # YOLOv5x model configuration
model = Model(model_path, ch=3, nc=80)  # ch=3: number of input channels, nc=80: number of classes

# Load pretrained weights
checkpoint = torch.load('yolov5m.pt', map_location='cpu')
model.load_state_dict(checkpoint['model'].state_dict())

# Create a dummy input tensor
x = torch.randn(1, 3, 640, 640)  # Example input size for YOLOv5

# Build the graph
transforms = [hl.transforms.Prune('Constant')]  # Remove constant values
graph = hl.build_graph(model, x, transforms=transforms)

# Save the graph to a file
graph.save('yolov5x_structure_hl', format='png')

print("Model structure graph saved as yolov5x_structure_hl.png")
