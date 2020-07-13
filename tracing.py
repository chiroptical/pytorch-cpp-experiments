import torch
import torchvision

model = torchvision.models.resnet18(pretrained=True)
model.fc = torch.nn.Linear(model.fc.in_features, 2)

example = torch.rand(1, 3, 224, 224)

trace = torch.jit.trace(model, example)

trace.save("traced_binary_resnet18.pt")
