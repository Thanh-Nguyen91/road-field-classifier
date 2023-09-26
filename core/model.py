import timm
import torch
from torch import nn

###=====================================================================
class ClassifierRoadField(nn.Module):

    def __init__(
        self,
        backbone: str,
        pretrained: bool,
        target_size: 2,
    ):
        super().__init__()
        self.model = timm.create_model(model_name=backbone, pretrained=pretrained)
        self.pool2d = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(0.25)

        in_features = self.model.get_classifier().in_features
        self.fc = nn.Linear(in_features, target_size)

    def forward(self, x: torch.Tensor):
        x = self.model.forward_features(x)
        x = self.flatten(self.pool2d(x))
        x = self.dropout(x)
        pred = self.fc(x)

        return pred

###=====================================================================
if __name__ == "__main__":
    import torch
    from torchsummary import summary

    model = ClassifierRoadField(
        backbone="efficientnet_b0",
        pretrained=True,
        target_size=2,
    )

    summary(model, input_size=(3, 380, 380),device="cpu")

    x = torch.rand(2, 3, 380, 380)
    label = model(x)
    print(label)
