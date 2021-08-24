from torch import nn

from src.model.common import Conv
from torch import nn

from src.model.common import Conv


class Tokenizer(nn.Module):
    def __init__(
            self,
            image_size: int,
            token_size: int,
            deep: int
    ):
        super().__init__()

        self.up_scaling = Conv(
            in_channels=3,
            out_channels=token_size,
            kernel_size=1
        )

    def forward(self, x):
        up_scaled = self.up_scaling(x)
