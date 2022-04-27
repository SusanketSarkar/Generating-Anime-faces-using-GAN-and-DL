discriminator_model = nn.Sequential(
    # It takes in a (3X64X64) tensor

    # Layer:Convolution
    # Input: a (3X3X64) tansor
    # Output: a (64X32X32) tensor
    nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(64),
    nn.LeakyReLU(0.2, inplace=True),

    # Layer:Convolution
    # Input: a (64 X 32 X 32) tensor
    # Output: a (128 x 16 x 16) tensor
    nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(128),
    nn.LeakyReLU(0.2, inplace=True),

    # Layer:Convolution
    # Input: a (128 x 16 x 16) tensor
    # Output: a (256 x 8 x 8) tensor
    nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(256),
    nn.LeakyReLU(0.2, inplace=True),

    # Layer:Convolution
    # Input: a (256 x 8 x 8) tensor
    # Output: a (512 x 4 x 4) tensor
    nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(512),
    nn.LeakyReLU(0.2, inplace=True),

    # Layer:Convolution
    # Input: a (512 x 4 x 4) tensor
    # Output: a (1 X 1 X 1) tensor
    nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False),

    nn.Flatten(),
    nn.Sigmoid())