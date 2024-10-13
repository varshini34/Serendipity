import torch
import torch.nn as nn


class Discriminator(nn.Module):

    def __init__(self, input_size=(3706, 3706)):
        super().__init__()

        self.model = nn.Sequential(
            
            nn.Linear(input_size[0] + input_size[1], 600),  
            nn.LayerNorm(600),
            nn.LeakyReLU(0.02),

            nn.Linear(600, 300),
            nn.LayerNorm(300),
            nn.LeakyReLU(0.02),

            nn.Linear(300, 1),
            nn.Sigmoid()
        )
        

        self.loss_function = nn.BCELoss()

        self.optimiser = torch.optim.SGD(self.parameters(), lr=0.0001)

        self.counter = 0
        self.progress = []
    
    def forward(self, vector_tensor, label_tensor):
        inputs = torch.cat((vector_tensor, label_tensor))
        #print("Size of concatenated input tensor:", inputs.size())
        return self.model(inputs)
    
    



class Generator(nn.Module):

    def __init__(self, input_size=(128, 3706), output_size=3706):
        super().__init__()

        self.model = nn.Sequential(
            
            nn.Linear(input_size[0] + input_size[1], 300), 
            nn.LayerNorm(300), 
            nn.LeakyReLU(0.02),

            nn.Linear(300, 600),
            nn.LayerNorm(600),
            nn.LeakyReLU(0.02),

            nn.Linear(600, output_size),
            nn.Sigmoid()
        )

        self.optimiser = torch.optim.SGD(self.parameters(), lr=0.0001)

        self.counter = 0
        self.progress = []

    def forward(self, seed_tensor, label_tensor):
        inputs = torch.cat((seed_tensor, label_tensor))
        return self.model(inputs)
