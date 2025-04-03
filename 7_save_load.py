# Persist model state with saving, loading and running model predictions.

import torch
import torchvision.models as models

# PyTorch models store the learned parameters in an internal state dictionary, called state_dict. 
# These can be persisted via the torch.save method:
model = models.vgg16(weights='IMAGENET1K_V1')
torch.save(model.state_dict(), 'model_weights.pth')

# To load model weights, you need to create an instance of the same model first, and then load the parameters using load_state_dict() method.
model = models.vgg16()  # we do not specify ``weights``, i.e. create untrained model
model.load_state_dict(torch.load('model_weights.pth', weights_only=True))
model.eval()

# We might want to save the structure of this class together with the model, in which case we can pass model (and not model.state_dict()) to the saving function:
torch.save(model, 'model.pth')

# Saving state_dict is considered the best practice. 
# However, below we use weights_only=False because this involves loading the model, which is a legacy use case for torch.save.
model = torch.load('model.pth', weights_only=False)