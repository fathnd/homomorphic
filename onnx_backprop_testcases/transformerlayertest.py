import torch
import onnxruntime as ort
import random
import os
import numpy as np

seed = 4
os.environ["PL_GLOBAL_SEED"] = str(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.latent_dim = 256
        self.num_heads = 4
        self.ff_size=1024
        self.dropout=0.1
        self.activation="gelu"
        self.num_layers = 4

        self.root_seqTransEncoderLayer = torch.nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                               nhead=self.num_heads,
                                                               dim_feedforward=self.ff_size,
                                                               dropout=self.dropout,
                                                               activation=self.activation)

        #self.root_seqTransEncoder = torch.nn.TransformerEncoder(root_seqTransEncoderLayer,
        #                                                  num_layers=self.num_layers)
        

    def forward(self, inputs):
        xseq = inputs[0]
        x = xseq.detach().requires_grad_()
        with torch.enable_grad():

            #x = self.root_seqTransEncoder.norm1(x + self.root_seqTransEncoder._sa_block(x, src_mask, src_key_padding_mask, is_causal=is_causal))
            #x = self.root_seqTransEncoder.norm2(x + self.root_seqTransEncoder._ff_block(x))
            x = self.root_seqTransEncoderLayer(x)

            loss = x.sum()

            return torch.autograd.grad([loss], [x])[0]


mdl = Model()
for p in mdl.parameters():
    p.requires_grad_(False)


testvars = [torch.randn([20, 2, 256]) for i in range(10)]


results = [mdl([t]) for t in testvars]


from torch import override_nonsense


results_withoverrides = [mdl([t]) for t in testvars]


print([ torch.abs(a-b).max() for a,b in zip(results, results_withoverrides) ])


print("export model")
from torch import override_nonsense
torch.onnx.export(
    Model(),
    [testvars[0]],
    "modelthing.onnx",
    input_names=["xseq"],
    opset_version=17,
    output_names=["lossgrad"],
    verbose=True
)

ort_session = ort.InferenceSession("modelthing.onnx")

existing_names = [x.name for x in ort_session.get_inputs()]

# run it to see if we get the same results:
results_onnx = [ ort_session.run(
    None,
    {"xseq": t.numpy()}
)[0] for t in testvars]

