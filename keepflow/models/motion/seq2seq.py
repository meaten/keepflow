from typing import Dict

from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

from keepflow.models import ModelTemplate

class Seq2SeqModel(ModelTemplate):
    """Sequence-to-sequence model for human motion prediction"""

    def __init__(self,
                 cfg,
                 number_of_actions=15,
                 one_hot=True,
                 dropout=0.0):

        super().__init__(cfg)

        self.HUMAN_SIZE = 54
        self.input_size = self.HUMAN_SIZE + \
            number_of_actions if one_hot else self.HUMAN_SIZE
        

        print("One hot is ", one_hot)
        print("Input size is %d" % self.input_size)

        # Summary writers for train and test runs

        self.source_seq_len = self.obs_len
        self.target_seq_len = self.pred_len
        self.rnn_size = 1024
        self.dropout = dropout

        # === Create the RNN that will keep the state ===
        print('rnn_size = {0}'.format(self.rnn_size))
        self.cell = torch.nn.GRUCell(self.input_size, self.rnn_size)

        self.fc1 = nn.Linear(self.rnn_size, self.input_size)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

        self.optimizers = [self.optimizer]

        self.output_path = Path(cfg.OUTPUT_DIR)

    def predict(self, data_dict: Dict) -> Dict:
        encoder_inputs = data_dict["obs"]
        decoder_inputs = data_dict["decoder_input"]
        
        def loop_function(prev, i):
            return prev

        batchsize = encoder_inputs.shape[1]

        state = torch.zeros(batchsize, self.rnn_size).cuda()

        for i in range(self.source_seq_len-1):
            state = self.cell(encoder_inputs[i], state)

            state = F.dropout(state, self.dropout, training=self.training)

        outputs = []
        prev = None
        for i, inp in enumerate(decoder_inputs):
            if loop_function is not None and prev is not None:
                inp = loop_function(prev, i)

            inp = inp.detach()

            state = self.cell(inp, state)

            output = inp + \
                self.fc1(F.dropout(state, self.dropout, training=self.training))

            outputs.append(output.view([1, batchsize, self.input_size]))
            if loop_function is not None:
                prev = output

        outputs = torch.cat(outputs, 0)
        data_dict[("pred", 0)] = outputs
        return data_dict

    def update(self, data_dict: Dict) -> Dict:
        self.optimizer.zero_grad()
        data_dict = self.predict(data_dict)
        loss = torch.mean(torch.sum(torch.abs(data_dict["gt"] - data_dict["pred"]), dim=2))
        loss.backward()
        self.optimizer.step()

        return {"l1": loss.item()}
