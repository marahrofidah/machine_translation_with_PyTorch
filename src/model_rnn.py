import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1, dropout=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, num_layers, 
                          batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0)
        
        self.fc = nn.Linear(hidden_size * 2, hidden_size)
        self.hidden_fc = nn.Linear(hidden_size * 2, hidden_size)
        
    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        
        outputs, hidden = self.gru(embedded, hidden)
        
        # Project bidirectional outputs to hidden_size
        outputs = self.fc(outputs)
        
        # Handle bidirectional hidden state
        if self.gru.bidirectional:
            # hidden shape: [num_layers * 2, batch, hidden_size]
            # We need to combine forward and backward hidden states
            batch_size = hidden.size(1)
            hidden = hidden.view(self.num_layers, 2, batch_size, self.hidden_size)
            # Take the last layer and combine forward + backward
            forward_hidden = hidden[-1, 0, :, :].unsqueeze(0)  # [1, batch, hidden_size]
            backward_hidden = hidden[-1, 1, :, :].unsqueeze(0)  # [1, batch, hidden_size]
            
            # Combine and project
            combined_hidden = torch.cat([forward_hidden, backward_hidden], dim=2)
            final_hidden = self.hidden_fc(combined_hidden.squeeze(0)).unsqueeze(0)
            
            # Expand to match num_layers if needed
            final_hidden = final_hidden.repeat(self.num_layers, 1, 1)
        else:
            final_hidden = hidden
        
        return outputs, final_hidden

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.energy = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)
        
    def forward(self, hidden, encoder_outputs, mask=None):
        batch_size = encoder_outputs.shape[0]
        seq_len = encoder_outputs.shape[1]
        
        # hidden: [batch, hidden_size] -> [batch, seq_len, hidden_size]
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        
        # Concatenate hidden and encoder outputs
        energy = torch.tanh(self.energy(torch.cat((hidden, encoder_outputs), dim=2)))
        
        # Calculate attention scores
        attention = self.v(energy)  # [batch, seq_len, 1]
        
        # Apply mask if provided
        if mask is not None:
            # Ensure mask has correct shape [batch, seq_len, 1]
            if len(mask.shape) == 2:
                mask = mask.unsqueeze(2)
            elif len(mask.shape) == 3 and mask.shape[1] == 1:
                mask = mask.transpose(1, 2)
            
            attention = attention.masked_fill(mask == 0, -1e10)
            
        return F.softmax(attention, dim=1)

class DecoderRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1, dropout=0.1):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.attention = Attention(hidden_size)
        self.gru = nn.GRU(embed_size + hidden_size, hidden_size, num_layers, 
                          batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc_out = nn.Linear(hidden_size * 2, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, hidden, encoder_outputs, mask=None):
        # x: [batch] -> [batch, 1]
        if len(x.shape) == 1:
            x = x.unsqueeze(1)
        
        # Embedding
        embedded = self.dropout(self.embedding(x))  # [batch, 1, embed_size]
        
        # Get hidden state for attention (use last layer)
        hidden_for_attention = hidden[-1] if len(hidden.shape) == 3 else hidden
        
        # Calculate attention
        attn_weights = self.attention(hidden_for_attention, encoder_outputs, mask)
        # attn_weights: [batch, seq_len, 1]
        
        # Calculate context vector
        attn_weights_transposed = attn_weights.transpose(1, 2)  # [batch, 1, seq_len]
        context = torch.bmm(attn_weights_transposed, encoder_outputs)  # [batch, 1, hidden_size]
        
        # Concatenate embedded input and context
        gru_input = torch.cat((embedded, context), dim=2)  # [batch, 1, embed_size + hidden_size]
        
        # GRU forward
        output, hidden = self.gru(gru_input, hidden)  # output: [batch, 1, hidden_size]
        
        # PERBAIKAN: Pastikan dimensi konsisten sebelum concatenation
        output_squeezed = output.squeeze(1)  # [batch, hidden_size]
        context_squeezed = context.squeeze(1)  # [batch, hidden_size]
        
        # Combine output and context for final prediction
        combined = torch.cat((output_squeezed, context_squeezed), dim=1)  # [batch, hidden_size * 2]
        prediction = self.fc_out(combined)  # [batch, vocab_size]
        
        return prediction, hidden, attn_weights.squeeze(2)

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.device = device
        
    def create_mask(self, src):
        # Create mask: 1 for non-padding tokens, 0 for padding
        mask = (src != self.src_pad_idx).float()
        return mask
        
    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        tgt_len = tgt.shape[1]
        tgt_vocab_size = self.decoder.fc_out.out_features
        
        # Initialize outputs tensor
        outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(self.device)
        
        # Encoder forward
        encoder_outputs, hidden = self.encoder(src)
        
        # Ensure hidden state has correct number of layers for decoder
        current_layers = hidden.shape[0]
        needed_layers = self.decoder.num_layers
        
        if current_layers != needed_layers:
            if needed_layers > current_layers:
                # Repeat hidden state if decoder has more layers
                repeat_times = needed_layers // current_layers
                remainder = needed_layers % current_layers
                
                repeated_hidden = hidden.repeat(repeat_times, 1, 1)
                if remainder > 0:
                    extra_hidden = hidden[:remainder]
                    hidden = torch.cat([repeated_hidden, extra_hidden], dim=0)
                else:
                    hidden = repeated_hidden
            else:
                # Take only the needed layers
                hidden = hidden[:needed_layers]
        
        # Create source mask
        mask = self.create_mask(src)  # [batch, seq_len]
        
        # Initialize decoder input (first token)
        dec_input = tgt[:, 0]  # [batch]
        
        # Decoder forward (skip first target token which is BOS)
        for t in range(1, tgt_len):
            dec_output, hidden, _ = self.decoder(dec_input, hidden, encoder_outputs, mask)
            
            outputs[:, t] = dec_output
            
            # Teacher forcing
            use_teacher_forcing = torch.rand(1).item() < teacher_forcing_ratio
            if use_teacher_forcing:
                dec_input = tgt[:, t]
            else:
                dec_input = dec_output.argmax(1)
                
        return outputs
    # INI JUGA RNN