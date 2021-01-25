class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder  # 约等于 DecoderLayer
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        
    def forward(self, src, tgt, src_mask, tgt_mask):
        """
        Take in and process masked src and target sequences.
        src: (B, Ti)
        tgt: (B, To)
        src_mask: (B, 1, Ti)
        tgt_mask: (B, To, To)
        """
        return self.decode(self.encode(src, src_mask), src_mask,
                            tgt, tgt_mask)
    
    def encode(self, src, src_mask):
        """
        src_embed: src (B, Ti) -> (B, Ti, H)
        tgt_embed: tgt (B, To) -> (B, To, H)
        """
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, memory, src_mask, tgt, tgt_mask):
        """
        memory: decoder output (B, Ti, H)
        """
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class Generator(nn.Module):
    """
    Define standard linear + softmax generation step.
    """
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


def clones(module, N):
    """
    Produce N identical layers.
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    """
    Core encoder is a stack of N layers.
    """
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        """
        Pass the input (and mask) through each layer in turn.
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    See http://jmlr.org/papers/v15/srivastava14a.html for dropout detail
    and https://arxiv.org/abs/1512.03385 for residual connection detail.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, sublayer):
        """
        Apply residual connection to any sublayer with the same size.
        """
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.module):
    """
    Encoder block consist of two sub-layers (define below): 
    - multi-head attention (self-attention) 
    - feed forward.
    """
    def __init__(self, size, self_attn=MultiHeadAttention, feed_forward=PositionwiseFeedForward, dropout):
        super(EncoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        
    def forward(self, x, mask):
        """
        Encoder block.
        x: (B, Ti, H).
        mask: (B, 1, Ti).
        """
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


def attention(query, key, value, mask=None, dropout=None):
        """
        q,k,v: (B, head, T, d_k)
        mask: (B, 1, 1, Tk) or (B, 1, T, T)

        Compute `Scale Dot-Product Attention`.
        
        :params query: linear projected query maxtrix, Q in above figure right
        :params key: linear projected key maxtrix, k in above figure right
        :params value: linear projected value maxtrix, v in above figure right
        :params mask: sub-sequence mask
        :params dropout: rate of dropout
        """
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)  # (B, head, Tq, Tk)
        if mask is not None:
            scores = scores.mask_fill(mask == 0, -1e9)
        # softmax matrix
        # 纵轴表示query, 横轴表示key, 每一行softmax，padding的地方softmax前取极小值
        p_attn = F.softmax(scores, dim=-1)  # (B, head, Tq, Tk)
        if dropout is not None:
            p_attn = self.dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn  # (B, head, Tq, d_k)


class MultiHeadAttention(nn.Module):
    """
    Build Multi-Head Attention sub-layer.
    """
    def __init__(self, h, d_model, dropout=0.1):
        """
        :params h: int, number of heads
        :params d_model: model size
        :params dropout: rate of dropout
        """
        super(MultiHeadAtention, self).__init__()
        assert d_model % h == 0
        
        # and d_v = d_k = d_model / h = 64
        self.d_k = d_model // h
        self.h = h
        # following K, Q, V and `Concat`, so we need 4 linears
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        """
        q,k,v: (B, T, d_model).
        mask: (B, 1, Ti).

        Implement Multi-Head Attention.
        
        :params query: query embedding matrix, Q in above figure left
        :params key: key embedding matrix, K in above figure left
        :params value value embedding matrix, V in above figure left
        :params mask: sub-sequence mask
        """
        if mask is not None:
            # same mask applied to all heads
            mask = mask.unsequeeze(1)  # (B, 1, 1, Ti)
        n_batch = query.size(0)        # B
        # 1. Do all the linear projections in batch from d_model to h x d_k
        query, key, value = [l(x).view(n_batch, -1, self.h, self.d_k).transpose(1, 2)  # (B, Ti, head, d_k)
                             for l, x in zip(self.linears, (query, key, value))]       # (B, head, Ti, d_k)
        # 2. Apply attention on all the projected vectors in batch
        x, self.attn = self.attention(query, key, value, mask=mask)
        # x: (B, head, Tq, dk)
        # attn: (B, head, Tq, Tk), 对于每一个head, 纵轴表示query, 横轴表示key, 每一行softmax，padding的地方softmax前取极小值
        # 3. `Concat` using a view and apply a final linear
        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k) # (B, T, d_model)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.module):
    """
    Implements FFN equation.
    """
    def __init__(self, d_model, d_ff, dropout):
        super(PositionwiseFeedForward, self).__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, f_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.w2(self.dropout(F.relu(self.w1(x))))


class PositionEncoding(nn.module):
    """
    Implements Position Encoding.
    """
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionEncoding, self).__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the position encodings once in log space
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * 
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
        def forward(self, x):
            x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
            return self.dropout(x)


def make_model(src_vocab=11, tgt_vocab=11, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    """
    Construct Transformer model.
    
    :params src_vocab: source language vocabulary
    :params tgt_vocab: target language vocabulary
    :params N: number of encoder or decoder stacks
    :params d_model: dimension of model input and output
    :params d_ff: dimension of feed forward layer
    :params h: number of attention head
    :params dropout: rate of dropout
    """
    c = copy.deepcopy
    attn = MultiHeadAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff)
    position = PositionEncoding(d_model, dropout)
    model = EncoderDecoder(
                Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout),N),   
                Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
                nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
                nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
                Generator(d_model, tgt_vocab)
    )
    
    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model


def subsequent_mask(size): # 生成下三角矩阵，下三角和主对角线为1, non future mask
    """
    Mask out subsequent position.
    """
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0  # (1, C, C)


class Batch:
    """
    Object for holding a batch of data with mask during training.
    """
    def __init__(self, src, tgt=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsequeeze(-2)  # (B, 1, 10)
        if tgt is not None:
            self.tgt = tgt[;, :-1]  # (B, 10-1)
            self.tgt_y = tgt[:, 1:] # (B, 1)
            self.tgtg_mask = self.make_std_mask(self.tgt, pad)
            self.ntokens = (self.tgt_y != pad).data.sum()
            

    def make_std_mask(tgt, pad=0):  # (B, 10-1)
        """
        Create a mask to hide padding and future words.
        """
        tgt_mask = (tgt != pad).unsequeeze(-2)  # (B, 1, 10-1)   non padding mask
        tgt_mask = tgt_mask & Variable(         # 这个运算，non padding mask 第1维复制;
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)  # non future mask 第0维复制
        )
        return tgt_mask  # (B, 10-1, 10-1)
        # 可能
        # 举例一个batch
        # tgt_mask，原来mask就这么简单，下面这个就说明最后一个是padding的
        # [[1, 0, 0, 0, 0],
        #  [1, 1, 0, 0, 0],
        #  [1, 1, 0, 0, 0],
        #  [1, 1, 1, 1, 0],
        #  [0, 0, 0, 0, 0]]

        """
        a = tensor([[1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0]])  # (B, T)
        b = tensor([[1, 0, 0, 0, 0],
                    [1, 1, 0, 0, 0],
                    [1, 1, 1, 0, 0],
                    [1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1]])  # (T, T)
        a_ = a.unsqueeze(-2)           # (B, 1, T)
        b_ = b.unsqueeze(0)            # (B, T, T)

        a_ & b_ = tensor([[[1, 0, 0, 0, 0],
                         [1, 1, 0, 0, 0],
                         [1, 1, 1, 0, 0],
                         [1, 1, 1, 1, 0],
                         [1, 1, 1, 1, 0]],

                        [[1, 0, 0, 0, 0],
                         [1, 1, 0, 0, 0],
                         [1, 1, 1, 0, 0],
                         [1, 1, 1, 1, 0],
                         [1, 1, 1, 1, 1]],

                        [[0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0]]])
        """



def data_gen(V=11, batch=30, nbatches=20):
    """
    Generate random data for src-tgt copy task.
    """
    for i in range(nbatches):
        data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10)))  # (B, 10)
        data[:, 0] = 1
        src = Variable(data, requires_grad=False)
        tgt = Variable(data, requires_grad=False)
        yield Batch(src, tgt, 0)
        # 返回 src 还是 src (B, Li)
        # tgt (B, Lo) 被分成2, 1是tgt=原tgt[:,:-1] (B, Lo-1), 2是tgt_y=原tgt[:,-1:] (B, 1)
        # 以及返回 src_mask (B, 1, Li), tgt_mask (B, Lo-1, Lo-1)


class SimleLossCompute:
    """
    A simple loss compute and train function.
    """
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt
        
    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1)) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.data[0] * norm


def run_epoch(data_iter, model, loss_compute):
    """
    Standard training and logging function.
    """
    start_time = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src,       # (B, Li)
                            batch.tgt,       # (B, Lo)
                            batch.src_mask,  # (B,  1, Li)
                            batch.tgt_mask)  # (B, Lo, Lo)
        loss = loss_compute(out, batch.tgt_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 50 == 1:
            elapsed = time.time() - start_time
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" 
                  % (i, loss / batch.ntokens, tokens / elapsed))
            start_time = time.time()
            token = 0
        return total_loss / total_tokens


V = 11
criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
model = make_model(V, V, N=2)
model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400,
                    torch.optim.Adam(model.parameters(), lr=0, 
                                     betas=(0.9, 0.98), eps=1e-9))
for epoch in range(10):
    model.train()
    run_epoch(data_gen(V, 30, 20), model, 
             SimpleLossCompute(model.generator, criterion, model_opt))
    model.evel()
    print(run_epoch(data_gen(V, 30, 5), model, 
                   SimpleLossCompute(model.generator, criterion, None)))












class Decoder(nn.module):
    """
    Generic N layer decoder with masking.
    """
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.module):
    """
    Encoder block consist of three sub-layers (define below): 
    - multi-head attention (self-attention) 
    - encoder multi-head attention 
    - feed forward.
    """
    def __init__(self, size, self_attn=MultiHeadAttention, src_attn=MultiHeadAttention, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn =  src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
        
    def forward(self, x, memory, src_mask, tgt_mask):
        """
        x: pre_target (B, To, H)
        memory: encoder output (B, Ti, H)
        src_
        Decoder block.
        """
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)





def greedy_decode(model, src, src_mask, max_len=10, start_symbol=1):
    """
    src: (B, Ti)
    scr_mask: (B, 1, Ti)
    """
    memory = model.encode(src, src_mask)  # (B=1, Ti, d_model)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src.data)  # (1, 1)
    for i in range(max_len-1):   # loop 9 times
        out = model.decode(memory, src_mask, 
                          Variable(ys), 
                          Variable(subsequent_mask(ys.size(1))
                                   .type_as(src.data)))
        # out: the same as ys's shape
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat([ys,
                       torch.ones(1, 1).type_as(src.data).fill_(next_word)],
                       dim=1)
    return ys
    
model.eval()
src = Variable(torch.LongTensor([[1,2,3,4,5,6,7,8,9,10]]))  # (1, 10)
src_mask = Variable(torch.ones(1, 1, 10))                   # (1, 1, 10)
print(greedy_decode(model, src, src_mask, max_len=10, start_symbol=1))











