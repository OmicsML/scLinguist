from transformers import PretrainedConfig


class HyenaConfig(PretrainedConfig):
    model_type = "hyena"
    def __init__(
        self,
        d_model=256,
        d_inner=None,
        use_bias=True,
        train_freq=True,
        max_seq_len=1024,
        emb_dim=5,
        n_layer=12,
        num_inner_mlps=2,
        hyena_order=2,
        short_filter_order=3,
        filter_order=64,
        activation_freq=1,
        embed_dropout=0.1, 
        hyena_dropout=0.0,
        hyena_filter_dropout=0.0,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        output_hidden_states=False,
        use_return_dict=False,
        **kwargs,
    ):
        self.d_model = d_model
        if d_inner is None:
            self.d_inner = 4 * d_model
        else:
            self.d_inner = d_inner
        self.use_bias = use_bias
        self.train_freq = train_freq
        self.max_seq_len = max_seq_len
        self.emb_dim = emb_dim
        self.n_layer = n_layer
        self.hyena_order = hyena_order
        self.filter_order = filter_order
        self.short_filter_order = short_filter_order
        self.activation_freq = activation_freq
        self.num_inner_mlps = num_inner_mlps
        self.embed_dropout = embed_dropout
        self.hyena_dropout = hyena_dropout
        self.hyena_filter_dropout = hyena_filter_dropout
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.output_hidden_states = output_hidden_states
        super().__init__(**kwargs)