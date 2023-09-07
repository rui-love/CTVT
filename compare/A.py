class GRUObservationCell(torch.nn.Module):
    """Implements discrete update based on the received observations."""

    def __init__(self, input_size, hidden_size, prep_hidden, bias=True):
        super().__init__()
        self.gru_d = torch.nn.GRUCell(prep_hidden * input_size, hidden_size, bias=bias)
        self.gru_debug = torch.nn.GRUCell(
            prep_hidden * input_size, hidden_size, bias=bias
        )

        ## prep layer and its initialization
        std = math.sqrt(2.0 / (4 + prep_hidden))
        self.w_prep = torch.nn.Parameter(std * torch.randn(input_size, 4, prep_hidden))
        self.bias_prep = torch.nn.Parameter(0.1 + torch.zeros(input_size, prep_hidden))

        self.input_size = input_size
        self.prep_hidden = prep_hidden
        self.var_eps = 1e-6

    def forward(self, h, p, X_obs, M_obs, i_obs):
        ## only updating rows that have observations
        p_obs = p[i_obs]
        mean, var = torch.chunk(p_obs, 2, dim=1)
        ## making var non-negative and also non-zero (by adding a small value)
        var = torch.abs(var) + self.var_eps
        error = (X_obs - mean) / torch.sqrt(var)

        ## log normal loss, over all observations
        loss = 0.5 * ((torch.pow(error, 2) + torch.log(var)) * M_obs).sum()

        ## TODO: try removing X_obs (they are included in error)
        gru_input = torch.stack([X_obs, mean, var, error], dim=2).unsqueeze(2)
        gru_input = torch.matmul(gru_input, self.w_prep).squeeze(2) + self.bias_prep
        gru_input.relu_()
        ## gru_input is (sample x feature x prep_hidden)
        gru_input = gru_input.permute(2, 0, 1)
        gru_input = (
            (gru_input * M_obs)
            .permute(1, 2, 0)
            .contiguous()
            .view(-1, self.prep_hidden * self.input_size)
        )

        temp = h.clone()
        temp[i_obs] = self.gru_d(gru_input, h[i_obs])
        h = temp

        return h, loss
