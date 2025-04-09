## Configurations Manual
This is the manual for package and experiment's configuration.

**data_dim** ***(int)*** ->  The dimensions of data. The properties number only (time-step is not included).

**in_len** ***(int)*** -> The time length of input. How long of the data chunk for input.

**out_len** ***(int)*** -> The time length of output. How long of the data chunk for output.

**out_len** ***(int)*** -> The time length of output. How long of the data chunk for output.

**seg_len** ***(int)*** -> The length of each segment in the data.

**window_size** ***(int)*** -> The size of the window for the sliding window approach.

**factor** ***(int)*** -> The factor used for scaling.

**model_dim** ***(int)*** -> The dimension of the model.

**feedforward_dim** ***(int)*** -> The dimension of the feedforward network.

**head_num** ***(int)*** -> The number of heads in the multi-head attention mechanism.

**layer_num** ***(int)*** -> The number of layers in the model.

**dropout** ***(float)*** -> The dropout rate for regularization.

**baseline** ***(bool)*** -> Whether to use a baseline model.

**learning_rate** ***(float)*** -> The learning rate for training.

**batch_size** ***(int)*** -> The batch size for training.

**csv_path** ***(str)*** -> The path to the CSV file containing data.

**split** ***(list)*** -> The split ratios for training, validation, and testing datasets.

**seed** ***(int)*** -> The random seed for reproducibility.

**accelerator** ***(str)*** -> The type of accelerator to use (e.g., "auto", "gpu").

**min_epochs** ***(int)*** -> The minimum number of epochs for training.

**max_epochs** ***(int)*** -> The maximum number of epochs for training.

**precision** ***(int)*** -> The precision level for training (e.g., 16, 32).

**patience** ***(int)*** -> The number of epochs to wait for improvement before early stopping.

**num_workers** ***(int)*** -> The number of workers to increase CPU loading.

**tracking_uri** ***(str)*** -> The uri for tracking server.

**registry_uri** ***(str)*** -> The uri for registry server.

**experiment_name** ***(str)*** -> The name of the experiment indicating different use cases.