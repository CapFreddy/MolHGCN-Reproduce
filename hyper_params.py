from argparse import Namespace


hyper_params = {
    'bace': Namespace(mean_fg_init=True,
                      use_cycle=True,
                      fully_connected_fg=False,
                      num_neurons=[],
                      nef_dp=0.,
                      reg_dp=0.1,
                      hid_dims=128),

    'bbbp': Namespace(mean_fg_init=True,
                      use_cycle=False,
                      fully_connected_fg=False,
                      num_neurons=[],
                      nef_dp=0.,
                      reg_dp=0.1,
                      hid_dims=32),

    'clintox': Namespace(mean_fg_init=False,
                         use_cycle=False,
                         fully_connected_fg=False,
                         num_neurons=[],
                         nef_dp=0.,
                         reg_dp=0.,
                         hid_dims=32),

    'sider': Namespace(mean_fg_init=True,
                       use_cycle=False,
                       fully_connected_fg=False,
                       num_neurons=[],
                       nef_dp=0.,
                       reg_dp=0.,
                       hid_dims=32),

    'tox21': Namespace(mean_fg_init=False,
                       use_cycle=False,
                       fully_connected_fg=False,
                       num_neurons=[],
                       nef_dp=0.2,
                       reg_dp=0.2,
                       hid_dims=32),

    # Not in paper, use the same hyper-parameters as Tox21.
    'toxcast': Namespace(mean_fg_init=False,
                         use_cycle=False,
                         fully_connected_fg=False,
                         num_neurons=[],
                         nef_dp=0.2,
                         reg_dp=0.2,
                         hid_dims=32),

    'hiv': Namespace(mean_fg_init=False,
                     use_cycle=False,
                     fully_connected_fg=False,
                     num_neurons=[],
                     nef_dp=0.2,
                     reg_dp=0.2,
                     hid_dims=32),

    'muv': Namespace(mean_fg_init=False,
                     use_cycle=False,
                     fully_connected_fg=False,
                     num_neurons=[],
                     nef_dp=0.2,
                     reg_dp=0.2,
                     hid_dims=32)
}
