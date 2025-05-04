def get_network(network_name):
    network_name = network_name.lower()
    if network_name == 'grconvnet':
        from .grconvnet import GenerativeResnet
        return GenerativeResnet
    else:
        raise NotImplementedError('Network {} is not implemented'.format(network_name))

