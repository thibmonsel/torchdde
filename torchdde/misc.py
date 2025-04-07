import torch


class TupleTensorTransformer:
    """
    Transforms a PyTorch tensor representation of a tuple back into the
    original tuple.

    This class is designed to handle the specific structure of the 'aug_state'
    tuple described in the problem, which is commonly used in adjoint
    sensitivity analysis. It can rebuild the tuple given a flattened tensor
    and metadata about the original tensor shapes and dtypes.

    Attributes:
        original_shapes (list[torch.Size]): The shapes of the tensors in the
        original tuple.
        original_dtypes (list[torch.dtype]): The data types of the tensors
        in the original tuple.
        original_devices (list[torch.device]): The devices of the tensors
        in the original tuple.
        param_indices (list[int]):  The starting index within the flattened
        tensor where each parameter's adjoint begins.

    """

    def __init__(
        self, original_shapes, original_dtypes, original_devices, param_indices
    ):
        """
        Initializes the TupleTensorTransformer.

        Args:
            original_shapes (list[torch.Size]): The shapes of the tensors
            in the original tuple.
            original_dtypes (list[torch.dtype]): The data types of the tensors
            in the original tuple.
            original_devices (list[torch.device]): The devices of the tensors
            in the original tuple.
            param_indices (list[int]):  The starting index within the
            flattened tensor where each parameter's adjoint begins.
        """
        self.original_shapes = original_shapes
        self.original_dtypes = original_dtypes
        self.original_devices = original_devices
        self.param_indices = param_indices

    @classmethod
    def from_tuple(cls, tuple_data):
        """
        Creates a TupleTensorTransformer from a sample tuple.
        Analyzes the tuple to determine the shapes, dtypes, and devices of
        each element, which is crucial for reconstructing the tuple later.

        Args:
            tuple_data (tuple or list): A sample of the tuple to
            be transformed.

        Returns:
            TupleTensorTransformer: An initialized TupleTensorTransformer obj.
        """
        original_shapes = [item.shape for item in tuple_data]
        original_dtypes = [item.dtype for item in tuple_data]
        original_devices = [item.device for item in tuple_data]

        # Find the index of params (start from aug_state[3])
        param_indices = [sum(item.numel() for item in tuple_data[:3])]
        current_index = param_indices[0]

        for i in range(3, len(tuple_data)):
            current_index += tuple_data[i - 1].numel()
            param_indices.append(current_index)

        return cls(original_shapes, original_dtypes, original_devices, param_indices)

    def flatten(self, tuple_data):
        """
        Flattens the tuple into a single PyTorch tensor with size [1, N].
        Concatenates all the tensors in the tuple along a single dimension
        and reshapes the result.

        Args:
            tuple_data (tuple or list): The tuple to be flattened.

        Returns:
            torch.Tensor: A flattened tensor representing the tuple,
            reshaped to [1, N].
        """
        flat_list = [item.flatten() for item in tuple_data]
        concatenated_tensor = torch.cat(flat_list)
        return concatenated_tensor.reshape(1, -1)  # Reshape to [1, N]

    def unflatten(self, flat_tensor):
        """
        Reconstructs the original tuple from the flattened tensor.
        Splits the tensor based on the stored shapes, dtypes, and devices,
        and reshapes each part to match the original structure.

        Args:
            flat_tensor (torch.Tensor): The flattened tensor to be unflattened.

        Returns:
            tuple: The reconstructed tuple.
        """
        flat_tensor = flat_tensor.reshape(-1)  # Reshape back to 1D
        reconstructed_tuple = []
        current_index = 0
        for shape, dtype, device in zip(
            self.original_shapes, self.original_dtypes, self.original_devices
        ):
            num_elements = torch.Size(shape).numel()
            tensor_slice = flat_tensor[current_index : current_index + num_elements]
            reconstructed_tensor = tensor_slice.reshape(shape).to(dtype).to(device)
            reconstructed_tuple.append(reconstructed_tensor)
            current_index += num_elements
        return reconstructed_tuple
