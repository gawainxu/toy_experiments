import torch
from torchvision import transforms


import numpy as np

def reservoir(num_seen_examples: int, buffer_size: int) -> int:
    """
    Reservoir sampling algorithm.
    :param num_seen_examples: the number of seen examples
    :param buffer_size: the maximum buffer size
    :return: the target index if the current image is sampled, else -1
    """
    if num_seen_examples < buffer_size:
        return num_seen_examples

    rand = np.random.randint(0, num_seen_examples + 1)
    if rand < buffer_size:
        return rand
    else:
        return -1


def ring(num_seen_examples: int, buffer_portion_size: int, task: int) -> int:
    return num_seen_examples % buffer_portion_size + task * buffer_portion_size


class buffer():
    def __init__(self, buffer_size, device, n_tasks=None, mode="reservior"):
        assert mode in ["ring", "reservoir"]
        self.buffer_size = buffer_size
        self.device = device
        self.num_seen_exemples = 0
        self.functional_index = eval(mode)
        if mode == "ring":
            assert n_tasks is not None
            self.task_number = n_tasks
            self.buffer_portion_size = buffer_size // n_tasks
        self.attributes = ["exemples", "labels", "logits", "task_labels"]
        self.class_mapping = {}


    def to(self, device):
        self.device = device
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                setattr(self, attr_str, getattr(self, attr_str).to(device))
        return self


    def __len__(self):
        return min(self.num_seen_exemples, self.buffer_size)


    def init_tensors(self, exemples: torch.Tensor, labels: torch.Tensor, 
                     logits: torch.Tensor, task_labels: torch.Tensor) -> None:

        for attr_str in self.attributes:
            attr = eval(attr_str)
            if attr is not None and not hasattr(self, attr_str):
                typ = torch.int64 if attr_str.endswith("els") else torch.float32
                setattr(self, attr_str, torch.zeros((self.buffer_size,
                        *attr.shape[1:]), dtype=typ, device=self.device))


    def add_data(self, exemples, labels=None, logits=None, task_labels=None):

        if not hasattr(self, "exemples"):
            self.init_tensors(exemples, labels, logits, task_labels)

        for i in range(exemples.shape[0]):
            index = reservoir(self.num_seen_exemples, self.buffer_size)
            self.num_seen_exemples += 1
            if index >= 0:
                self.exemples[index] = exemples[i].to(self.device)
                if labels is not None:
                    self.labels[index] = labels[i].to(self.device)
                if logits is not None:
                    self.logits[index] = logits[i].to(self.device)
                if task_labels is not None:
                    self.task_labels[index] = task_labels[i].to(self.device)


    def delete_data(self, deleted_idx):

        for i in range(deleted_idx, self.num_seen_examples):
            self.exemples[i-1] = self.exemples[i]
            self.labels[i-1] = self.labels[i]

        self.num_seen_examples -= 1
        

    def get_data(self, size: int, transform: transforms=None, return_idx=False) -> Tuple:

        if size > min(self.num_seen_exemples, self.exemples.shape[0]):
            size = min(self.num_seen_exemples, self.exemples.shape[0])

        choice = np.random.choice(min(self.num_seen_examples, self.examples.shape[0]),
                                  size=size, replace=False)
        if transform is None:
            transform = lambda x: x

        ret_tuple = (torch.stack([transform(ee.cpu())
                            for ee in self.examples[choice]]).to(self.device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                ret_tuple += (attr[choice],)
        
        if not return_idx:
            return ret_tuple
        else:
            return (torch.tensor(choice).to(self.device), ) + ret_tuple

        return ret_tuple


    def get_data_by_index(self, indexes, transform: transforms=None) -> Tuple:
        """
        Returns the data by the given index.
        :param index: the index of the item
        :param transform: the transformation to be applied (data augmentation)
        :return:
        """
        if transform is None: transform = lambda x: x
        ret_tuple = (torch.stack([transform(ee.cpu())
                            for ee in self.examples[indexes]]).to(self.device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str).to(self.device)
                ret_tuple += (attr[indexes],)
        return ret_tuple

    
    def is_empty(self) -> bool:
        """
        Returns true if the buffer is empty, false otherwise.
        """
        if self.num_seen_examples == 0:
            return True
        else:
            return False


    def get_all_data(self, transform: transforms=None) -> Tuple:
        """
        Return all the items in the memory buffer.
        :param transform: the transformation to be applied (data augmentation)
        :return: a tuple with all the items in the memory buffer
        """
        if transform is None: transform = lambda x: x
        ret_tuple = (torch.stack([transform(ee.cpu())
                            for ee in self.examples]).to(self.device),)
        for attr_str in self.attributes[1:]:
            if hasattr(self, attr_str):
                attr = getattr(self, attr_str)
                ret_tuple += (attr,)
        return ret_tuple


    def empty(self) -> None:
        """
        Set all the tensors to None.
        """
        for attr_str in self.attributes:
            if hasattr(self, attr_str):
                delattr(self, attr_str)
        self.num_seen_examples = 0




        
        

        