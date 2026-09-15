import os
import random
from scipy.io import loadmat
from collections import defaultdict

from .oxford_pets import OxfordPets
from .utils import Datum, DatasetBase, read_json

"""
template = ['a photo of a {}, a type of flower.']
"""
template = ['a photo of a {}.']

class OxfordFlowers(DatasetBase):

    dataset_dir = 'oxford_flowers'

    def __init__(self, root, num_shots, setting="standard", seed=1):
        assert setting in ("standard", "base2new")
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        self.image_dir = os.path.join(self.dataset_dir, 'jpg')
        self.label_file = os.path.join(self.dataset_dir, 'imagelabels.mat')
        self.lab2cname_file = os.path.join(self.dataset_dir, 'cat_to_name.json')
        self.split_path = os.path.join(self.dataset_dir, 'split_zhou_OxfordFlowers.json')

        self.template = template

        # load default training, validation and test splits of the dataset
        train, val, test = OxfordPets.read_split(self.split_path, self.image_dir)

        # make it a few-shot dataset by subsampling both the training and validation splits 
        # NOTE: these will be overwritten by line 33, but leaving this here s.t. you know how to create
        # a new FSL dataset on the fly if needed
        n_shots_val = min(num_shots, 4)
        val = self.generate_fewshot_dataset(val, num_shots=n_shots_val)
        train = self.generate_fewshot_dataset(train, num_shots=num_shots)

        # load the preprocessed jsonl for the few-shot split according to the (seed, shots) combo
        # NOTE: u should download the shared pickles on drive first
        preprocessed_train = os.path.join(self.dataset_dir, "split_fewshot", f"shot_{num_shots}-seed_{seed}_train.jsonl")
        train = OxfordPets.load_preprocessed_jsonl(preprocessed_train)
        preprocessed_val = os.path.join(self.dataset_dir, "split_fewshot", f"shot_{num_shots}-seed_{seed}_val.jsonl")
        val = OxfordPets.load_preprocessed_jsonl(preprocessed_val)

        # subsample the classes in the sets into base/novel 
        # (for the test set, we return both base/novel test loader to compute both accs in a single script)
        if setting == "base2new":
            train, val, test_base, test_new = OxfordPets.base2new_split(train, val, test)
        elif setting == "standard":
            train, val, test_base, test_new = train, val, test, None        

        super().__init__(train_x=train, val=val, test=test_base, test_new=test_new)
        

        print(f"Number of Train, Val and Test classes = {len(self.classnames)}, {len(self.val_classnames)} and {len(self.test_classnames)}")
        if setting == "base2new":
            assert all([ct == cv for ct, cv in zip(self.classnames, self.val_classnames)]) # ensure train and val classes are the same
            assert not any([ct == cv for ct, cv in zip(self.classnames, self.test_new_classnames)]) # ensure train and test classes are different
    
    def read_data(self):
        tracker = defaultdict(list)
        label_file = loadmat(self.label_file)['labels'][0]
        for i, label in enumerate(label_file):
            imname = f'image_{str(i + 1).zfill(5)}.jpg'
            impath = os.path.join(self.image_dir, imname)
            label = int(label)
            tracker[label].append(impath)
        
        print('Splitting data into 50% train, 20% val, and 30% test')

        def _collate(ims, y, c):
            items = []
            for im in ims:
                item = Datum(
                    impath=im,
                    label=y-1, # convert to 0-based label
                    classname=c
                )
                items.append(item)
            return items

        lab2cname = read_json(self.lab2cname_file)
        train, val, test = [], [], []
        for label, impaths in tracker.items():
            random.shuffle(impaths)
            n_total = len(impaths)
            n_train = round(n_total * 0.5)
            n_val = round(n_total * 0.2)
            n_test = n_total - n_train - n_val
            assert n_train > 0 and n_val > 0 and n_test > 0
            cname = lab2cname[str(label)]
            train.extend(_collate(impaths[:n_train], label, cname))
            val.extend(_collate(impaths[n_train:n_train+n_val], label, cname))
            test.extend(_collate(impaths[n_train+n_val:], label, cname))
        
        return train, val, test